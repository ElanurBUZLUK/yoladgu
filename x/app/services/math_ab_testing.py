import numpy as np
import random
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
import statistics
from enum import Enum

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Statistical tests will be limited.")

from app.models.math_profile import MathProfile
from app.models.question import Question

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Deney durumu"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ANALYZED = "analyzed"


class StatisticalTest(Enum):
    """İstatistiksel test türü"""
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"


@dataclass
class ExperimentVariant:
    """Deney varyantı"""
    name: str
    description: str
    parameters: Dict[str, Any]
    traffic_percentage: float
    is_control: bool = False


@dataclass
class Experiment:
    """A/B test deneyi"""
    id: str
    name: str
    description: str
    variants: List[ExperimentVariant]
    status: ExperimentStatus
    start_date: datetime
    end_date: Optional[datetime] = None
    target_metrics: List[str] = None
    statistical_test: StatisticalTest = StatisticalTest.T_TEST
    significance_level: float = 0.05
    minimum_sample_size: int = 100
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class ExperimentResult:
    """Deney sonucu"""
    experiment_id: str
    variant_name: str
    sample_size: int
    metric_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    is_significant: bool = False


class MathABTesting:
    """Matematik A/B test framework'ü"""
    
    def __init__(self):
        self.config = {
            # Deney parametreleri
            "default_significance_level": 0.05,
            "default_power": 0.8,
            "minimum_sample_size": 50,
            "maximum_experiment_duration": 30,  # gün
            
            # Randomization parametreleri
            "randomization_seed": 42,
            "sticky_assignment": True,  # Kullanıcı aynı varyantta kalır
            
            # Analiz parametreleri
            "confidence_level": 0.95,
            "bootstrap_samples": 1000,
            "effect_size_threshold": 0.1,
            
            # Monitoring parametreleri
            "check_interval_hours": 6,
            "early_stopping_threshold": 0.01,
            "safety_monitoring": True,
        }
        
        # Aktif deneyler
        self.active_experiments: Dict[str, Experiment] = {}
        
        # Kullanıcı varyant atamaları
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {experiment_id: variant_name}
        
        # Deney sonuçları
        self.experiment_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}  # experiment_id -> variant_name -> results
        
        # Random seed ayarla
        random.seed(self.config["randomization_seed"])
        np.random.seed(self.config["randomization_seed"])
    
    async def create_experiment(
        self, 
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        target_metrics: List[str],
        statistical_test: str = "t_test",
        significance_level: float = 0.05,
        minimum_sample_size: int = 100,
        duration_days: int = 14
    ) -> Experiment:
        """Yeni deney oluştur"""
        
        experiment_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{name.lower().replace(' ', '_')}"
        
        # Varyantları oluştur
        experiment_variants = []
        total_percentage = 0.0
        
        for i, variant_data in enumerate(variants):
            is_control = variant_data.get("is_control", i == 0)
            traffic_percentage = variant_data.get("traffic_percentage", 100.0 / len(variants))
            
            variant = ExperimentVariant(
                name=variant_data["name"],
                description=variant_data["description"],
                parameters=variant_data["parameters"],
                traffic_percentage=traffic_percentage,
                is_control=is_control
            )
            
            experiment_variants.append(variant)
            total_percentage += traffic_percentage
        
        # Traffic percentage'leri normalize et
        if total_percentage != 100.0:
            for variant in experiment_variants:
                variant.traffic_percentage = (variant.traffic_percentage / total_percentage) * 100.0
        
        # Deney oluştur
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            variants=experiment_variants,
            status=ExperimentStatus.DRAFT,
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=duration_days),
            target_metrics=target_metrics,
            statistical_test=StatisticalTest(statistical_test),
            significance_level=significance_level,
            minimum_sample_size=minimum_sample_size,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Deneyi kaydet
        self.active_experiments[experiment_id] = experiment
        self.experiment_results[experiment_id] = {variant.name: [] for variant in experiment_variants}
        
        logger.info(f"✅ Experiment created: {experiment_id} - {name}")
        
        return experiment
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Deneyi başlat"""
        
        if experiment_id not in self.active_experiments:
            logger.error(f"❌ Experiment not found: {experiment_id}")
            return False
        
        experiment = self.active_experiments[experiment_id]
        experiment.status = ExperimentStatus.ACTIVE
        experiment.updated_at = datetime.utcnow()
        
        logger.info(f"🚀 Experiment started: {experiment_id}")
        return True
    
    async def pause_experiment(self, experiment_id: str) -> bool:
        """Deneyi duraklat"""
        
        if experiment_id not in self.active_experiments:
            logger.error(f"❌ Experiment not found: {experiment_id}")
            return False
        
        experiment = self.active_experiments[experiment_id]
        experiment.status = ExperimentStatus.PAUSED
        experiment.updated_at = datetime.utcnow()
        
        logger.info(f"⏸️ Experiment paused: {experiment_id}")
        return True
    
    async def complete_experiment(self, experiment_id: str) -> bool:
        """Deneyi tamamla"""
        
        if experiment_id not in self.active_experiments:
            logger.error(f"❌ Experiment not found: {experiment_id}")
            return False
        
        experiment = self.active_experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.utcnow()
        experiment.updated_at = datetime.utcnow()
        
        logger.info(f"✅ Experiment completed: {experiment_id}")
        return True
    
    async def assign_variant(
        self, 
        experiment_id: str,
        user_id: str
    ) -> Optional[str]:
        """Kullanıcıya varyant ata"""
        
        if experiment_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.ACTIVE:
            return None
        
        # Sticky assignment kontrolü
        if self.config["sticky_assignment"]:
            if user_id in self.user_assignments and experiment_id in self.user_assignments[user_id]:
                return self.user_assignments[user_id][experiment_id]
        
        # Random varyant atama
        variant_name = self._random_variant_assignment(experiment.variants)
        
        # Atamayı kaydet
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        self.user_assignments[user_id][experiment_id] = variant_name
        
        return variant_name
    
    async def record_experiment_event(
        self, 
        experiment_id: str,
        user_id: str,
        variant_name: str,
        event_type: str,
        event_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Deney olayını kaydet"""
        
        if experiment_id not in self.active_experiments:
            return False
        
        if experiment_id not in self.experiment_results:
            return False
        
        if variant_name not in self.experiment_results[experiment_id]:
            return False
        
        # Olay verisi
        event_record = {
            "user_id": user_id,
            "variant_name": variant_name,
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": timestamp or datetime.utcnow()
        }
        
        # Sonucu kaydet
        self.experiment_results[experiment_id][variant_name].append(event_record)
        
        return True
    
    async def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Deneyi analiz et"""
        
        if experiment_id not in self.active_experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.active_experiments[experiment_id]
        
        if experiment.status not in [ExperimentStatus.COMPLETED, ExperimentStatus.ACTIVE]:
            return {"error": "Experiment not in analyzable state"}
        
        # Varyant sonuçlarını hesapla
        variant_results = {}
        for variant in experiment.variants:
            variant_data = self.experiment_results[experiment_id][variant.name]
            
            if not variant_data:
                continue
            
            # Metrikleri hesapla
            metrics = self._calculate_variant_metrics(variant_data, experiment.target_metrics)
            
            # Güven aralıkları hesapla
            confidence_intervals = self._calculate_confidence_intervals(metrics, experiment.target_metrics)
            
            variant_results[variant.name] = {
                "sample_size": len(variant_data),
                "metrics": metrics,
                "confidence_intervals": confidence_intervals
            }
        
        # İstatistiksel testler
        statistical_tests = {}
        if len(variant_results) >= 2:
            control_variant = next((v for v in experiment.variants if v.is_control), experiment.variants[0])
            control_name = control_variant.name
            
            for variant in experiment.variants:
                if variant.name == control_name:
                    continue
                
                test_results = self._perform_statistical_test(
                    experiment_id, control_name, variant.name, experiment
                )
                statistical_tests[f"{control_name}_vs_{variant.name}"] = test_results
        
        # Deney durumunu güncelle
        experiment.status = ExperimentStatus.ANALYZED
        experiment.updated_at = datetime.utcnow()
        
        analysis_result = {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "status": experiment.status.value,
            "variant_results": variant_results,
            "statistical_tests": statistical_tests,
            "recommendations": self._generate_recommendations(variant_results, statistical_tests),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"📊 Experiment analyzed: {experiment_id}")
        
        return analysis_result
    
    async def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Deney durumunu al"""
        
        if experiment_id not in self.active_experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.active_experiments[experiment_id]
        
        # Varyant istatistikleri
        variant_stats = {}
        for variant in experiment.variants:
            variant_data = self.experiment_results[experiment_id][variant.name]
            variant_stats[variant.name] = {
                "sample_size": len(variant_data),
                "traffic_percentage": variant.traffic_percentage,
                "is_control": variant.is_control
            }
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "start_date": experiment.start_date.isoformat(),
            "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
            "variant_stats": variant_stats,
            "total_participants": sum(len(data) for data in self.experiment_results[experiment_id].values())
        }
    
    async def list_experiments(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Deneyleri listele"""
        
        experiments = []
        
        for experiment in self.active_experiments.values():
            if status_filter and experiment.status.value != status_filter:
                continue
            
            experiments.append({
                "id": experiment.id,
                "name": experiment.name,
                "status": experiment.status.value,
                "start_date": experiment.start_date.isoformat(),
                "end_date": experiment.end_date.isoformat() if experiment.end_date else None,
                "variant_count": len(experiment.variants),
                "target_metrics": experiment.target_metrics
            })
        
        return experiments
    
    def _random_variant_assignment(self, variants: List[ExperimentVariant]) -> str:
        """Random varyant atama"""
        
        # Traffic percentage'lere göre atama
        rand_value = random.random() * 100.0
        cumulative_percentage = 0.0
        
        for variant in variants:
            cumulative_percentage += variant.traffic_percentage
            if rand_value <= cumulative_percentage:
                return variant.name
        
        # Fallback
        return variants[0].name
    
    def _calculate_variant_metrics(
        self,
        variant_data: List[Dict[str, Any]],
        target_metrics: List[str]
    ) -> Dict[str, float]:
        """Varyant metriklerini hesapla"""
        
        metrics = {}
        
        for metric in target_metrics:
            metric_values = []
            
            for event in variant_data:
                if metric in event["event_data"]:
                    metric_values.append(event["event_data"][metric])
            
            if metric_values:
                metrics[metric] = {
                    "mean": np.mean(metric_values),
                    "median": np.median(metric_values),
                    "std": np.std(metric_values),
                    "min": np.min(metric_values),
                    "max": np.max(metric_values),
                    "count": len(metric_values)
                }
            else:
                metrics[metric] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0
                }
        
        return metrics
    
    def _calculate_metrics(
        self,
        events: List[Dict[str, Any]],
        target_metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Metrikleri hesapla"""
        
        metrics = {}
        
        for metric in target_metrics:
            metric_values = []
            
            for event in events:
                if metric in event:
                    metric_values.append(event[metric])
            
            if metric_values:
                metrics[metric] = {
                    "mean": np.mean(metric_values),
                    "median": np.median(metric_values),
                    "std": np.std(metric_values),
                    "min": np.min(metric_values),
                    "max": np.max(metric_values),
                    "count": len(metric_values)
                }
            else:
                metrics[metric] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0
                }
        
        return metrics
    
    def _calculate_confidence_intervals(
        self, 
        metrics: Dict[str, Dict[str, float]],
        target_metrics: List[str]
    ) -> Dict[str, Tuple[float, float]]:
        """Güven aralıkları hesapla"""
        
        confidence_intervals = {}
        
        for metric in target_metrics:
            if metric in metrics:
                mean = metrics[metric]["mean"]
                std = metrics[metric]["std"]
                count = metrics[metric]["count"]
                
                if count > 1:
                    # 95% confidence interval
                    margin_of_error = 1.96 * (std / np.sqrt(count))
                    confidence_intervals[metric] = (mean - margin_of_error, mean + margin_of_error)
                else:
                    confidence_intervals[metric] = (mean, mean)
            else:
                confidence_intervals[metric] = (0.0, 0.0)
        
        return confidence_intervals
    
    def _calculate_confidence_interval(
        self,
        values: List[float],
        confidence_level: float
    ) -> Tuple[float, float]:
        """Tek bir değer listesi için güven aralığı hesapla"""
        
        if not values:
            return (0.0, 0.0)
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        count = len(values)
        
        if count > 1:
            # Z-score for confidence level
            if confidence_level == 0.95:
                z_score = 1.96
            elif confidence_level == 0.99:
                z_score = 2.576
            else:
                z_score = 1.96  # Default to 95%
            
            margin_of_error = z_score * (std / np.sqrt(count))
            return (mean - margin_of_error, mean + margin_of_error)
        else:
            return (mean, mean)
    
    def _perform_statistical_test(
        self, 
        experiment_id: str,
        control_variant: str,
        treatment_variant: str,
        experiment: Experiment
    ) -> Dict[str, Any]:
        """İstatistiksel test uygula"""
        
        control_data = self.experiment_results[experiment_id][control_variant]
        treatment_data = self.experiment_results[experiment_id][treatment_variant]
        
        if not control_data or not treatment_data:
            return {"error": "Insufficient data"}
        
        # Primary metric için test
        primary_metric = experiment.target_metrics[0] if experiment.target_metrics else "accuracy"
        
        control_values = [
            event["event_data"][primary_metric] 
            for event in control_data 
            if primary_metric in event["event_data"]
        ]
        treatment_values = [
            event["event_data"][primary_metric] 
            for event in treatment_data 
            if primary_metric in event["event_data"]
        ]
        
        if not control_values or not treatment_values:
            return {"error": "No data for primary metric"}
        
        # T-test
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available for statistical tests"}
        
        try:
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values, ddof=1) + 
                                 (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)) / 
                                (len(control_values) + len(treatment_values) - 2))
            
            effect_size = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std
            
            is_significant = p_value < experiment.significance_level
            
            return {
                "test_type": "t_test",
                "primary_metric": primary_metric,
                "control_mean": np.mean(control_values),
                "treatment_mean": np.mean(treatment_values),
                "t_statistic": t_stat,
                "p_value": p_value,
                "effect_size": effect_size,
                "is_significant": is_significant,
                "control_sample_size": len(control_values),
                "treatment_sample_size": len(treatment_values)
            }
        
        except Exception as e:
            return {"error": f"Statistical test failed: {str(e)}"}
    
    def _perform_statistical_test_values(
        self,
        control_values: List[float],
        treatment_values: List[float],
        test_type: StatisticalTest,
        experiment: Experiment
    ) -> Dict[str, Any]:
        """Değer listeleri için istatistiksel test uygula"""
        
        if not control_values or not treatment_values:
            return {"error": "Insufficient data"}
        
        # T-test
        if not SCIPY_AVAILABLE:
            return {"error": "scipy not available for statistical tests"}
        
        try:
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values, ddof=1) + 
                                 (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)) / 
                                (len(control_values) + len(treatment_values) - 2))
            
            if pooled_std > 0:
                effect_size = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std
            else:
                effect_size = 0.0
            
            is_significant = p_value < experiment.significance_level
            
            return {
                "test_type": "t_test",
                "control_mean": np.mean(control_values),
                "treatment_mean": np.mean(treatment_values),
                "t_statistic": t_stat,
                "p_value": p_value,
                "effect_size": effect_size,
                "is_significant": is_significant,
                "control_sample_size": len(control_values),
                "treatment_sample_size": len(treatment_values)
            }
            
        except Exception as e:
            return {"error": f"Statistical test failed: {str(e)}"}
    
    def _generate_recommendations(
        self, 
        variant_results: Dict[str, Any],
        statistical_tests: Dict[str, Any]
    ) -> List[str]:
        """Öneriler oluştur"""
        
        recommendations = []
        
        # Sample size kontrolü
        for variant_name, results in variant_results.items():
            sample_size = results["sample_size"]
            if sample_size < 50:
                recommendations.append(f"Variant {variant_name} has small sample size ({sample_size})")
        
        # İstatistiksel anlamlılık
        for test_name, test_result in statistical_tests.items():
            if "error" not in test_result:
                if test_result["is_significant"]:
                    effect_size = test_result["effect_size"]
                    if abs(effect_size) > 0.5:
                        recommendations.append(f"Large effect size detected in {test_name} (d={effect_size:.3f})")
                    elif abs(effect_size) > 0.2:
                        recommendations.append(f"Medium effect size detected in {test_name} (d={effect_size:.3f})")
                    else:
                        recommendations.append(f"Small but significant effect in {test_name} (d={effect_size:.3f})")
                else:
                    recommendations.append(f"No significant difference detected in {test_name}")
        
        # Güven aralığı kontrolü
        for variant_name, results in variant_results.items():
            for metric, ci in results["confidence_intervals"].items():
                ci_width = ci[1] - ci[0]
                if ci_width > 0.5:  # Geniş güven aralığı
                    recommendations.append(f"Wide confidence interval for {metric} in {variant_name}")
        
        return recommendations
    
    async def calculate_sample_size(
        self, 
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.8,
        ratio: float = 1.0
    ) -> int:
        """Gerekli sample size hesapla"""
        
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available for sample size calculation")
            return 100  # Default fallback
        
        # Power analysis
        required_n = stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(power)
        required_n = (required_n / effect_size) ** 2
        
        # Ratio adjustment
        total_n = int(required_n * (1 + ratio))
        
        return total_n
    
    async def check_early_stopping(
        self, 
        experiment_id: str,
        threshold: float = 0.01
    ) -> Dict[str, Any]:
        """Erken durdurma kontrolü"""
        
        if experiment_id not in self.active_experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.active_experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.ACTIVE:
            return {"error": "Experiment not active"}
        
        # Varyant verilerini al
        variant_data = {}
        for variant in experiment.variants:
            if experiment_id in self.experiment_results and variant.name in self.experiment_results[experiment_id]:
                variant_data[variant.name] = self.experiment_results[experiment_id][variant.name]
        
        if len(variant_data) < 2:
            return {"should_stop": False, "reason": "Insufficient variants"}
        
        # Primary metric için test
        primary_metric = experiment.target_metrics[0] if experiment.target_metrics else "accuracy"
        
        # Her varyant için değerleri topla
        variant_values = {}
        for variant_name, data in variant_data.items():
            values = [
                event["event_data"][primary_metric] 
                for event in data 
                if primary_metric in event["event_data"]
            ]
            if values:
                variant_values[variant_name] = values
        
        if len(variant_values) < 2:
            return {"should_stop": False, "reason": "Insufficient data"}
        
        # İstatistiksel test uygula
        variant_names = list(variant_values.keys())
        control_values = variant_values[variant_names[0]]
        treatment_values = variant_values[variant_names[1]]
        
        try:
            if not SCIPY_AVAILABLE:
                return {"should_stop": False, "reason": "scipy not available for statistical tests"}
            
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            
            should_stop = p_value < threshold
            
            return {
                "should_stop": should_stop,
                "p_value": p_value,
                "threshold": threshold,
                "reason": "Early stopping threshold met" if should_stop else "Continue experiment"
            }
            
        except Exception as e:
            return {"should_stop": False, "reason": f"Statistical test failed: {str(e)}"}
    
    def check_early_stopping_values(
        self,
        control_values: List[float],
        treatment_values: List[float],
        threshold: float = 0.01
    ) -> bool:
        """Değer listeleri için erken durdurma kontrolü"""
        
        if not control_values or not treatment_values:
            return False
        
        try:
            if not SCIPY_AVAILABLE:
                return False
            
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            return p_value < threshold
        except Exception:
            return False


# Global instance
math_ab_testing = MathABTesting()
