import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from app.models.math_profile import MathProfile
from app.models.question import Question

logger = logging.getLogger(__name__)


class MathAnalyticsService:
    """Matematik analitik servisi - matematik.md'deki analytics features"""
    
    def __init__(self):
        self.config = {
            "learning_curve_window": 30,  # days
            "performance_threshold": 0.7,
            "improvement_threshold": 0.1,
            "stagnation_threshold": 0.05,
            "mastery_threshold": 0.9,
        }
    
    def analyze_learning_progress(
        self, 
        profile: MathProfile,
        performance_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Öğrenme ilerlemesini analiz et"""
        
        if not performance_history:
            return self._get_default_learning_analysis()
        
        # Zaman serisi analizi
        time_series = self._create_time_series(performance_history)
        
        # Trend analizi
        trend_analysis = self._analyze_trends(time_series)
        
        # Öğrenme hızı analizi
        learning_rate = self._calculate_learning_rate(time_series)
        
        # Zorluk adaptasyonu analizi
        difficulty_adaptation = self._analyze_difficulty_adaptation(performance_history)
        
        # Konu bazlı performans
        topic_performance = self._analyze_topic_performance(performance_history)
        
        # Öğrenme aşaması belirleme
        learning_stage = self._determine_learning_stage(profile, trend_analysis)
        
        return {
            "learning_stage": learning_stage,
            "trend_analysis": trend_analysis,
            "learning_rate": learning_rate,
            "difficulty_adaptation": difficulty_adaptation,
            "topic_performance": topic_performance,
            "recommendations": self._generate_learning_recommendations(
                learning_stage, trend_analysis, topic_performance
            ),
            "progress_metrics": {
                "overall_progress": self._calculate_overall_progress(profile),
                "consistency_score": self._calculate_consistency_score(performance_history),
                "engagement_level": self._calculate_engagement_level(performance_history),
            }
        }
    
    def analyze_algorithm_performance(
        self, 
        profile: MathProfile,
        selection_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Algoritma performansını analiz et"""
        
        if not selection_history:
            return self._get_default_algorithm_analysis()
        
        # Seçim modu analizi
        mode_analysis = self._analyze_selection_modes(selection_history)
        
        # Bandit performansı
        bandit_performance = self._analyze_bandit_performance(profile, selection_history)
        
        # SRS etkinliği
        srs_effectiveness = self._analyze_srs_effectiveness(profile, selection_history)
        
        # Zorluk uyumu
        difficulty_match = self._analyze_difficulty_matching(selection_history)
        
        # Algoritma optimizasyon önerileri
        optimization_suggestions = self._generate_optimization_suggestions(
            mode_analysis, bandit_performance, srs_effectiveness
        )
        
        return {
            "mode_analysis": mode_analysis,
            "bandit_performance": bandit_performance,
            "srs_effectiveness": srs_effectiveness,
            "difficulty_match": difficulty_match,
            "optimization_suggestions": optimization_suggestions,
            "algorithm_metrics": {
                "selection_accuracy": self._calculate_selection_accuracy(selection_history),
                "adaptation_speed": self._calculate_adaptation_speed(selection_history),
                "exploration_efficiency": self._calculate_exploration_efficiency(selection_history),
            }
        }
    
    def predict_performance_trajectory(
        self, 
        profile: MathProfile,
        performance_history: List[Dict[str, Any]],
        prediction_horizon: int = 30
    ) -> Dict[str, Any]:
        """Performans trajectory'sini tahmin et"""
        
        if len(performance_history) < 10:
            return self._get_default_prediction()
        
        # Zaman serisi hazırla
        time_series = self._create_time_series(performance_history)
        
        # Trend projeksiyonu
        trend_projection = self._project_trend(time_series, prediction_horizon)
        
        # Güven aralıkları
        confidence_intervals = self._calculate_prediction_intervals(trend_projection)
        
        # Milestone tahminleri
        milestone_predictions = self._predict_milestones(profile, trend_projection)
        
        # Risk analizi
        risk_analysis = self._analyze_learning_risks(profile, trend_projection)
        
        return {
            "prediction_horizon": prediction_horizon,
            "trend_projection": trend_projection,
            "confidence_intervals": confidence_intervals,
            "milestone_predictions": milestone_predictions,
            "risk_analysis": risk_analysis,
            "prediction_metrics": {
                "prediction_confidence": self._calculate_prediction_confidence(trend_projection),
                "model_accuracy": self._estimate_model_accuracy(performance_history),
            }
        }
    
    def generate_adaptive_recommendations(
        self, 
        profile: MathProfile,
        performance_history: List[Dict[str, Any]],
        current_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adaptif öneriler oluştur"""
        
        # Performans analizi
        performance_analysis = self.analyze_learning_progress(profile, performance_history)
        
        # Algoritma performansı
        algorithm_analysis = self.analyze_algorithm_performance(profile, [])  # Placeholder
        
        # Context-aware öneriler
        context_recommendations = self._generate_context_recommendations(
            profile, current_context, performance_analysis
        )
        
        # Öğrenme stratejisi önerileri
        strategy_recommendations = self._generate_strategy_recommendations(
            profile, performance_analysis
        )
        
        # Motivasyon önerileri
        motivation_recommendations = self._generate_motivation_recommendations(
            profile, performance_analysis
        )
        
        return {
            "context_recommendations": context_recommendations,
            "strategy_recommendations": strategy_recommendations,
            "motivation_recommendations": motivation_recommendations,
            "priority_actions": self._prioritize_recommendations(
                context_recommendations, strategy_recommendations, motivation_recommendations
            ),
            "recommendation_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "confidence_level": self._calculate_recommendation_confidence(performance_analysis),
                "validity_period": "24h",
            }
        }
    
    def _create_time_series(self, performance_history: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Zaman serisi oluştur"""
        
        # Tarihe göre sırala
        sorted_history = sorted(performance_history, key=lambda x: x.get("timestamp", datetime.utcnow()))
        
        # Metrikleri ayır
        accuracies = [p.get("accuracy", 0.5) for p in sorted_history]
        speeds = [p.get("speed", 0.5) for p in sorted_history]
        difficulties = [p.get("difficulty", 2.5) for p in sorted_history]
        
        return {
            "accuracies": accuracies,
            "speeds": speeds,
            "difficulties": difficulties,
            "timestamps": [p.get("timestamp", datetime.utcnow()) for p in sorted_history]
        }
    
    def _analyze_trends(self, time_series: Dict[str, List[float]]) -> Dict[str, Any]:
        """Trend analizi"""
        
        accuracies = time_series["accuracies"]
        speeds = time_series["speeds"]
        
        if len(accuracies) < 3:
            return {"trend": "insufficient_data", "slope": 0, "r_squared": 0}
        
        # Linear regression
        x = np.arange(len(accuracies))
        accuracy_slope = np.polyfit(x, accuracies, 1)[0]
        speed_slope = np.polyfit(x, speeds, 1)[0]
        
        # R-squared hesapla
        accuracy_r_squared = self._calculate_r_squared(x, accuracies)
        speed_r_squared = self._calculate_r_squared(x, speeds)
        
        # Trend belirleme
        overall_trend = "stable"
        if accuracy_slope > self.config["improvement_threshold"]:
            overall_trend = "improving"
        elif accuracy_slope < -self.config["improvement_threshold"]:
            overall_trend = "declining"
        
        return {
            "trend": overall_trend,
            "accuracy_trend": {
                "slope": accuracy_slope,
                "r_squared": accuracy_r_squared,
                "direction": "improving" if accuracy_slope > 0 else "declining"
            },
            "speed_trend": {
                "slope": speed_slope,
                "r_squared": speed_r_squared,
                "direction": "improving" if speed_slope > 0 else "declining"
            }
        }
    
    def _calculate_learning_rate(self, time_series: Dict[str, List[float]]) -> Dict[str, Any]:
        """Öğrenme hızını hesapla"""
        
        accuracies = time_series["accuracies"]
        
        if len(accuracies) < 5:
            return {"rate": 0, "phase": "initial", "efficiency": 0.5}
        
        # Öğrenme hızı (accuracy artışı / zaman)
        total_improvement = accuracies[-1] - accuracies[0]
        time_span = len(accuracies)
        learning_rate = total_improvement / time_span if time_span > 0 else 0
        
        # Öğrenme fazı belirleme
        if learning_rate > 0.05:
            phase = "rapid_learning"
        elif learning_rate > 0.02:
            phase = "steady_learning"
        elif learning_rate > 0:
            phase = "slow_learning"
        else:
            phase = "plateau"
        
        # Verimlilik hesapla
        efficiency = min(1.0, learning_rate * 20)  # Normalize to 0-1
        
        return {
            "rate": learning_rate,
            "phase": phase,
            "efficiency": efficiency,
            "total_improvement": total_improvement
        }
    
    def _analyze_difficulty_adaptation(self, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Zorluk adaptasyonunu analiz et"""
        
        if len(performance_history) < 5:
            return {"adaptation_score": 0.5, "optimal_range": (0.6, 0.8)}
        
        # Zorluk ve performans ilişkisi
        difficulties = [p.get("difficulty", 2.5) for p in performance_history]
        accuracies = [p.get("accuracy", 0.5) for p in performance_history]
        
        # Optimal zorluk aralığı (accuracy 0.6-0.8 arası)
        optimal_performances = [i for i, acc in enumerate(accuracies) if 0.6 <= acc <= 0.8]
        
        if optimal_performances:
            optimal_difficulties = [difficulties[i] for i in optimal_performances]
            optimal_range = (min(optimal_difficulties), max(optimal_difficulties))
        else:
            optimal_range = (2.0, 3.0)  # Default range
        
        # Adaptasyon skoru
        adaptation_score = len(optimal_performances) / len(performance_history)
        
        return {
            "adaptation_score": adaptation_score,
            "optimal_range": optimal_range,
            "current_difficulty": difficulties[-1] if difficulties else 2.5,
            "difficulty_variance": np.var(difficulties) if len(difficulties) > 1 else 0
        }
    
    def _analyze_topic_performance(self, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Konu bazlı performans analizi"""
        
        topic_stats = defaultdict(lambda: {"count": 0, "correct": 0, "total_time": 0})
        
        for performance in performance_history:
            topic = performance.get("topic", "unknown")
            is_correct = performance.get("is_correct", False)
            response_time = performance.get("response_time", 60)
            
            topic_stats[topic]["count"] += 1
            if is_correct:
                topic_stats[topic]["correct"] += 1
            topic_stats[topic]["total_time"] += response_time
        
        # Konu bazlı metrikler
        topic_performance = {}
        for topic, stats in topic_stats.items():
            accuracy = stats["correct"] / stats["count"] if stats["count"] > 0 else 0
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 60
            
            topic_performance[topic] = {
                "accuracy": accuracy,
                "average_time": avg_time,
                "question_count": stats["count"],
                "strength_level": self._categorize_strength(accuracy)
            }
        
        return topic_performance
    
    def _determine_learning_stage(self, profile: MathProfile, trend_analysis: Dict[str, Any]) -> str:
        """Öğrenme aşamasını belirle"""
        
        accuracy = profile.ema_accuracy
        trend = trend_analysis.get("trend", "stable")
        
        if accuracy >= self.config["mastery_threshold"]:
            return "mastery"
        elif accuracy >= self.config["performance_threshold"] and trend == "improving":
            return "proficient"
        elif accuracy >= 0.5 and trend in ["improving", "stable"]:
            return "developing"
        elif accuracy < 0.5 or trend == "declining":
            return "struggling"
        else:
            return "beginning"
    
    def _generate_learning_recommendations(
        self, 
        learning_stage: str, 
        trend_analysis: Dict[str, Any],
        topic_performance: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Öğrenme önerileri oluştur"""
        
        recommendations = []
        
        # Aşama bazlı öneriler
        if learning_stage == "struggling":
            recommendations.append({
                "type": "foundation",
                "priority": "high",
                "message": "Temel konulara odaklanın ve daha kolay sorularla pratik yapın",
                "action": "focus_on_basics"
            })
        elif learning_stage == "beginning":
            recommendations.append({
                "type": "practice",
                "priority": "medium",
                "message": "Düzenli pratik yaparak temel becerilerinizi geliştirin",
                "action": "regular_practice"
            })
        elif learning_stage == "developing":
            recommendations.append({
                "type": "challenge",
                "priority": "medium",
                "message": "Zorluk seviyesini artırarak becerilerinizi test edin",
                "action": "increase_challenge"
            })
        elif learning_stage == "proficient":
            recommendations.append({
                "type": "mastery",
                "priority": "low",
                "message": "İleri seviye konulara geçerek uzmanlaşın",
                "action": "advanced_topics"
            })
        
        # Trend bazlı öneriler
        if trend_analysis.get("trend") == "declining":
            recommendations.append({
                "type": "recovery",
                "priority": "high",
                "message": "Performansınız düşüyor, kolay sorularla güveninizi geri kazanın",
                "action": "confidence_building"
            })
        
        # Konu bazlı öneriler
        weak_topics = [topic for topic, perf in topic_performance.items() 
                      if perf.get("strength_level") == "weak"]
        if weak_topics:
            recommendations.append({
                "type": "topic_focus",
                "priority": "medium",
                "message": f"Zayıf konularınıza odaklanın: {', '.join(weak_topics[:3])}",
                "action": "focus_weak_topics"
            })
        
        return recommendations
    
    def _get_default_learning_analysis(self) -> Dict[str, Any]:
        """Varsayılan öğrenme analizi"""
        return {
            "learning_stage": "beginning",
            "trend_analysis": {"trend": "insufficient_data"},
            "learning_rate": {"rate": 0, "phase": "initial"},
            "difficulty_adaptation": {"adaptation_score": 0.5},
            "topic_performance": {},
            "recommendations": [],
            "progress_metrics": {"overall_progress": 0, "consistency_score": 0.5, "engagement_level": 0.5}
        }
    
    def _get_default_algorithm_analysis(self) -> Dict[str, Any]:
        """Varsayılan algoritma analizi"""
        return {
            "mode_analysis": {"modes": {}},
            "bandit_performance": {"exploration_rate": 0.1, "exploitation_efficiency": 0.5},
            "srs_effectiveness": {"retention_rate": 0.5, "spacing_efficiency": 0.5},
            "difficulty_match": {"match_rate": 0.5},
            "optimization_suggestions": [],
            "algorithm_metrics": {"selection_accuracy": 0.5, "adaptation_speed": 0.5, "exploration_efficiency": 0.5}
        }
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """Varsayılan tahmin"""
        return {
            "prediction_horizon": 30,
            "trend_projection": {"projected_accuracy": 0.5},
            "confidence_intervals": {"lower": 0.3, "upper": 0.7},
            "milestone_predictions": [],
            "risk_analysis": {"risk_level": "low"},
            "prediction_metrics": {"prediction_confidence": 0.5, "model_accuracy": 0.5}
        }
    
    def _calculate_r_squared(self, x: np.ndarray, y: List[float]) -> float:
        """R-squared hesapla"""
        try:
            correlation_matrix = np.corrcoef(x, y)
            correlation = correlation_matrix[0, 1]
            return correlation ** 2
        except:
            return 0.0
    
    def _categorize_strength(self, accuracy: float) -> str:
        """Güç seviyesini kategorize et"""
        if accuracy >= 0.8:
            return "strong"
        elif accuracy >= 0.6:
            return "moderate"
        else:
            return "weak"
    
    def _calculate_overall_progress(self, profile: MathProfile) -> float:
        """Genel ilerlemeyi hesapla"""
        return min(1.0, profile.global_skill / 5.0)
    
    def _calculate_consistency_score(self, performance_history: List[Dict[str, Any]]) -> float:
        """Tutarlılık skorunu hesapla"""
        if len(performance_history) < 2:
            return 0.5
        
        outcomes = [p.get("is_correct", False) for p in performance_history]
        changes = sum(1 for i in range(1, len(outcomes)) if outcomes[i] != outcomes[i-1])
        return 1 - (changes / (len(outcomes) - 1))
    
    def _calculate_engagement_level(self, performance_history: List[Dict[str, Any]]) -> float:
        """Katılım seviyesini hesapla"""
        if not performance_history:
            return 0.5
        
        # Son 7 gün içindeki aktivite
        recent_activity = [p for p in performance_history 
                          if p.get("timestamp", datetime.utcnow()) > datetime.utcnow() - timedelta(days=7)]
        
        return min(1.0, len(recent_activity) / 10)  # Normalize to 0-1


# Global instance
math_analytics_service = MathAnalyticsService()
