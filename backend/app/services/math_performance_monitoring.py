import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

from app.models.math_profile import MathProfile
from app.models.question import Question

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performans metriği"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any]


@dataclass
class Alert:
    """Uyarı"""
    level: str  # info, warning, error, critical
    message: str
    timestamp: datetime
    metric_name: str
    threshold: float
    current_value: float
    metadata: Dict[str, Any]


class MathPerformanceMonitoring:
    """Matematik performans izleme servisi"""
    
    def __init__(self):
        self.config = {
            # Metrik toplama
            "metrics_window": 100,  # Son 100 metrik
            "aggregation_interval": 60,  # 60 saniye
            "retention_days": 30,  # 30 gün saklama
            
            # Zorluk uyumu metrikleri
            "difficulty_match_threshold": 0.7,
            "difficulty_match_window": 20,
            
            # Kurtarma başarısı metrikleri
            "recovery_success_threshold": 0.8,
            "recovery_window": 10,
            
            # Öğrenme hızı metrikleri
            "learning_rate_threshold": 0.02,
            "learning_rate_window": 50,
            
            # Tekrar etkinliği metrikleri
            "srs_effectiveness_threshold": 0.7,
            "srs_window": 30,
            
            # Seçim gecikmesi metrikleri
            "selection_latency_threshold": 100,  # ms
            "selection_latency_window": 50,
            
            # Uyarı eşikleri
            "warning_threshold": 0.8,
            "error_threshold": 0.6,
            "critical_threshold": 0.4,
        }
        
        # Metrik depolama
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config["metrics_window"]))
        self.alerts: List[Alert] = []
        self.session_data: Dict[str, Dict[str, Any]] = {}
        
        # Performans sayaçları
        self.counters = {
            "questions_selected": 0,
            "answers_submitted": 0,
            "recovery_attempts": 0,
            "srs_reviews": 0,
            "errors_occurred": 0,
        }
    
    async def track_question_selection(
        self, 
        user_id: str,
        question: Question,
        profile: MathProfile,
        selection_rationale: Dict[str, Any],
        latency_ms: int
    ):
        """Soru seçimini izle"""
        
        timestamp = datetime.utcnow()
        
        # Temel metrikler
        self._add_metric("selection_latency", latency_ms, {
            "user_id": user_id,
            "question_id": str(question.id),
            "difficulty": question.estimated_difficulty or question.difficulty_level,
            "mode": selection_rationale.get("mode", "unknown")
        })
        
        # Zorluk uyumu metriği
        difficulty_match = self._calculate_difficulty_match(question, profile)
        self._add_metric("difficulty_match", difficulty_match, {
            "user_id": user_id,
            "question_id": str(question.id),
            "user_skill": profile.global_skill,
            "question_difficulty": question.estimated_difficulty or question.difficulty_level
        })
        
        # Seçim modu metrikleri
        mode = selection_rationale.get("mode", "unknown")
        self._add_metric(f"selection_mode_{mode}", 1.0, {
            "user_id": user_id,
            "rationale": selection_rationale
        })
        
        # Sayaç güncelle
        self.counters["questions_selected"] += 1
        
        # Uyarı kontrolü
        await self._check_selection_alerts(user_id, latency_ms, difficulty_match)
    
    async def track_answer_submission(
        self, 
        user_id: str,
        question: Question,
        profile: MathProfile,
        is_correct: bool,
        response_time: float,
        partial_credit: Optional[float] = None
    ):
        """Cevap gönderimini izle"""
        
        timestamp = datetime.utcnow()
        
        # Doğruluk metriği
        accuracy = 1.0 if is_correct else 0.0
        self._add_metric("answer_accuracy", accuracy, {
            "user_id": user_id,
            "question_id": str(question.id),
            "difficulty": question.estimated_difficulty or question.difficulty_level,
            "response_time": response_time
        })
        
        # Yanıt süresi metriği
        self._add_metric("response_time", response_time, {
            "user_id": user_id,
            "question_id": str(question.id),
            "is_correct": is_correct
        })
        
        # Kısmi puan metriği
        if partial_credit is not None:
            self._add_metric("partial_credit", partial_credit, {
                "user_id": user_id,
                "question_id": str(question.id),
                "is_correct": is_correct
            })
        
        # Kurtarma başarısı kontrolü
        if profile.needs_recovery():
            recovery_success = 1.0 if is_correct else 0.0
            self._add_metric("recovery_success", recovery_success, {
                "user_id": user_id,
                "question_id": str(question.id)
            })
        
        # Sayaç güncelle
        self.counters["answers_submitted"] += 1
        
        # Uyarı kontrolü
        await self._check_answer_alerts(user_id, is_correct, response_time)
    
    async def track_recovery_attempt(
        self, 
        user_id: str,
        profile: MathProfile,
        success: bool
    ):
        """Kurtarma girişimini izle"""
        
        timestamp = datetime.utcnow()
        
        # Kurtarma başarısı metriği
        recovery_success = 1.0 if success else 0.0
        self._add_metric("recovery_attempt_success", recovery_success, {
            "user_id": user_id,
            "profile_state": {
                "global_skill": profile.global_skill,
                "ema_accuracy": profile.ema_accuracy,
                "streak_wrong": profile.streak_wrong
            }
        })
        
        # Sayaç güncelle
        self.counters["recovery_attempts"] += 1
        
        # Uyarı kontrolü
        await self._check_recovery_alerts(user_id, success)
    
    async def track_srs_review(
        self, 
        user_id: str,
        question_id: str,
        success: bool,
        ease_factor_change: float
    ):
        """SRS tekrarını izle"""
        
        timestamp = datetime.utcnow()
        
        # SRS etkinliği metriği
        srs_success = 1.0 if success else 0.0
        self._add_metric("srs_review_success", srs_success, {
            "user_id": user_id,
            "question_id": question_id,
            "ease_factor_change": ease_factor_change
        })
        
        # Ease factor değişimi metriği
        self._add_metric("ease_factor_change", ease_factor_change, {
            "user_id": user_id,
            "question_id": question_id,
            "success": success
        })
        
        # Sayaç güncelle
        self.counters["srs_reviews"] += 1
        
        # Uyarı kontrolü
        await self._check_srs_alerts(user_id, success, ease_factor_change)
    
    async def track_error(
        self, 
        error_type: str,
        error_message: str,
        context: Dict[str, Any]
    ):
        """Hata izleme"""
        
        timestamp = datetime.utcnow()
        
        # Hata metriği
        self._add_metric("error_rate", 1.0, {
            "error_type": error_type,
            "error_message": error_message,
            "context": context
        })
        
        # Sayaç güncelle
        self.counters["errors_occurred"] += 1
        
        # Uyarı oluştur
        alert = Alert(
            level="error",
            message=f"Error occurred: {error_type} - {error_message}",
            timestamp=timestamp,
            metric_name="error_rate",
            threshold=0.0,
            current_value=1.0,
            metadata={"error_type": error_type, "context": context}
        )
        self.alerts.append(alert)
    
    def get_performance_metrics(
        self, 
        metric_name: Optional[str] = None,
        user_id: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Performans metriklerini al"""
        
        if metric_name:
            metrics = self._filter_metrics(metric_name, user_id, time_window)
            return self._calculate_metric_statistics(metric_name, metrics)
        else:
            # Tüm metrikler
            all_metrics = {}
            for metric_name in self.metrics.keys():
                metrics = self._filter_metrics(metric_name, user_id, time_window)
                all_metrics[metric_name] = self._calculate_metric_statistics(metric_name, metrics)
            
            return all_metrics
    
    def get_alerts(
        self, 
        level: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """Uyarıları al"""
        
        filtered_alerts = self.alerts
        
        if level:
            filtered_alerts = [alert for alert in filtered_alerts if alert.level == level]
        
        if time_window:
            cutoff_time = datetime.utcnow() - time_window
            filtered_alerts = [alert for alert in filtered_alerts if alert.timestamp >= cutoff_time]
        
        return [asdict(alert) for alert in filtered_alerts]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Sistem sağlığını kontrol et"""
        
        # Metrik sağlığı
        metric_health = {}
        for metric_name, metrics in self.metrics.items():
            if metrics:
                recent_metrics = list(metrics)[-10:]  # Son 10 metrik
                avg_value = statistics.mean(m.value for m in recent_metrics)
                
                # Sağlık skoru hesapla
                if metric_name == "selection_latency":
                    health_score = 1.0 if avg_value < self.config["selection_latency_threshold"] else 0.5
                elif metric_name == "difficulty_match":
                    health_score = 1.0 if avg_value > self.config["difficulty_match_threshold"] else 0.5
                elif metric_name == "answer_accuracy":
                    health_score = avg_value
                else:
                    health_score = min(1.0, avg_value)
                
                metric_health[metric_name] = {
                    "health_score": health_score,
                    "average_value": avg_value,
                    "metric_count": len(metrics)
                }
        
        # Genel sağlık skoru
        overall_health = statistics.mean(
            health["health_score"] for health in metric_health.values()
        ) if metric_health else 1.0
        
        # Sayaç durumu
        counter_status = {
            name: count for name, count in self.counters.items()
        }
        
        return {
            "overall_health": overall_health,
            "metric_health": metric_health,
            "counter_status": counter_status,
            "active_alerts": len([a for a in self.alerts if a.level in ["error", "critical"]]),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _add_metric(self, name: str, value: float, metadata: Dict[str, Any]):
        """Metrik ekle"""
        
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            value=value,
            metadata=metadata
        )
        
        self.metrics[name].append(metric)
    
    def _filter_metrics(
        self, 
        metric_name: str, 
        user_id: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> List[PerformanceMetric]:
        """Metrikleri filtrele"""
        
        if metric_name not in self.metrics:
            return []
        
        metrics = list(self.metrics[metric_name])
        
        # Kullanıcı filtresi
        if user_id:
            metrics = [m for m in metrics if m.metadata.get("user_id") == user_id]
        
        # Zaman filtresi
        if time_window:
            cutoff_time = datetime.utcnow() - time_window
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return metrics
    
    def _calculate_metric_statistics(self, metric_name: str, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Metrik istatistiklerini hesapla"""
        
        if not metrics:
            return {
                "count": 0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "trend": "stable"
            }
        
        values = [m.value for m in metrics]
        
        # Temel istatistikler
        stats = {
            "count": len(values),
            "average": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0
        }
        
        # Trend hesaplama
        if len(values) >= 5:
            recent_avg = statistics.mean(values[-5:])
            older_avg = statistics.mean(values[:-5])
            
            if recent_avg > older_avg * 1.1:
                trend = "improving"
            elif recent_avg < older_avg * 0.9:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        stats["trend"] = trend
        
        return stats
    
    def _calculate_difficulty_match(self, question: Question, profile: MathProfile) -> float:
        """Zorluk uyumu hesapla"""
        
        question_difficulty = question.estimated_difficulty or question.difficulty_level
        user_skill = profile.global_skill
        
        # Hedef zorluk aralığı
        target_low, target_high = profile.get_target_difficulty_range()
        
        if target_low <= question_difficulty <= target_high:
            return 1.0
        else:
            # Mesafeye göre skor
            distance = min(
                abs(question_difficulty - target_low),
                abs(question_difficulty - target_high)
            )
            return max(0.0, 1.0 - (distance / 2.0))
    
    async def _check_selection_alerts(self, user_id: str, latency_ms: int, difficulty_match: float):
        """Seçim uyarılarını kontrol et"""
        
        # Gecikme uyarısı
        if latency_ms > self.config["selection_latency_threshold"]:
            alert = Alert(
                level="warning",
                message=f"Question selection latency ({latency_ms}ms) exceeds threshold",
                timestamp=datetime.utcnow(),
                metric_name="selection_latency",
                threshold=self.config["selection_latency_threshold"],
                current_value=latency_ms,
                metadata={"user_id": user_id}
            )
            self.alerts.append(alert)
        
        # Zorluk uyumu uyarısı
        if difficulty_match < self.config["difficulty_match_threshold"]:
            alert = Alert(
                level="warning",
                message=f"Difficulty match ({difficulty_match:.2f}) is below threshold",
                timestamp=datetime.utcnow(),
                metric_name="difficulty_match",
                threshold=self.config["difficulty_match_threshold"],
                current_value=difficulty_match,
                metadata={"user_id": user_id}
            )
            self.alerts.append(alert)
    
    async def _check_answer_alerts(self, user_id: str, is_correct: bool, response_time: float):
        """Cevap uyarılarını kontrol et"""
        
        # Yanıt süresi uyarısı
        if response_time < 5:  # Çok hızlı
            alert = Alert(
                level="warning",
                message=f"Response time ({response_time}s) is suspiciously fast",
                timestamp=datetime.utcnow(),
                metric_name="response_time",
                threshold=5.0,
                current_value=response_time,
                metadata={"user_id": user_id, "is_correct": is_correct}
            )
            self.alerts.append(alert)
    
    async def _check_recovery_alerts(self, user_id: str, success: bool):
        """Kurtarma uyarılarını kontrol et"""
        
        # Kurtarma başarısı metriklerini kontrol et
        recovery_metrics = self._filter_metrics("recovery_attempt_success", user_id)
        
        if len(recovery_metrics) >= self.config["recovery_window"]:
            recent_success_rate = statistics.mean(
                m.value for m in recovery_metrics[-self.config["recovery_window"]:]
            )
            
            if recent_success_rate < self.config["recovery_success_threshold"]:
                alert = Alert(
                    level="warning",
                    message=f"Recovery success rate ({recent_success_rate:.2f}) is below threshold",
                    timestamp=datetime.utcnow(),
                    metric_name="recovery_success_rate",
                    threshold=self.config["recovery_success_threshold"],
                    current_value=recent_success_rate,
                    metadata={"user_id": user_id}
                )
                self.alerts.append(alert)
    
    async def _check_srs_alerts(self, user_id: str, success: bool, ease_factor_change: float):
        """SRS uyarılarını kontrol et"""
        
        # SRS etkinliği metriklerini kontrol et
        srs_metrics = self._filter_metrics("srs_review_success", user_id)
        
        if len(srs_metrics) >= self.config["srs_window"]:
            recent_success_rate = statistics.mean(
                m.value for m in srs_metrics[-self.config["srs_window"]:]
            )
            
            if recent_success_rate < self.config["srs_effectiveness_threshold"]:
                alert = Alert(
                    level="warning",
                    message=f"SRS effectiveness ({recent_success_rate:.2f}) is below threshold",
                    timestamp=datetime.utcnow(),
                    metric_name="srs_effectiveness",
                    threshold=self.config["srs_effectiveness_threshold"],
                    current_value=recent_success_rate,
                    metadata={"user_id": user_id}
                )
                self.alerts.append(alert)


# Global instance
math_performance_monitoring = MathPerformanceMonitoring()
