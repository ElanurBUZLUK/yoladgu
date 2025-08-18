import random
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from app.models.math_profile import MathProfile
from app.models.question import Question

logger = logging.getLogger(__name__)


class AdvancedMathAlgorithms:
    """Gelişmiş matematik seçim algoritmaları - matematik.md'deki advanced features"""
    
    def __init__(self):
        self.config = {
            # Adaptive difficulty parameters
            "learning_rate": 0.1,
            "momentum": 0.9,
            "adaptive_window": 10,
            "confidence_threshold": 0.8,
            
            # Thompson sampling parameters
            "exploration_rate": 0.1,
            "decay_factor": 0.95,
            "min_alpha_beta": 1,
            "max_alpha_beta": 100,
            
            # SRS parameters
            "srs_intervals": [0, 1, 3, 7, 16, 35, 70],  # Extended intervals
            "srs_ease_factor": 2.5,
            "srs_min_ease": 1.3,
            "srs_max_ease": 2.5,
            
            # Scoring parameters
            "freshness_weight": 0.2,
            "difficulty_weight": 0.4,
            "quality_weight": 0.2,
            "diversity_weight": 0.2,
            
            # Performance tracking
            "performance_window": 20,
            "trend_threshold": 0.05,
        }
    
    def enhanced_adaptive_difficulty(
        self, 
        profile: MathProfile, 
        recent_performance: List[Dict[str, Any]]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Gelişmiş adaptif zorluk hesaplama"""
        
        # Temel zorluk hesaplama
        base_difficulty = profile.global_skill * profile.difficulty_factor
        
        # Performans analizi
        performance_metrics = self._analyze_recent_performance(recent_performance)
        
        # Trend analizi
        trend = self._calculate_performance_trend(recent_performance)
        
        # Güven aralığı hesaplama
        confidence_interval = self._calculate_confidence_interval(performance_metrics)
        
        # Dinamik zorluk ayarlama
        if trend == "improving" and performance_metrics["accuracy"] > self.config["confidence_threshold"]:
            # Hızlı ilerleme - zorluğu artır
            difficulty_adjustment = 0.5 + (performance_metrics["speed"] * 0.3)
            target_difficulty = base_difficulty + difficulty_adjustment
        elif trend == "declining" or performance_metrics["accuracy"] < 0.4:
            # Gerileme - zorluğu azalt
            difficulty_adjustment = -0.3 - (1 - performance_metrics["accuracy"]) * 0.5
            target_difficulty = base_difficulty + difficulty_adjustment
        else:
            # Stabil - küçük ayarlamalar
            difficulty_adjustment = (performance_metrics["accuracy"] - 0.7) * 0.2
            target_difficulty = base_difficulty + difficulty_adjustment
        
        # Sınırları uygula
        target_difficulty = max(0.0, min(5.0, target_difficulty))
        
        # Zorluk aralığı hesapla
        range_width = confidence_interval["width"] * 0.5
        target_low = max(0.0, target_difficulty - range_width)
        target_high = min(5.0, target_difficulty + range_width)
        
        rationale = {
            "base_difficulty": base_difficulty,
            "performance_metrics": performance_metrics,
            "trend": trend,
            "confidence_interval": confidence_interval,
            "difficulty_adjustment": difficulty_adjustment,
            "target_difficulty": target_difficulty,
            "range_width": range_width,
        }
        
        return target_low, target_high, rationale
    
    def advanced_thompson_sampling(
        self, 
        profile: MathProfile, 
        context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Gelişmiş Thompson Sampling"""
        
        # Context-aware bandit arms
        context_arms = self._get_context_aware_arms(context)
        
        # Decay factor uygula (eski verilerin etkisini azalt)
        decayed_arms = self._apply_decay_to_arms(profile.bandit_arms)
        
        # Exploration vs exploitation balance
        if random.random() < self.config["exploration_rate"]:
            # Exploration: rastgele arm seç
            selected_delta = random.choice(context_arms)
            rationale = {"mode": "exploration", "arms": context_arms}
        else:
            # Exploitation: Thompson sampling
            best_delta = 0.0
            best_sample = -1
            samples = {}
            
            for delta in context_arms:
                delta_key = str(delta)
                if delta_key in decayed_arms:
                    alpha, beta = decayed_arms[delta_key]
                    # Minimum/maximum sınırları uygula
                    alpha = max(self.config["min_alpha_beta"], min(self.config["max_alpha_beta"], alpha))
                    beta = max(self.config["min_alpha_beta"], min(self.config["max_alpha_beta"], beta))
                    
                    sample = random.betavariate(alpha, beta)
                    samples[delta] = sample
                    
                    if sample > best_sample:
                        best_sample = sample
                        best_delta = delta
            
            selected_delta = best_delta
            rationale = {
                "mode": "exploitation", 
                "arms": context_arms,
                "samples": samples,
                "best_sample": best_sample,
                "decay_applied": True
            }
        
        return selected_delta, rationale
    
    def enhanced_srs_algorithm(
        self, 
        profile: MathProfile, 
        question_id: str,
        is_correct: bool,
        response_time: float,
        difficulty: float
    ) -> Dict[str, Any]:
        """Gelişmiş SRS algoritması (SM-2+ benzeri)"""
        
        # SRS item'ı bul veya oluştur
        srs_item = self._find_or_create_srs_item(profile, question_id)
        
        # Ease factor hesaplama
        ease_factor = srs_item.get("ease_factor", self.config["srs_ease_factor"])
        
        # Response quality hesaplama (0-5 scale)
        response_quality = self._calculate_response_quality(
            is_correct, response_time, difficulty, profile
        )
        
        # Ease factor güncelleme
        new_ease_factor = ease_factor + (0.1 - (5 - response_quality) * (0.08 + (5 - response_quality) * 0.02))
        new_ease_factor = max(self.config["srs_min_ease"], min(self.config["srs_max_ease"], new_ease_factor))
        
        # Interval hesaplama
        if response_quality >= 3:  # Başarılı
            if srs_item["level"] == 0:
                interval = 1
            elif srs_item["level"] == 1:
                interval = 6
            else:
                interval = int(srs_item["interval"] * new_ease_factor)
            
            srs_item["level"] = min(6, srs_item["level"] + 1)
        else:  # Başarısız
            srs_item["level"] = max(0, srs_item["level"] - 1)
            interval = self.config["srs_intervals"][srs_item["level"]]
        
        # Yeni vade tarihi
        due_date = datetime.utcnow() + timedelta(days=interval)
        
        # SRS item'ı güncelle
        srs_item.update({
            "ease_factor": new_ease_factor,
            "interval": interval,
            "due_date": due_date.isoformat(),
            "last_review": datetime.utcnow().isoformat(),
            "response_quality": response_quality,
            "review_count": srs_item.get("review_count", 0) + 1,
        })
        
        return {
            "srs_item": srs_item,
            "response_quality": response_quality,
            "ease_factor_change": new_ease_factor - ease_factor,
            "interval_days": interval,
            "next_review": due_date.isoformat(),
        }
    
    def advanced_scoring_algorithm(
        self, 
        candidates: List[Question], 
        target_difficulty: float,
        profile: MathProfile,
        context: Dict[str, Any]
    ) -> List[Tuple[Question, float, Dict[str, Any]]]:
        """Gelişmiş soru skorlama algoritması"""
        
        scored_candidates = []
        
        for question in candidates:
            # Temel mesafe skoru
            difficulty = question.estimated_difficulty or question.difficulty_level
            distance_score = abs(difficulty - target_difficulty)
            
            # Tazelik skoru
            freshness_score = self._calculate_freshness_score(question, profile)
            
            # Kalite skoru
            quality_score = self._calculate_quality_score(question)
            
            # Çeşitlilik skoru
            diversity_score = self._calculate_diversity_score(question, profile, context)
            
            # Ağırlıklı toplam skor
            total_score = (
                self.config["difficulty_weight"] * (1 - distance_score / 5.0) +
                self.config["freshness_weight"] * freshness_score +
                self.config["quality_weight"] * quality_score +
                self.config["diversity_weight"] * diversity_score
            )
            
            # Rastgelelik ekle
            random_factor = random.random() * 0.1
            final_score = total_score + random_factor
            
            rationale = {
                "distance_score": distance_score,
                "freshness_score": freshness_score,
                "quality_score": quality_score,
                "diversity_score": diversity_score,
                "total_score": total_score,
                "random_factor": random_factor,
                "final_score": final_score,
            }
            
            scored_candidates.append((question, final_score, rationale))
        
        # Skora göre sırala (yüksek skor = daha iyi)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates
    
    def _analyze_recent_performance(self, recent_performance: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Son performansı analiz et"""
        
        if not recent_performance:
            return {
                "accuracy": 0.5,
                "speed": 0.5,
                "consistency": 0.5,
                "trend": "stable"
            }
        
        # Doğruluk hesapla
        correct_count = sum(1 for p in recent_performance if p.get("is_correct", False))
        accuracy = correct_count / len(recent_performance)
        
        # Hız hesapla
        response_times = [p.get("response_time", 60) for p in recent_performance]
        avg_response_time = sum(response_times) / len(response_times)
        speed = max(0, 1 - (avg_response_time - 30) / 120)  # Normalize to 0-1
        
        # Tutarlılık hesapla
        outcomes = [p.get("is_correct", False) for p in recent_performance]
        consistency = 1 - (sum(abs(outcomes[i] - outcomes[i-1]) for i in range(1, len(outcomes))) / (len(outcomes) - 1)) if len(outcomes) > 1 else 1
        
        return {
            "accuracy": accuracy,
            "speed": speed,
            "consistency": consistency,
            "sample_size": len(recent_performance)
        }
    
    def _calculate_performance_trend(self, recent_performance: List[Dict[str, Any]]) -> str:
        """Performans trendini hesapla"""
        
        if len(recent_performance) < 5:
            return "stable"
        
        # Son 5 vs önceki 5 karşılaştır
        recent_5 = recent_performance[-5:]
        previous_5 = recent_performance[-10:-5] if len(recent_performance) >= 10 else recent_performance[:-5]
        
        recent_accuracy = sum(1 for p in recent_5 if p.get("is_correct", False)) / len(recent_5)
        previous_accuracy = sum(1 for p in previous_5 if p.get("is_correct", False)) / len(previous_5)
        
        if recent_accuracy > previous_accuracy + self.config["trend_threshold"]:
            return "improving"
        elif recent_accuracy < previous_accuracy - self.config["trend_threshold"]:
            return "declining"
        else:
            return "stable"
    
    def _calculate_confidence_interval(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Güven aralığı hesapla"""
        
        # Basit güven aralığı (Wilson score interval benzeri)
        accuracy = performance_metrics["accuracy"]
        sample_size = performance_metrics["sample_size"]
        
        if sample_size == 0:
            return {"width": 1.0, "confidence": 0.5}
        
        # Wilson score interval approximation
        z = 1.96  # 95% confidence
        denominator = 1 + z**2 / sample_size
        centre_adjusted_probability = (accuracy + z * z / (2 * sample_size)) / denominator
        adjusted_standard_error = z * math.sqrt((accuracy * (1 - accuracy) + z * z / (4 * sample_size)) / sample_size) / denominator
        
        lower_bound = centre_adjusted_probability - adjusted_standard_error
        upper_bound = centre_adjusted_probability + adjusted_standard_error
        
        return {
            "width": upper_bound - lower_bound,
            "confidence": 0.95,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
    
    def _get_context_aware_arms(self, context: Dict[str, Any]) -> List[float]:
        """Context-aware bandit arms"""
        
        # Temel arms
        base_arms = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        # Context'e göre arms'ları filtrele/ayarla
        if context.get("needs_recovery", False):
            # Kurtarma modu - daha küçük arms
            return [-0.5, -0.25, 0.0, 0.25]
        elif context.get("high_performance", False):
            # Yüksek performans - daha büyük arms
            return [-0.5, 0.0, 0.5, 1.0, 1.5]
        else:
            return base_arms
    
    def _apply_decay_to_arms(self, bandit_arms: Dict[str, List[int]]) -> Dict[str, List[float]]:
        """Bandit arms'lara decay uygula"""
        
        decayed_arms = {}
        for delta_key, (alpha, beta) in bandit_arms.items():
            # Decay factor uygula
            decayed_alpha = alpha * self.config["decay_factor"]
            decayed_beta = beta * self.config["decay_factor"]
            
            # Minimum değerleri koru
            decayed_alpha = max(self.config["min_alpha_beta"], decayed_alpha)
            decayed_beta = max(self.config["min_alpha_beta"], decayed_beta)
            
            decayed_arms[delta_key] = [decayed_alpha, decayed_beta]
        
        return decayed_arms
    
    def _find_or_create_srs_item(self, profile: MathProfile, question_id: str) -> Dict[str, Any]:
        """SRS item'ı bul veya oluştur"""
        
        for item in profile.srs_queue:
            if item.get("question_id") == question_id:
                return item
        
        # Yeni item oluştur
        new_item = {
            "question_id": question_id,
            "level": 0,
            "interval": 0,
            "ease_factor": self.config["srs_ease_factor"],
            "due_date": datetime.utcnow().isoformat(),
            "review_count": 0,
        }
        
        profile.srs_queue.append(new_item)
        return new_item
    
    def _calculate_response_quality(
        self, 
        is_correct: bool, 
        response_time: float, 
        difficulty: float,
        profile: MathProfile
    ) -> float:
        """Cevap kalitesini hesapla (0-5 scale)"""
        
        # Doğruluk puanı (0-3)
        correctness_score = 3 if is_correct else 0
        
        # Hız puanı (0-2)
        expected_time = 60 + (difficulty * 30)  # Zorluk arttıkça daha fazla zaman
        if response_time <= expected_time * 0.5:
            speed_score = 2  # Çok hızlı
        elif response_time <= expected_time:
            speed_score = 1.5  # Normal
        elif response_time <= expected_time * 2:
            speed_score = 1  # Yavaş
        else:
            speed_score = 0.5  # Çok yavaş
        
        return correctness_score + speed_score
    
    def _calculate_freshness_score(self, question: Question, profile: MathProfile) -> float:
        """Tazelik skorunu hesapla"""
        
        # Son görülme zamanına göre
        if question.last_seen_at:
            days_since_seen = (datetime.utcnow() - question.last_seen_at).days
            freshness = max(0, 1 - (days_since_seen / 30))  # 30 günde tamamen eski
        else:
            freshness = 1.0  # Hiç görülmemiş
        
        # Freshness score'a göre ayarla
        if question.freshness_score:
            freshness = (freshness + (1 - question.freshness_score)) / 2
        
        return freshness
    
    def _calculate_quality_score(self, question: Question) -> float:
        """Kalite skorunu hesapla"""
        
        quality_score = 0.5  # Temel skor
        
        # Quality flags'a göre ayarla
        if question.quality_flags:
            if question.quality_flags.get("reviewed", False):
                quality_score += 0.2
            if not question.quality_flags.get("ambiguous", True):
                quality_score += 0.2
            if question.quality_flags.get("verified", False):
                quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _calculate_diversity_score(
        self, 
        question: Question, 
        profile: MathProfile,
        context: Dict[str, Any]
    ) -> float:
        """Çeşitlilik skorunu hesapla"""
        
        # Son sorulan sorularla karşılaştır
        recent_questions = context.get("recent_questions", [])
        
        if not recent_questions:
            return 1.0  # İlk soru - maksimum çeşitlilik
        
        # Konu çeşitliliği
        topic_diversity = 1.0
        if question.topic_category:
            recent_topics = [q.get("topic_category") for q in recent_questions if q.get("topic_category")]
            if question.topic_category in recent_topics:
                topic_diversity = 0.5
        
        # Zorluk çeşitliliği
        difficulty_diversity = 1.0
        recent_difficulties = [q.get("difficulty_level", 3) for q in recent_questions]
        if recent_difficulties:
            avg_recent_difficulty = sum(recent_difficulties) / len(recent_difficulties)
            difficulty_diff = abs((question.estimated_difficulty or question.difficulty_level) - avg_recent_difficulty)
            difficulty_diversity = min(1.0, difficulty_diff / 2.0)
        
        return (topic_diversity + difficulty_diversity) / 2


# Global instance
advanced_math_algorithms = AdvancedMathAlgorithms()
