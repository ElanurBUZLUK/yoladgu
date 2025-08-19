import random
import math
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from datetime import datetime, timedelta
import logging

from app.models.question import Question, Subject
from app.models.math_profile import MathProfile
from app.services.cache_service import cache_service
from app.services.advanced_math_algorithms import advanced_math_algorithms

logger = logging.getLogger(__name__)


class MathSelector:
    """Matematik soru seçici - matematik.md'deki Selector durum makinesi"""
    
    def __init__(self):
        self.config = {
            "window_size": 20,
            "lambda_freshness": 0.2,
            "epsilon_random": 0.01,
            "gamma_recovery": 0.6,
            "ema_accuracy": 0.9,
            "ema_speed": 0.8,
            "bandit_deltas": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "srs_intervals": [0, 1, 3, 7, 16],  # days
        }
    
    def clip(self, x: float, a: float, b: float) -> float:
        """Değeri [a, b] aralığında sınırla"""
        return max(a, min(b, x))
    
    def sigmoid(self, x: float) -> float:
        """Sigmoid fonksiyonu"""
        return 1 / (1 + math.exp(-x))
    
    async def select_question(
        self, 
        db: AsyncSession, 
        profile: MathProfile,
        exclude_question_ids: Optional[List[str]] = None
    ) -> Tuple[Question, Dict[str, Any]]:
        """Öğrenci için en uygun matematik sorusunu seç"""
        
        # Öncelik sırası: Kurtarma > SRS > Normal
        if profile.needs_recovery():
            return await self._pick_recovery_question(db, profile, exclude_question_ids)
        
        if profile.has_due_srs():
            return await self._pick_srs_question(db, profile, exclude_question_ids)
        
        return await self._pick_normal_question(db, profile, exclude_question_ids)
    
    async def _pick_recovery_question(
        self, 
        db: AsyncSession, 
        profile: MathProfile,
        exclude_question_ids: Optional[List[str]] = None
    ) -> Tuple[Question, Dict[str, Any]]:
        """Kurtarma modu - kolay soru seç"""
        
        # Hedef zorluk: global_skill - gamma
        target_difficulty = profile.global_skill - self.config["gamma_recovery"]
        target_difficulty = self.clip(target_difficulty, 0.0, 5.0)
        
        # Zorluk aralığı
        target_low = max(0.0, target_difficulty - 0.5)
        target_high = min(5.0, target_difficulty + 0.5)
        
        # Soru havuzundan filtrele
        candidates = await self._filter_questions(
            db, target_low, target_high, exclude_question_ids
        )
        
        if not candidates:
            # Fallback: daha geniş aralık
            candidates = await self._filter_questions(
                db, 0.0, target_difficulty + 1.0, exclude_question_ids
            )
        
        if not candidates:
            raise ValueError("No suitable recovery questions found")
        
        # En kolay soruyu seç
        selected_question = min(candidates, key=lambda q: q.estimated_difficulty or q.difficulty_level)
        
        rationale = {
            "mode": "recovery",
            "target_difficulty": target_difficulty,
            "target_range": (target_low, target_high),
            "reason": f"Student needs recovery (streak_wrong={profile.streak_wrong}, ema_accuracy={profile.ema_accuracy:.2f})"
        }
        
        return selected_question, rationale
    
    async def _pick_srs_question(
        self, 
        db: AsyncSession, 
        profile: MathProfile,
        exclude_question_ids: Optional[List[str]] = None
    ) -> Tuple[Question, Dict[str, Any]]:
        """SRS modu - vadesi gelen tekrar sorusu seç"""
        
        # Vadesi gelen ilk SRS kartını al
        current_time = datetime.utcnow()
        due_srs_items = []
        
        for srs_item in profile.srs_queue:
            if srs_item.get("due_date"):
                due_date = datetime.fromisoformat(srs_item["due_date"])
                if current_time >= due_date:
                    due_srs_items.append(srs_item)
        
        if not due_srs_items:
            # Fallback to normal selection
            return await self._pick_normal_question(db, profile, exclude_question_ids)
        
        # İlk vadesi gelen soruyu seç
        srs_item = due_srs_items[0]
        question_id = srs_item.get("question_id")
        
        # Soruyu veritabanından al
        stmt = select(Question).where(Question.id == question_id)
        result = await db.execute(stmt)
        question = result.scalar_one_or_none()
        
        if not question:
            # Soru bulunamadı, SRS'den kaldır
            profile.srs_queue = [item for item in profile.srs_queue if item.get("question_id") != question_id]
            return await self._pick_normal_question(db, profile, exclude_question_ids)
        
        rationale = {
            "mode": "srs",
            "srs_item": srs_item,
            "reason": f"SRS card due for repetition (level={srs_item.get('level', 0)})"
        }
        
        return question, rationale
    
    async def _pick_normal_question(
        self, 
        db: AsyncSession, 
        profile: MathProfile,
        exclude_question_ids: Optional[List[str]] = None
    ) -> Tuple[Question, Dict[str, Any]]:
        """Normal akış - gelişmiş adaptif + bandit seçimi"""
        
        # Son performans verilerini al
        recent_performance = self._get_recent_performance_data(profile)
        
        # Gelişmiş adaptif zorluk hesaplama
        target_low, target_high, difficulty_rationale = advanced_math_algorithms.enhanced_adaptive_difficulty(
            profile, recent_performance
        )
        
        # Aday soruları filtrele
        candidates = await self._filter_questions(
            db, target_low, target_high, exclude_question_ids
        )
        
        if not candidates:
            # Fallback: daha geniş aralık
            candidates = await self._filter_questions(
                db, 0.0, 5.0, exclude_question_ids
            )
        
        if not candidates:
            raise ValueError("No suitable questions found")
        
        # Context hazırla
        context = {
            "needs_recovery": profile.needs_recovery(),
            "high_performance": profile.ema_accuracy > 0.8,
            "recent_questions": self._get_recent_questions_context(profile),
            "performance_metrics": difficulty_rationale["performance_metrics"],
        }
        
        # İlk 50 etkileşimde epsilon-greedy
        total_interactions = len(profile.last_k_outcomes or [])
        if total_interactions < 50:
            selected_question = self._epsilon_greedy_selection(candidates, target_low, target_high)
            rationale = {
                "mode": "normal_epsilon_greedy",
                "target_range": (target_low, target_high),
                "difficulty_rationale": difficulty_rationale,
                "reason": f"Epsilon-greedy selection (interactions={total_interactions})"
            }
        else:
            # Gelişmiş Thompson Sampling ile bandit seçimi
            delta, bandit_rationale = advanced_math_algorithms.advanced_thompson_sampling(profile, context)
            adj_low = self.clip(target_low + delta, 0.0, 5.0)
            adj_high = self.clip(target_high + delta, 0.0, 5.0)
            
            # Bandit ile filtrelenmiş adaylar
            bandit_candidates = await self._filter_questions(
                db, adj_low, adj_high, exclude_question_ids
            )
            
            if bandit_candidates:
                candidates = bandit_candidates
            
            # Gelişmiş skorlama algoritması
            scored_candidates = advanced_math_algorithms.advanced_scoring_algorithm(
                candidates, (target_low + target_high) / 2, profile, context
            )
            
            selected_question = scored_candidates[0][0] if scored_candidates else candidates[0]
            scoring_rationale = scored_candidates[0][2] if scored_candidates else {}
            
            rationale = {
                "mode": "normal_advanced_thompson",
                "target_range": (target_low, target_high),
                "difficulty_rationale": difficulty_rationale,
                "bandit_rationale": bandit_rationale,
                "scoring_rationale": scoring_rationale,
                "adjusted_range": (adj_low, adj_high),
                "reason": f"Advanced Thompson sampling with delta={delta}"
            }
        
        return selected_question, rationale
    
    async def _filter_questions(
        self, 
        db: AsyncSession, 
        target_low: float, 
        target_high: float,
        exclude_question_ids: Optional[List[str]] = None
    ) -> List[Question]:
        """Zorluk aralığına göre soruları filtrele"""
        
        conditions = [
            Question.subject == Subject.MATH,
            Question.estimated_difficulty >= target_low,
            Question.estimated_difficulty <= target_high
        ]
        
        if exclude_question_ids:
            conditions.append(Question.id.notin_(exclude_question_ids))
        
        stmt = select(Question).where(and_(*conditions))
        result = await db.execute(stmt)
        return result.scalars().all()
    
    def _epsilon_greedy_selection(
        self, 
        candidates: List[Question], 
        target_low: float, 
        target_high: float
    ) -> Question:
        """Epsilon-greedy seçim (ilk 50 etkileşim için)"""
        
        if random.random() < self.config["epsilon_random"]:
            # Random exploration
            return random.choice(candidates)
        else:
            # Greedy exploitation
            target = (target_low + target_high) / 2
            return min(candidates, key=lambda q: abs((q.estimated_difficulty or q.difficulty_level) - target))
    
    def _thompson_arm_selection(self, profile: MathProfile) -> float:
        """Thompson Sampling ile bandit arm seçimi"""
        
        best_delta = 0.0
        best_sample = -1
        
        for delta in self.config["bandit_deltas"]:
            delta_key = str(delta)
            if delta_key in profile.bandit_arms:
                alpha, beta = profile.bandit_arms[delta_key]
                sample = random.betavariate(alpha, beta)
                if sample > best_sample:
                    best_sample = sample
                    best_delta = delta
        
        return best_delta
    
    def _score_and_pick(
        self, 
        candidates: List[Question], 
        target_low: float, 
        target_high: float,
        profile: MathProfile
    ) -> Question:
        """Skorlama ve seçim"""
        
        target = (target_low + target_high) / 2
        
        def score_question(question: Question) -> float:
            # Temel mesafe skoru
            difficulty = question.estimated_difficulty or question.difficulty_level
            distance_score = abs(difficulty - target)
            
            # Tazelik skoru
            freshness = 1 - (question.freshness_score or 0)
            freshness_score = self.config["lambda_freshness"] * freshness
            
            # Rastgelelik
            random_score = random.random() * self.config["epsilon_random"]
            
            return distance_score + freshness_score + random_score
        
        return min(candidates, key=score_question)
    
    async def update_srs_after_answer(
        self, 
        profile: MathProfile, 
        question_id: str, 
        is_correct: bool
    ):
        """SRS queue'yu yanıt sonrası güncelle"""
        
        # SRS item'ı bul
        srs_item = None
        for item in profile.srs_queue:
            if item.get("question_id") == question_id:
                srs_item = item
                break
        
        if not srs_item:
            # Yeni SRS item oluştur
            srs_item = {
                "question_id": question_id,
                "level": 0,
                "due_date": datetime.utcnow().isoformat()
            }
            profile.srs_queue.append(srs_item)
        
        # Seviye güncelle
        if is_correct:
            srs_item["level"] = min(4, srs_item["level"] + 1)
        else:
            srs_item["level"] = max(0, srs_item["level"] - 1)
        
        # Yeni vade tarihi hesapla
        interval_days = self.config["srs_intervals"][srs_item["level"]]
        due_date = datetime.utcnow() + timedelta(days=interval_days)
        srs_item["due_date"] = due_date.isoformat()
        
        # Queue'yu güncelle
        profile.srs_queue = [item for item in profile.srs_queue if item.get("question_id") != question_id]
        profile.srs_queue.append(srs_item)
    
    def get_selection_statistics(self, profile: MathProfile) -> Dict[str, Any]:
        """Seçim istatistiklerini döndür"""
        
        return {
            "total_interactions": len(profile.last_k_outcomes or []),
            "current_streak_right": profile.streak_right,
            "current_streak_wrong": profile.streak_wrong,
            "ema_accuracy": profile.ema_accuracy,
            "ema_speed": profile.ema_speed,
            "global_skill": profile.global_skill,
            "difficulty_factor": profile.difficulty_factor,
            "needs_recovery": profile.needs_recovery(),
            "has_due_srs": profile.has_due_srs(),
            "srs_queue_size": len(profile.srs_queue),
            "target_difficulty_range": profile.get_target_difficulty_range(),
            "bandit_arms": profile.bandit_arms,
        }
    
    def _get_recent_performance_data(self, profile: MathProfile) -> List[Dict[str, Any]]:
        """Son performans verilerini al"""
        
        # Basit implementasyon - gerçekte veritabanından alınabilir
        outcomes = profile.last_k_outcomes or []
        performance_data = []
        
        for i, is_correct in enumerate(outcomes[-20:]):  # Son 20 sonuç
            performance_data.append({
                "is_correct": is_correct,
                "response_time": 60 + (i * 5),  # Simulated response time
                "difficulty": profile.global_skill + (i * 0.1),  # Simulated difficulty
                "timestamp": datetime.utcnow() - timedelta(hours=i)
            })
        
        return performance_data
    
    def _get_recent_questions_context(self, profile: MathProfile) -> List[Dict[str, Any]]:
        """Son soruların context'ini al"""
        
        # Basit implementasyon - gerçekte veritabanından alınabilir
        outcomes = profile.last_k_outcomes or []
        recent_questions = []
        
        for i, is_correct in enumerate(outcomes[-10:]):  # Son 10 soru
            recent_questions.append({
                "topic_category": f"topic_{i % 5}",  # Simulated topic
                "difficulty_level": 2 + (i % 4),  # Simulated difficulty
                "question_type": "multiple_choice",
                "is_correct": is_correct
            })
        
        return recent_questions


# Global instance
math_selector = MathSelector()
