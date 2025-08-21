from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from collections import deque
import json

from app.database import Base


class MathProfile(Base):
    """Matematik öğrenci profili - matematik.md'deki StudentProfile modeli"""
    
    __tablename__ = "math_profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True)
    
    # Core skill metrics
    global_skill = Column(Float, default=2.5, nullable=False)  # Öğrenci yetenek kestirimi (0.0-5.0)
    difficulty_factor = Column(Float, default=1.0, nullable=False)  # Dinamik çarpan (0.1-1.5)
    
    # EMA (Exponential Moving Average) metrics
    ema_accuracy = Column(Float, default=0.5, nullable=False)  # Üstel hareketli ortalama doğruluk
    ema_speed = Column(Float, default=0.5, nullable=False)  # Normalize edilmiş hız skoru
    
    # Streak tracking
    streak_right = Column(Integer, default=0, nullable=False)  # Doğru cevap serisi
    streak_wrong = Column(Integer, default=0, nullable=False)  # Yanlış cevap serisi
    
    # Last K outcomes (stored as JSON)
    last_k_outcomes = Column(JSON, default=list, nullable=False)  # Son K sonuç (deque as list)
    
    # SRS (Spaced Repetition System) queue
    srs_queue = Column(JSON, default=list, nullable=False)  # SM-2 Lite planı
    
    # Bandit arms for Thompson Sampling
    bandit_arms = Column(JSON, default=dict, nullable=False)  # {delta: (alpha, beta)}
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="math_profile")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize default bandit arms
        if not self.bandit_arms:
            self.bandit_arms = {
                "-1.0": [1, 1],  # (alpha, beta) for delta = -1.0
                "-0.5": [1, 1],  # (alpha, beta) for delta = -0.5
                "0.0": [1, 1],   # (alpha, beta) for delta = 0.0
                "0.5": [1, 1],   # (alpha, beta) for delta = 0.5
                "1.0": [1, 1],   # (alpha, beta) for delta = 1.0
            }
    
    def get_last_k_outcomes(self, k: int = 5) -> deque:
        """Son K sonucu deque olarak döndür"""
        outcomes = self.last_k_outcomes or []
        return deque(outcomes[-k:], maxlen=k)
    
    def add_outcome(self, is_correct: bool):
        """Yeni sonuç ekle"""
        outcomes = self.last_k_outcomes or []
        outcomes.append(is_correct)
        # Keep only last 20 outcomes
        self.last_k_outcomes = outcomes[-20:]
        
        # Update streaks
        if is_correct:
            self.streak_right += 1
            self.streak_wrong = 0
        else:
            self.streak_wrong += 1
            self.streak_right = 0
    
    def get_last_accuracy(self, k: int = 5) -> float:
        """Son K sonucun doğruluk oranını hesapla"""
        outcomes = self.get_last_k_outcomes(k)
        if not outcomes:
            return 0.5
        return sum(outcomes) / len(outcomes)
    
    def needs_recovery(self) -> bool:
        """Kurtarma moduna geçmeli mi?"""
        return (
            self.streak_wrong >= 2 or
            self.get_last_accuracy(5) < 0.4 or
            self.ema_accuracy < 0.35
        )
    
    def has_due_srs(self) -> bool:
        """Vadesi gelen SRS kartı var mı?"""
        if not self.srs_queue:
            return False
        
        current_time = datetime.utcnow()
        for srs_item in self.srs_queue:
            if srs_item.get("due_date"):
                due_date = datetime.fromisoformat(srs_item["due_date"])
                if current_time >= due_date:
                    return True
        return False
    
    def get_target_difficulty_range(self) -> tuple[float, float]:
        """Hedef zorluk aralığını hesapla"""
        base = self.global_skill * self.difficulty_factor
        
        # Son 5 doğruluk > 0.85 ise aralığı yukarı kaydır
        if self.get_last_accuracy(5) > 0.85:
            return (base + 0.7, base + 1.3)
        else:
            return (base - 0.3, base + 0.5)
    
    def update_after_answer(self, question_difficulty: float, is_correct: bool, time_ratio: float, delta_used: float):
        """Yanıt sonrası profili güncelle"""
        # Update EMA metrics
        self.ema_accuracy = 0.9 * self.ema_accuracy + 0.1 * (1 if is_correct else 0)
        self.ema_speed = 0.8 * self.ema_speed + 0.2 * (1 - min(time_ratio, 2.0))
        
        # Update global skill
        import math
        err = question_difficulty - self.global_skill
        skill_delta = 0.1 * (1 if is_correct else -1) * (1 / (1 + math.exp(-err)))
        self.global_skill = max(0.0, min(5.0, self.global_skill + skill_delta))
        
        # Update difficulty factor
        df_delta = 0.02 * (1 if is_correct else -1) + 0.01 * (1 - min(time_ratio, 2.0))
        self.difficulty_factor = max(0.1, min(1.5, self.difficulty_factor + df_delta))
        
        # Update bandit arms
        delta_key = str(delta_used)
        if delta_key in self.bandit_arms:
            alpha, beta = self.bandit_arms[delta_key]
            if is_correct:
                alpha += 1
            else:
                beta += 1
            self.bandit_arms[delta_key] = [alpha, beta]
        
        # Add outcome
        self.add_outcome(is_correct)
    
    def to_dict(self) -> dict:
        """Profil verilerini dict olarak döndür"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "global_skill": self.global_skill,
            "difficulty_factor": self.difficulty_factor,
            "ema_accuracy": self.ema_accuracy,
            "ema_speed": self.ema_speed,
            "streak_right": self.streak_right,
            "streak_wrong": self.streak_wrong,
            "last_k_outcomes": self.last_k_outcomes,
            "srs_queue": self.srs_queue,
            "bandit_arms": self.bandit_arms,
            "needs_recovery": self.needs_recovery(),
            "has_due_srs": self.has_due_srs(),
            "target_difficulty_range": self.get_target_difficulty_range(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
