from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
import logging

from app.models.math_profile import MathProfile
from app.models.user import User
from app.services.math_selector import math_selector
from app.core.cache import cache_service

logger = logging.getLogger(__name__)


class MathProfileManager:
    """Matematik profil yönetici servisi"""
    
    def __init__(self):
        pass
    
    async def get_or_create_profile(
        self, 
        db: AsyncSession, 
        user: User
    ) -> MathProfile:
        """Öğrenci için matematik profilini al veya oluştur"""
        
        # Cache'den kontrol et
        cache_key = f"math_profile:{user.id}"
        cached_profile = await cache_service.get(cache_key)
        if cached_profile:
            return MathProfile(**cached_profile)
        
        # Veritabanından kontrol et
        stmt = select(MathProfile).where(MathProfile.user_id == user.id)
        result = await db.execute(stmt)
        profile = result.scalar_one_or_none()
        
        if not profile:
            # Yeni profil oluştur
            profile = MathProfile(user_id=user.id)
            db.add(profile)
            await db.commit()
            await db.refresh(profile)
            
            logger.info(f"Created new math profile for user {user.id}")
        
        # Cache'e kaydet
        await cache_service.set(cache_key, profile.to_dict(), ttl=3600)
        
        return profile
    
    async def update_profile_after_answer(
        self,
        db: AsyncSession,
        profile: MathProfile,
        question_difficulty: float,
        is_correct: bool,
        time_taken: float,
        expected_time: float = 60.0,  # varsayılan 60 saniye
        partial_credit: Optional[float] = None
    ) -> MathProfile:
        """Yanıt sonrası profili güncelle"""
        
        # Kısmi puan varsa doğruluk oranını hesapla
        if partial_credit is not None:
            is_correct = partial_credit >= 0.5  # %50 üzeri doğru sayılır
        
        # Zaman oranını hesapla
        time_ratio = time_taken / expected_time if expected_time > 0 else 1.0
        
        # Bandit delta'sını hesapla (basit yaklaşım)
        delta_used = 0.0  # Normal seçim için 0.0
        
        # Profili güncelle
        profile.update_after_answer(
            question_difficulty=question_difficulty,
            is_correct=is_correct,
            time_ratio=time_ratio,
            delta_used=delta_used
        )
        
        # SRS queue'yu güncelle
        await math_selector.update_srs_after_answer(
            profile=profile,
            question_id=str(profile.id),  # Bu kısım düzeltilmeli
            is_correct=is_correct
        )
        
        # Veritabanını güncelle
        await db.commit()
        await db.refresh(profile)
        
        # Cache'i güncelle
        cache_key = f"math_profile:{profile.user_id}"
        await cache_service.set(cache_key, profile.to_dict(), ttl=3600)
        
        logger.info(f"Updated math profile for user {profile.user_id}: correct={is_correct}, skill={profile.global_skill:.2f}")
        
        return profile
    
    async def get_profile_statistics(
        self,
        profile: MathProfile
    ) -> Dict[str, Any]:
        """Profil istatistiklerini döndür"""
        
        stats = profile.to_dict()
        stats.update({
            "selection_stats": math_selector.get_selection_statistics(profile),
            "performance_metrics": {
                "accuracy_trend": self._calculate_accuracy_trend(profile),
                "speed_trend": self._calculate_speed_trend(profile),
                "skill_progression": self._calculate_skill_progression(profile),
                "difficulty_adjustment": self._calculate_difficulty_adjustment(profile),
            },
            "recommendations": self._generate_recommendations(profile),
        })
        
        return stats
    
    def _calculate_accuracy_trend(self, profile: MathProfile) -> Dict[str, Any]:
        """Doğruluk trendini hesapla"""
        
        outcomes = profile.last_k_outcomes or []
        if len(outcomes) < 5:
            return {"trend": "insufficient_data", "value": profile.ema_accuracy}
        
        recent_5 = outcomes[-5:]
        recent_10 = outcomes[-10:] if len(outcomes) >= 10 else outcomes
        
        recent_5_acc = sum(recent_5) / len(recent_5)
        recent_10_acc = sum(recent_10) / len(recent_10)
        
        if recent_5_acc > recent_10_acc + 0.1:
            trend = "improving"
        elif recent_5_acc < recent_10_acc - 0.1:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "current": profile.ema_accuracy,
            "recent_5": recent_5_acc,
            "recent_10": recent_10_acc,
        }
    
    def _calculate_speed_trend(self, profile: MathProfile) -> Dict[str, Any]:
        """Hız trendini hesapla"""
        
        return {
            "current_speed": profile.ema_speed,
            "trend": "stable",  # Basit implementasyon
        }
    
    def _calculate_skill_progression(self, profile: MathProfile) -> Dict[str, Any]:
        """Yetenek ilerlemesini hesapla"""
        
        # Basit implementasyon - gerçekte daha karmaşık olabilir
        skill_level = "beginner"
        if profile.global_skill >= 4.0:
            skill_level = "advanced"
        elif profile.global_skill >= 2.5:
            skill_level = "intermediate"
        
        return {
            "current_skill": profile.global_skill,
            "skill_level": skill_level,
            "progression_rate": "stable",  # Basit implementasyon
        }
    
    def _calculate_difficulty_adjustment(self, profile: MathProfile) -> Dict[str, Any]:
        """Zorluk ayarlamasını hesapla"""
        
        target_low, target_high = profile.get_target_difficulty_range()
        current_difficulty = profile.global_skill * profile.difficulty_factor
        
        adjustment_needed = "none"
        if current_difficulty < target_low:
            adjustment_needed = "increase"
        elif current_difficulty > target_high:
            adjustment_needed = "decrease"
        
        return {
            "current_difficulty": current_difficulty,
            "target_range": (target_low, target_high),
            "adjustment_needed": adjustment_needed,
            "difficulty_factor": profile.difficulty_factor,
        }
    
    def _generate_recommendations(self, profile: MathProfile) -> Dict[str, Any]:
        """Öğrenci için öneriler oluştur"""
        
        recommendations = []
        
        # Kurtarma modu önerisi
        if profile.needs_recovery():
            recommendations.append({
                "type": "recovery",
                "priority": "high",
                "message": "Kolay sorularla pratik yaparak güveninizi geri kazanın",
                "action": "practice_easy_questions"
            })
        
        # SRS önerisi
        if profile.has_due_srs():
            recommendations.append({
                "type": "srs",
                "priority": "medium",
                "message": "Tekrar gereken sorularınız var",
                "action": "review_srs_questions"
            })
        
        # Doğruluk önerisi
        if profile.ema_accuracy < 0.6:
            recommendations.append({
                "type": "accuracy",
                "priority": "medium",
                "message": "Doğruluk oranınızı artırmak için daha dikkatli çalışın",
                "action": "focus_on_accuracy"
            })
        
        # Hız önerisi
        if profile.ema_speed < 0.4:
            recommendations.append({
                "type": "speed",
                "priority": "low",
                "message": "Çözüm hızınızı artırmak için daha fazla pratik yapın",
                "action": "practice_for_speed"
            })
        
        # Genel öneri
        if not recommendations:
            recommendations.append({
                "type": "general",
                "priority": "low",
                "message": "Harika gidiyorsunuz! Zorluk seviyesini artırmaya hazırsınız",
                "action": "increase_difficulty"
            })
        
        return {
            "recommendations": recommendations,
            "next_actions": [rec["action"] for rec in recommendations],
        }
    
    async def reset_profile(
        self,
        db: AsyncSession,
        profile: MathProfile
    ) -> MathProfile:
        """Profili sıfırla (test amaçlı)"""
        
        profile.global_skill = 2.5
        profile.difficulty_factor = 1.0
        profile.ema_accuracy = 0.5
        profile.ema_speed = 0.5
        profile.streak_right = 0
        profile.streak_wrong = 0
        profile.last_k_outcomes = []
        profile.srs_queue = []
        profile.bandit_arms = {
            "-1.0": [1, 1],
            "-0.5": [1, 1],
            "0.0": [1, 1],
            "0.5": [1, 1],
            "1.0": [1, 1],
        }
        
        await db.commit()
        await db.refresh(profile)
        
        # Cache'i güncelle
        cache_key = f"math_profile:{profile.user_id}"
        await cache_service.set(cache_key, profile.to_dict(), ttl=3600)
        
        logger.info(f"Reset math profile for user {profile.user_id}")
        
        return profile
    
    async def get_learning_path(
        self,
        profile: MathProfile
    ) -> Dict[str, Any]:
        """Öğrenme yolunu hesapla"""
        
        current_skill = profile.global_skill
        target_skill = min(5.0, current_skill + 0.5)  # Bir sonraki hedef
        
        # Basit öğrenme yolu
        learning_path = {
            "current_level": self._get_skill_level(current_skill),
            "target_level": self._get_skill_level(target_skill),
            "progress_percentage": (current_skill / 5.0) * 100,
            "next_milestone": target_skill,
            "estimated_questions": self._estimate_questions_needed(current_skill, target_skill),
            "focus_areas": self._get_focus_areas(profile),
        }
        
        return learning_path
    
    def _get_skill_level(self, skill: float) -> str:
        """Yetenek seviyesini belirle"""
        if skill >= 4.5:
            return "expert"
        elif skill >= 3.5:
            return "advanced"
        elif skill >= 2.5:
            return "intermediate"
        elif skill >= 1.5:
            return "beginner"
        else:
            return "novice"
    
    def _estimate_questions_needed(self, current: float, target: float) -> int:
        """Hedef seviyeye ulaşmak için gereken soru sayısını tahmin et"""
        skill_gap = target - current
        # Basit tahmin: her 0.1 skill artışı için 10 soru
        return max(10, int(skill_gap * 100))
    
    def _get_focus_areas(self, profile: MathProfile) -> Dict[str, Any]:
        """Odaklanılması gereken alanları belirle"""
        
        focus_areas = []
        
        if profile.ema_accuracy < 0.7:
            focus_areas.append("accuracy")
        
        if profile.ema_speed < 0.6:
            focus_areas.append("speed")
        
        if profile.streak_wrong > 0:
            focus_areas.append("consistency")
        
        if not focus_areas:
            focus_areas.append("advanced_concepts")
        
        return {
            "primary": focus_areas[0] if focus_areas else "general",
            "secondary": focus_areas[1:] if len(focus_areas) > 1 else [],
            "all_areas": focus_areas,
        }


# Global instance
math_profile_manager = MathProfileManager()
