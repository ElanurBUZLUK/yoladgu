"""
Service that selects the next math question for a user based on their profile.
"""

from typing import List, Optional, Dict, Any

from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.profile_service import profile_service
from app.db.repositories.attempt import attempt_repository
from app.db.models import MathItem


class MathNextQuestionService:
    """
    Kullanıcının profilini (theta_math + error_profile_math) kullanarak
    uygun bir sonraki math sorusunu seçer.
    Şimdilik sadece mevcut MathItem kayıtlarından seçim yapıyor.
    """

    async def _select_weak_skills(
        self,
        error_profile_math: Dict[str, Any],
        top_k: int = 3,
    ) -> List[str]:
        """
        error_profile_math kabaca şöyleyse:
        {
            "fractions": {"attempts": 10, "errors": 6, "error_rate": 0.6},
            "equations": {"attempts": 5, "errors": 1, "error_rate": 0.2},
            ...
        }
        error_rate'e göre en zayıf skill'leri al.
        """
        if not error_profile_math:
            return []

        sorted_skills = sorted(
            error_profile_math.items(),
            key=lambda kv: kv[1].get("error_rate", 0.0),
            reverse=True,
        )
        return [name for name, _stats in sorted_skills[:top_k]]

    async def _choose_item_simple(
        self,
        session: AsyncSession,
        theta_math: float,
        weak_skills: List[str],
    ) -> Optional[MathItem]:
        """
        Çok basit bir seçim:
        1. Eğer weak_skills boş değilse, skills'leri bunlardan biri olan sorulara bak.
        2. difficulty_b (veya senin kullandığın zorluk alanı) theta'ya en yakın olanı seç.
        3. Hiçbiri bulunamazsa rastgele bir soru seç.
        """
        query = select(MathItem).where(MathItem.status == "active")

        if weak_skills:
            skill_filters = [
                MathItem.skills.contains([skill])
                for skill in weak_skills
                if skill
            ]
            if skill_filters:
                query = query.where(or_(*skill_filters))

        query = query.order_by(func.abs(MathItem.difficulty_b - theta_math))

        result = await session.execute(query)
        items = result.scalars().all()

        if items:
            return items[0]

        fallback_query = (
            select(MathItem)
            .where(MathItem.status == "active")
            .order_by(func.random())
            .limit(1)
        )
        fallback_result = await session.execute(fallback_query)
        return fallback_result.scalars().first()

    async def get_next_question(
        self,
        session: AsyncSession,
        user_id: str,
    ) -> Optional[MathItem]:
        """
        1. Kullanıcının profilini al (theta_math + error_profile_math)
        2. (Opsiyonel) son attempt'leri al, ileride kullanmak için
        3. Weak skill'leri belirle
        4. Bu bilgilere göre uygun MathItem seç
        """
        profile = await profile_service.get_user_profile(
            session=session,
            user_id=user_id,
            use_cache=True,
        )

        if not profile:
            theta_math = 0.0
            error_profile_math: Dict[str, Any] = {}
        else:
            theta_math = profile.get("theta_math") or 0.0
            error_profile_math = profile.get("error_profile_math") or {}

        recent_attempts = await attempt_repository.get_recent_attempts(
            session=session,
            user_id=user_id,
            hours=24,
        )
        # TODO: recent_attempts verisini ileride daha sofistike analizde kullan.

        weak_skills = await self._select_weak_skills(
            error_profile_math=error_profile_math,
            top_k=3,
        )

        item = await self._choose_item_simple(
            session=session,
            theta_math=theta_math,
            weak_skills=weak_skills,
        )

        return item


math_next_question_service = MathNextQuestionService()
