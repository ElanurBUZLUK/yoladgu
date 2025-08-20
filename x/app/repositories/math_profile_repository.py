from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.repositories.base_repository import BaseRepository
from app.models.math_profile import MathProfile
from app.schemas.math_profile import MathProfileCreate, MathProfileUpdate # Assuming these schemas exist or will be created


class MathProfileRepository(BaseRepository[MathProfile, MathProfileCreate, MathProfileUpdate]):
    """MathProfile repository - Matematik profili işlemleri için özel metodlar"""
    
    def __init__(self):
        super().__init__(MathProfile)
    
    async def get_by_user_id(self, db: AsyncSession, user_id: str) -> Optional[MathProfile]:
        """Kullanıcı ID'sine göre matematik profilini getir"""
        result = await db.execute(select(self.model).where(self.model.user_id == user_id))
        return result.scalar_one_or_none()

    async def get_math_profiles_by_skill_range(
        self,
        db: AsyncSession,
        min_skill: float,
        max_skill: float,
        limit: int = 100
    ) -> List[MathProfile]:
        """Belirli bir yetenek aralığındaki matematik profillerini getir"""
        result = await db.execute(
            select(self.model)
            .where(
                self.model.global_skill.between(min_skill, max_skill)
            )
            .limit(limit)
        )
        return result.scalars().all()

# Global math profile repository instance
math_profile_repository = MathProfileRepository()
