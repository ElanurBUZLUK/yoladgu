from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from app.repositories.base_repository import BaseRepository
from app.models.error_pattern import ErrorPattern
from app.models.question import Subject
from app.schemas.error_pattern import ErrorPatternCreate, ErrorPatternUpdate


class ErrorPatternRepository(BaseRepository[ErrorPattern, ErrorPatternCreate, ErrorPatternUpdate]):
    """ErrorPattern repository - Hata kalıbı işlemleri için özel metodlar"""
    
    def __init__(self):
        super().__init__(ErrorPattern)

    async def get_by_user_and_type(
        self,
        db: AsyncSession,
        user_id: str,
        subject: Subject,
        error_type: str
    ) -> Optional[ErrorPattern]:
        """Kullanıcı, konu ve hata türüne göre hata kalıbını getir"""
        result = await db.execute(
            select(self.model)
            .where(
                and_(
                    self.model.user_id == user_id,
                    self.model.subject == subject,
                    self.model.error_type == error_type
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_top_n_errors_by_user(
        self,
        db: AsyncSession,
        user_id: str,
        subject: Subject,
        limit: int = 5
    ) -> List[ErrorPattern]:
        """Kullanıcının en sık yaptığı hata kalıplarını getir"""
        result = await db.execute(
            select(self.model)
            .where(
                and_(
                    self.model.user_id == user_id,
                    self.model.subject == subject
                )
            )
            .order_by(self.model.error_count.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def get_recent_error_patterns_by_user(
        self,
        db: AsyncSession,
        user_id: str,
        limit: int = 5
    ) -> List[ErrorPattern]:
        """Kullanıcının en son yaptığı hata kalıplarını getir"""
        result = await db.execute(
            select(self.model)
            .where(self.model.user_id == user_id)
            .order_by(self.model.last_occurrence.desc()) # Assuming last_occurrence is a DateTime or comparable field
            .limit(limit)
        )
        return result.scalars().all()

# Global error pattern repository instance
error_pattern_repository = ErrorPatternRepository()
