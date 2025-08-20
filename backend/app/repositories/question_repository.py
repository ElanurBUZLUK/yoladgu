from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.repositories.base_repository import BaseRepository
from app.models.question import Question, Subject
from app.schemas.question import QuestionCreate, QuestionUpdate


class QuestionRepository(BaseRepository[Question, QuestionCreate, QuestionUpdate]):
    """Question repository - Soru işlemleri için özel metodlar"""
    
    def __init__(self):
        super().__init__(Question)

    async def get_by_subject_and_difficulty_range(
        self,
        db: AsyncSession,
        subject: Subject,
        min_difficulty: float,
        max_difficulty: float,
        limit: int = 100
    ) -> List[Question]:
        """Belirli bir konu ve zorluk aralığındaki soruları getir"""
        result = await db.execute(
            select(self.model)
            .where(
                self.model.subject == subject,
                self.model.estimated_difficulty.between(min_difficulty, max_difficulty)
            )
            .limit(limit)
        )
        return result.scalars().all()

    async def get_random_by_subject_and_difficulty(
        self,
        db: AsyncSession,
        subject: Subject,
        difficulty: float
    ) -> Optional[Question]:
        """Belirli bir konu ve zorlukta rastgele bir soru getir"""
        # This is a simplified approach. For true randomness on large datasets,
        # consider more advanced techniques or pre-indexed random IDs.
        result = await db.execute(
            select(self.model)
            .where(
                self.model.subject == subject,
                self.model.estimated_difficulty == difficulty # Exact match for simplicity
            )
            .order_by(func.random())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_questions_by_topic_category(
        self,
        db: AsyncSession,
        subject: Subject,
        topic_category: str,
        limit: int = 100
    ) -> List[Question]:
        """Belirli bir konu ve kategoriye göre soruları getir"""
        result = await db.execute(
            select(self.model)
            .where(
                self.model.subject == subject,
                self.model.topic_category == topic_category
            )
            .limit(limit)
        )
        return result.scalars().all()

# Global question repository instance
question_repository = QuestionRepository()
