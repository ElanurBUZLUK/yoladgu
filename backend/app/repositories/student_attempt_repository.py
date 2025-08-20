from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.repositories.base_repository import BaseRepository
from app.models.student_attempt import StudentAttempt
from app.models.question import Question # Import Question model
from app.schemas.student_attempt import StudentAttemptCreate, StudentAttemptUpdate


class StudentAttemptRepository(BaseRepository[StudentAttempt, StudentAttemptCreate, StudentAttemptUpdate]):
    """StudentAttempt repository - Öğrenci deneme işlemleri için özel metodlar"""
    
    def __init__(self):
        super().__init__(StudentAttempt)

    async def get_attempts_by_user_id(
        self,
        db: AsyncSession,
        user_id: str,
        limit: int = 100
    ) -> List[StudentAttempt]:
        """Belirli bir kullanıcıya ait tüm denemeleri getir"""
        result = await db.execute(
            select(self.model)
            .where(self.model.user_id == user_id)
            .limit(limit)
        )
        return result.scalars().all()

    async def get_incorrect_attempts_by_user_id(
        self,
        db: AsyncSession,
        user_id: str,
        limit: int = 100
    ) -> List[StudentAttempt]:
        """Belirli bir kullanıcıya ait yanlış denemeleri getir"""
        result = await db.execute(
            select(self.model)
            .where(
                self.model.user_id == user_id,
                self.model.is_correct == False
            )
            .limit(limit)
        )
        return result.scalars().all()

    async def get_incorrect_attempts_for_question(
        self,
        db: AsyncSession,
        question_id: str,
        limit: int = 100
    ) -> List[StudentAttempt]:
        """Belirli bir soruya ait yanlış denemeleri getir"""
        result = await db.execute(
            select(self.model)
            .where(
                self.model.question_id == question_id,
                self.model.is_correct == False
            )
            .limit(limit)
        )
        return result.scalars().all()

    async def get_wrongly_answered_questions_by_users(
        self,
        db: AsyncSession,
        user_ids: List[str],
        limit: int = 10
    ) -> List[Question]:
        """Belirli kullanıcılar tarafından yanlış cevaplanan soruları getir"""
        result = await db.execute(
            select(Question)
            .join(StudentAttempt, StudentAttempt.question_id == Question.id)
            .where(
                StudentAttempt.user_id.in_(user_ids),
                StudentAttempt.is_correct == False
            )
            .distinct(Question.id) # Ensure unique questions
            .limit(limit)
        )
        return result.scalars().all()

# Global student attempt repository instance
student_attempt_repository = StudentAttemptRepository()
