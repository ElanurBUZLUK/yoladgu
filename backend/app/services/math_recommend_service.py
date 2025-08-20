import logging
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.repositories.question_repository import QuestionRepository
from app.repositories.math_profile_repository import MathProfileRepository
from app.repositories.student_attempt_repository import StudentAttemptRepository
from app.repositories.error_pattern_repository import ErrorPatternRepository
from app.models.question import Question, DifficultyLevel, Subject
from app.models.math_profile import MathProfile
from app.utils.distlock_idem import idempotent_singleflight, IdempotencyConfig
from app.core.config import settings

logger = logging.getLogger(__name__)

class MathRecommendService:
    def __init__(
        self,
        question_repo: QuestionRepository,
        math_profile_repo: MathProfileRepository,
        student_attempt_repo: StudentAttemptRepository,
        error_pattern_repo: ErrorPatternRepository
    ):
        self.question_repo = question_repo
        self.math_profile_repo = math_profile_repo
        self.student_attempt_repo = student_attempt_repo
        self.error_pattern_repo = error_pattern_repo
        self.redis_client = Redis.from_url(settings.redis_url) # Initialize Redis client

    async def _recommend_questions_worker(
        self,
        session: AsyncSession,
        user_id: str,
        limit: int
    ) -> List[Question]:
        """Worker function for question recommendation logic."""
        user_math_profile = await self.math_profile_repo.get_by_user_id(session, user_id)
        if not user_math_profile:
            logger.warning(f"MathProfile not found for user {user_id}. Cannot recommend questions.")
            return []

        # 1. Determine target difficulty range (level+1 band)
        target_min_difficulty = user_math_profile.global_skill + 0.5
        target_max_difficulty = user_math_profile.global_skill + 1.5

        target_min_difficulty = max(0.0, target_min_difficulty)
        target_max_difficulty = min(5.0, target_max_difficulty)

        # 2. Get questions within the target difficulty range
        recommended_questions = await self.question_repo.get_by_subject_and_difficulty_range(
            session,
            subject=Subject.MATH,
            min_difficulty=target_min_difficulty,
            max_difficulty=target_max_difficulty,
            limit=limit
        )

        # 3. Incorporate neighbor-wrongs
        similar_students_profiles = await self.math_profile_repo.get_math_profiles_by_skill_range(
            session,
            min_skill=user_math_profile.global_skill - 0.5,
            max_skill=user_math_profile.global_skill + 0.5
        )
        similar_student_ids = [profile.user_id for profile in similar_students_profiles]

        wrongly_answered_questions_by_neighbors = await self.student_attempt_repo.get_wrongly_answered_questions_by_users(
            session,
            user_ids=similar_student_ids,
            limit=limit // 2
        )

        # Combine and deduplicate recommendations
        combined_recommendations = recommended_questions + wrongly_answered_questions_by_neighbors
        unique_question_ids = set()
        final_recommendations = []
        for q in combined_recommendations:
            if q.id not in unique_question_ids:
                final_recommendations.append(q)
                unique_question_ids.add(q.id)

        final_recommendations = final_recommendations[:limit]

        logger.info(f"Recommended {len(final_recommendations)} questions for user {user_id}.")
        # Convert Question objects to a serializable format (e.g., dict) for caching
        return [q.to_dict() for q in final_recommendations]

    async def recommend_questions(
        self,
        session: AsyncSession,
        user_id: str,
        limit: int = 5
    ) -> List[Question]:
        """Recommends math questions based on user's skill level and neighbor-wrongs, with idempotency."""
        idempotency_key = f"math_recommendation:{user_id}:{limit}"
        config = IdempotencyConfig(scope="math_recommendation", ttl_seconds=300) # Cache for 5 minutes

        try:
            # Use idempotent_singleflight to ensure only one recommendation process runs per user/limit
            # and to cache the results.
            recommended_questions_data = await idempotent_singleflight(
                client=self.redis_client,
                key=idempotency_key,
                config=config,
                worker=lambda: self._recommend_questions_worker(session, user_id, limit)
            )
            # Convert back from dict to Question objects if necessary, or adjust schema to return dicts
            # For now, assuming Question.from_dict() exists or similar reconstruction is handled upstream
            # If Question objects are complex, consider caching only IDs and fetching full objects later.
            # For simplicity, assuming to_dict() and from_dict() are available for Question model.
            return [Question(**q_data) for q_data in recommended_questions_data]

        except Exception as e:
            logger.error(f"Error in MathRecommendService for user {user_id}: {e}", exc_info=True)
            return []

# Global instance (for dependency injection)
math_recommend_service = MathRecommendService(
    question_repo=QuestionRepository(),
    math_profile_repo=MathProfileRepository(),
    student_attempt_repo=StudentAttemptRepository(),
    error_pattern_repo=ErrorPatternRepository()
)