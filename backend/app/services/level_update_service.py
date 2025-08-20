import logging
from typing import Dict, Any
from app.models.user import User
from app.models.math_profile import MathProfile
from app.repositories.user_repository import UserRepository
from app.repositories.math_profile_repository import MathProfileRepository
from app.schemas.math_profile import MathProfileCreate
from app.core.database import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

logger = logging.getLogger(__name__)

class LevelUpdateService:
    def __init__(self, user_repository: UserRepository, math_profile_repository: MathProfileRepository):
        self.user_repository = user_repository
        self.math_profile_repository = math_profile_repository
        self.K_FACTOR = 32 # ELO K-factor, can be adjusted

    async def update_skill_level(
        self,
        user_id: str,
        skill_name: str, # e.g., 'grammar', 'vocab', 'reading'
        outcome: bool, # True if correct, False if incorrect
        expected_outcome: float, # Expected probability of correct answer (0.0 to 1.0)
        session: AsyncSession # Inject session for DB operations
    ):
        """Updates a specific skill level for a user using an ELO-like formula.
        
        Args:
            user_id: The ID of the user.
            skill_name: The name of the skill to update.
            outcome: True if the answer was correct, False otherwise.
            expected_outcome: The expected probability of the user getting the answer correct
                              based on their current level and question difficulty.
            session: The database session.
        """
        try:
            user = await self.user_repository.get_by_id(session, user_id)
            if not user:
                logger.warning(f"User {user_id} not found for skill level update.")
                return

            # Initialize skill scores if they don't exist
            if not user.skill_scores:
                user.skill_scores = {}
            if skill_name not in user.skill_scores:
                user.skill_scores[skill_name] = 1500 # Default ELO rating

            current_score = user.skill_scores[skill_name]
            actual_outcome = 1 if outcome else 0

            # ELO-like update formula: new_score = current_score + K * (actual_outcome - expected_outcome)
            new_score = current_score + self.K_FACTOR * (actual_outcome - expected_outcome)
            user.skill_scores[skill_name] = round(new_score)

            # Update overall English level based on average or specific logic
            # For simplicity, let's average the main skills (grammar, vocab, reading)
            avg_score = 0
            skill_count = 0
            for skill in ['grammar', 'vocab', 'reading']:
                if skill in user.skill_scores:
                    avg_score += user.skill_scores[skill]
                    skill_count += 1
            
            if skill_count > 0:
                avg_score /= skill_count
                # Map ELO score to CEFR-like level (example mapping)
                if avg_score < 1600: # Example threshold
                    user.current_english_level = 1 # A1/A2
                elif avg_score < 1800:
                    user.current_english_level = 2 # B1
                elif avg_score < 2000:
                    user.current_english_level = 3 # B2
                else:
                    user.current_english_level = 4 # C1/C2

            await self.user_repository.update(session, user)
            logger.info(f"User {user_id} skill '{skill_name}' updated to {user.skill_scores[skill_name]}. Overall level: {user.current_english_level}")

        except Exception as e:
            logger.error(f"Error updating skill level for user {user_id}, skill {skill_name}: {e}", exc_info=True)
            raise

    async def update_math_skill_level(
        self,
        user_id: str,
        question_difficulty: float,
        is_correct: bool,
        time_ratio: float,
        delta_used: float,
        session: AsyncSession
    ):
        """Updates a user's math skill level using the MathProfile's internal logic.
        
        Args:
            user_id: The ID of the user.
            question_difficulty: The estimated difficulty of the question (0.0-5.0).
            is_correct: True if the answer was correct, False otherwise.
            time_ratio: Time taken to answer / average time for this question type.
            delta_used: The delta value used for question selection (from bandit arms).
            session: The database session.
        """
        try:
            math_profile = await self.math_profile_repository.get_by_user_id(session, user_id)
            
            if not math_profile:
                # Create a new MathProfile if one doesn't exist for the user
                new_profile_data = MathProfileCreate(user_id=user_id)
                math_profile = await self.math_profile_repository.create(session, new_profile_data)
                logger.info(f"Created new MathProfile for user {user_id}")

            math_profile.update_after_answer(question_difficulty, is_correct, time_ratio, delta_used)
            await self.math_profile_repository.update(session, math_profile, math_profile)
            logger.info(f"User {user_id} math skill updated. Global skill: {math_profile.global_skill}")

        except Exception as e:
            logger.error(f"Error updating math skill level for user {user_id}: {e}", exc_info=True)
            raise

async def get_level_update_service(session: AsyncSession = Depends(get_async_session)) -> LevelUpdateService:
    return LevelUpdateService(user_repository=UserRepository(), math_profile_repository=MathProfileRepository())

level_update_service = LevelUpdateService(user_repository=UserRepository(), math_profile_repository=MathProfileRepository())

