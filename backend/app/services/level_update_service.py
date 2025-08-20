import logging
from typing import Dict, Any, Optional
from app.models.user import User
from app.models.math_profile import MathProfile
from app.repositories.user_repository import UserRepository
from app.repositories.math_profile_repository import MathProfileRepository
from app.repositories.student_attempt_repository import StudentAttemptRepository
from app.schemas.math_profile import MathProfileCreate
from app.core.database import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
import math

logger = logging.getLogger(__name__)

class LevelUpdateService:
    def __init__(self, user_repository: UserRepository, math_profile_repository: MathProfileRepository, student_attempt_repository: StudentAttemptRepository):
        self.user_repository = user_repository
        self.math_profile_repository = math_profile_repository
        self.student_attempt_repository = student_attempt_repository
        self.K_FACTOR = 32 # ELO K-factor, can be adjusted
        self.PFA_ALPHA = 0.1 # PFA learning rate
        self.PFA_BETA = 0.05 # PFA forgetting rate

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

            # Apply PFA (Performance Factor Analysis) update
            await self._apply_pfa_update(math_profile, question_difficulty, is_correct, time_ratio, session)
            
            # Apply existing MathProfile update logic
            math_profile.update_after_answer(question_difficulty, is_correct, time_ratio, delta_used)
            await self.math_profile_repository.update(session, math_profile, math_profile)
            logger.info(f"User {user_id} math skill updated. Global skill: {math_profile.global_skill}")

        except Exception as e:
            logger.error(f"Error updating math skill level for user {user_id}: {e}", exc_info=True)
            raise

    async def _apply_pfa_update(
        self,
        math_profile: MathProfile,
        question_difficulty: float,
        is_correct: bool,
        time_ratio: float,
        session: AsyncSession
    ):
        """Applies Performance Factor Analysis (PFA) update to math profile."""
        try:
            # Initialize PFA parameters if not exists
            if not hasattr(math_profile, 'pfa_parameters'):
                math_profile.pfa_parameters = {
                    'alpha': self.PFA_ALPHA,
                    'beta': self.PFA_BETA,
                    'skill_estimates': {},
                    'last_update': None
                }

            # Calculate performance factor
            performance_factor = self._calculate_performance_factor(
                is_correct, time_ratio, question_difficulty
            )

            # Update skill estimates using PFA
            current_skill = math_profile.global_skill
            new_skill = current_skill + self.PFA_ALPHA * (performance_factor - current_skill)

            # Apply forgetting factor (skill decay over time)
            if math_profile.pfa_parameters['last_update']:
                time_diff = (math_profile.pfa_parameters['last_update'] - math_profile.pfa_parameters['last_update']).total_seconds()
                if time_diff > 86400:  # 24 hours
                    decay_factor = math.exp(-self.PFA_BETA * time_diff / 86400)
                    new_skill = new_skill * decay_factor

            # Update global skill
            math_profile.global_skill = max(0.0, min(5.0, new_skill))
            math_profile.pfa_parameters['last_update'] = math_profile.pfa_parameters.get('last_update')

            logger.info(f"PFA update applied: {current_skill:.3f} -> {math_profile.global_skill:.3f}")

        except Exception as e:
            logger.error(f"Error in PFA update: {e}", exc_info=True)

    def _calculate_performance_factor(self, is_correct: bool, time_ratio: float, difficulty: float) -> float:
        """Calculates performance factor based on correctness, time, and difficulty."""
        # Base performance (0-1)
        base_performance = 1.0 if is_correct else 0.0
        
        # Time penalty (if too slow)
        time_penalty = 0.0
        if time_ratio > 2.0:  # More than 2x average time
            time_penalty = min(0.3, (time_ratio - 2.0) * 0.1)
        
        # Difficulty bonus (correct answers on hard questions get bonus)
        difficulty_bonus = 0.0
        if is_correct and difficulty > 3.0:
            difficulty_bonus = min(0.2, (difficulty - 3.0) * 0.1)
        
        # Final performance factor
        performance_factor = base_performance - time_penalty + difficulty_bonus
        return max(0.0, min(1.0, performance_factor))

    async def trigger_automatic_update(
        self,
        user_id: str,
        question_id: str,
        is_correct: bool,
        time_taken: float,
        session: AsyncSession
    ):
        """Automatically triggers level update after student attempt."""
        try:
            # Get question details
            from app.repositories.question_repository import QuestionRepository
            question_repo = QuestionRepository()
            question = await question_repo.get_by_id(session, question_id)
            
            if not question:
                logger.warning(f"Question {question_id} not found for automatic update")
                return

            # Calculate time ratio (assuming average time is stored in question metadata)
            avg_time = question.question_metadata.get('average_time', 60.0) if question.question_metadata else 60.0
            time_ratio = time_taken / avg_time if avg_time > 0 else 1.0

            # Update based on subject
            if question.subject.value == "math":
                await self.update_math_skill_level(
                    user_id=user_id,
                    question_difficulty=float(question.difficulty_level),
                    is_correct=is_correct,
                    time_ratio=time_ratio,
                    delta_used=0.1,  # Default delta
                    session=session
                )
            elif question.subject.value == "english":
                # For English, we need to determine which skill was tested
                skill_name = self._determine_english_skill(question)
                expected_outcome = self._calculate_expected_outcome(user_id, question.difficulty_level, session)
                
                await self.update_skill_level(
                    user_id=user_id,
                    skill_name=skill_name,
                    outcome=is_correct,
                    expected_outcome=expected_outcome,
                    session=session
                )

            logger.info(f"Automatic level update completed for user {user_id}, question {question_id}")

        except Exception as e:
            logger.error(f"Error in automatic level update: {e}", exc_info=True)

    def _determine_english_skill(self, question) -> str:
        """Determines which English skill was tested based on question content."""
        # Simple heuristic based on question type and content
        if question.question_type.value == "fill_blank":
            return "grammar"
        elif "vocabulary" in question.topic_category.lower():
            return "vocab"
        else:
            return "reading"  # Default

    async def _calculate_expected_outcome(self, user_id: str, difficulty: int, session: AsyncSession) -> float:
        """Calculates expected outcome probability based on user's current level."""
        try:
            user = await self.user_repository.get_by_id(session, user_id)
            if not user or not user.skill_scores:
                return 0.5  # Default probability

            # Calculate average skill level
            avg_skill = sum(user.skill_scores.values()) / len(user.skill_scores)
            
            # Map skill to probability (simplified)
            skill_prob = avg_skill / 2000.0  # Normalize to 0-1
            difficulty_factor = 1.0 - (difficulty / 5.0)  # Higher difficulty = lower probability
            
            expected_prob = skill_prob * difficulty_factor
            return max(0.1, min(0.9, expected_prob))  # Clamp between 0.1 and 0.9

        except Exception as e:
            logger.error(f"Error calculating expected outcome: {e}")
            return 0.5

async def get_level_update_service(session: AsyncSession = Depends(get_async_session)) -> LevelUpdateService:
    return LevelUpdateService(
        user_repository=UserRepository(), 
        math_profile_repository=MathProfileRepository(),
        student_attempt_repository=StudentAttemptRepository()
    )

level_update_service = LevelUpdateService(
    user_repository=UserRepository(), 
    math_profile_repository=MathProfileRepository(),
    student_attempt_repository=StudentAttemptRepository()
)

