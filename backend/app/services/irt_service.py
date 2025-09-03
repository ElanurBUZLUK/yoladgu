"""
Item Response Theory (IRT) service for student ability estimation.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.optimize import minimize_scalar
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.repositories.user import user_repository
from app.db.repositories.attempt import attempt_repository


class IRTService:
    """IRT 2PL model implementation for theta estimation."""
    
    def __init__(self):
        self.learning_rate = settings.IRT_LEARNING_RATE
        self.max_iterations = settings.IRT_MAX_ITERATIONS
        self.convergence_threshold = settings.IRT_CONVERGENCE_THRESHOLD
    
    def probability_2pl(self, theta: float, a: float, b: float) -> float:
        """
        Calculate probability of correct response using 2PL model.
        
        P(θ, a, b) = 1 / (1 + exp(-a * (θ - b)))
        
        Args:
            theta: Student ability parameter
            a: Item discrimination parameter
            b: Item difficulty parameter
            
        Returns:
            Probability of correct response (0-1)
        """
        try:
            exponent = -a * (theta - b)
            # Prevent overflow
            if exponent > 700:
                return 0.0
            elif exponent < -700:
                return 1.0
            return 1.0 / (1.0 + math.exp(exponent))
        except (OverflowError, ValueError):
            return 0.5  # Default probability
    
    def log_likelihood(self, theta: float, responses: List[Tuple[int, float, float]]) -> float:
        """
        Calculate log-likelihood for given theta and response pattern.
        
        Args:
            theta: Student ability parameter
            responses: List of (response, a, b) tuples
            
        Returns:
            Log-likelihood value
        """
        log_likelihood = 0.0
        
        for response, a, b in responses:
            prob = self.probability_2pl(theta, a, b)
            
            # Prevent log(0)
            prob = max(min(prob, 0.9999), 0.0001)
            
            if response == 1:  # Correct
                log_likelihood += math.log(prob)
            else:  # Incorrect
                log_likelihood += math.log(1 - prob)
        
        return log_likelihood
    
    def estimate_theta_mle(self, responses: List[Tuple[int, float, float]]) -> float:
        """
        Estimate theta using Maximum Likelihood Estimation.
        
        Args:
            responses: List of (response, a, b) tuples
            
        Returns:
            Estimated theta value
        """
        if not responses:
            return 0.0
        
        # Define negative log-likelihood function for minimization
        def neg_log_likelihood(theta):
            return -self.log_likelihood(theta, responses)
        
        # Find theta that maximizes likelihood (minimizes negative log-likelihood)
        result = minimize_scalar(
            neg_log_likelihood,
            bounds=(-4.0, 4.0),
            method='bounded'
        )
        
        return result.x if result.success else 0.0
    
    def update_theta_online(
        self, 
        current_theta: float, 
        response: int, 
        a: float, 
        b: float
    ) -> float:
        """
        Update theta using online learning (gradient ascent).
        
        Args:
            current_theta: Current theta estimate
            response: Response (1=correct, 0=incorrect)
            a: Item discrimination parameter
            b: Item difficulty parameter
            
        Returns:
            Updated theta value
        """
        prob = self.probability_2pl(current_theta, a, b)
        
        # Calculate gradient of log-likelihood
        gradient = a * (response - prob)
        
        # Update theta using gradient ascent
        new_theta = current_theta + self.learning_rate * gradient
        
        # Bound theta to reasonable range
        return max(min(new_theta, 4.0), -4.0)
    
    async def update_user_theta(
        self,
        session: AsyncSession,
        user_id: str,
        item_type: str,
        response: int,
        item_a: float = 1.0,
        item_b: float = 0.0,
        use_online: bool = True
    ) -> Optional[float]:
        """
        Update user's theta based on new response.
        
        Args:
            session: Database session
            user_id: User ID
            item_type: "math" or "en"
            response: 1 for correct, 0 for incorrect
            item_a: Item discrimination parameter
            item_b: Item difficulty parameter
            use_online: Use online update vs full MLE
            
        Returns:
            Updated theta value
        """
        user = await user_repository.get(session, user_id)
        if not user:
            return None
        
        current_theta = user.theta_math if item_type == "math" else user.theta_en
        if current_theta is None:
            current_theta = 0.0
        
        if use_online:
            # Online update (faster)
            new_theta = self.update_theta_online(current_theta, response, item_a, item_b)
        else:
            # Full MLE update (more accurate but slower)
            recent_attempts = await attempt_repository.get_recent_attempts(
                session, user_id, hours=24*7, item_type=item_type  # Last week
            )
            
            # Convert attempts to response format
            responses = []
            for attempt in recent_attempts:
                # For now, use default parameters - in production, get from item
                responses.append((1 if attempt.correct else 0, 1.0, 0.0))
            
            # Add current response
            responses.append((response, item_a, item_b))
            
            new_theta = self.estimate_theta_mle(responses)
        
        # Update user theta
        if item_type == "math":
            await user_repository.update_theta(session, user_id, theta_math=new_theta)
        else:
            await user_repository.update_theta(session, user_id, theta_en=new_theta)
        
        return new_theta
    
    def get_difficulty_match_score(self, theta: float, item_b: float, tolerance: float = 0.3) -> float:
        """
        Calculate how well item difficulty matches student ability.
        
        Args:
            theta: Student ability
            item_b: Item difficulty
            tolerance: Acceptable difference
            
        Returns:
            Match score (1.0 = perfect match, 0.0 = poor match)
        """
        diff = abs(theta - item_b)
        if diff <= tolerance:
            return 1.0 - (diff / tolerance)
        else:
            return 0.0
    
    def predict_success_probability(self, theta: float, item_a: float, item_b: float) -> float:
        """
        Predict probability of success for given student and item.
        
        Args:
            theta: Student ability
            item_a: Item discrimination
            item_b: Item difficulty
            
        Returns:
            Predicted success probability
        """
        return self.probability_2pl(theta, item_a, item_b)
    
    def get_optimal_difficulty_range(
        self, 
        theta: float, 
        target_success_rate: float = 0.7,
        tolerance: float = 0.1
    ) -> Tuple[float, float]:
        """
        Get optimal difficulty range for given theta and target success rate.
        
        Args:
            theta: Student ability
            target_success_rate: Desired success probability
            tolerance: Acceptable range around target
            
        Returns:
            (min_difficulty, max_difficulty) tuple
        """
        # For 2PL with a=1, solve for b given target probability
        # P = 1/(1 + exp(-(θ-b))) = target_rate
        # exp(-(θ-b)) = (1-target_rate)/target_rate
        # -(θ-b) = ln((1-target_rate)/target_rate)
        # b = θ + ln((1-target_rate)/target_rate)
        
        try:
            ln_odds = math.log((1 - target_success_rate) / target_success_rate)
            optimal_b = theta + ln_odds
            
            # Create range around optimal difficulty
            min_b = optimal_b - tolerance
            max_b = optimal_b + tolerance
            
            return (min_b, max_b)
        except (ValueError, ZeroDivisionError):
            # Fallback to theta-centered range
            return (theta - 0.5, theta + 0.5)


class ErrorProfileService:
    """Service for managing student error profiles."""
    
    def __init__(self):
        self.decay_factor = 0.9  # How much to weight previous errors
        self.min_attempts = 3    # Minimum attempts before updating profile
    
    async def update_error_profile(
        self,
        session: AsyncSession,
        user_id: str,
        item_type: str,
        skill_tags: List[str],
        is_correct: bool
    ) -> Optional[Dict[str, float]]:
        """
        Update user's error profile based on attempt.
        
        Args:
            session: Database session
            user_id: User ID
            item_type: "math" or "en"
            skill_tags: Skills/error tags for the item
            is_correct: Whether the attempt was correct
            
        Returns:
            Updated error profile
        """
        if is_correct or not skill_tags:
            return None  # Only update on errors
        
        user = await user_repository.get(session, user_id)
        if not user:
            return None
        
        # Get current error profile
        current_profile = (
            user.error_profile_math if item_type == "math" 
            else user.error_profile_en
        ) or {}
        
        # Update error rates for each skill
        updated_profile = current_profile.copy()
        
        for skill in skill_tags:
            current_rate = updated_profile.get(skill, 0.0)
            # Increase error rate (weighted average with new error)
            new_rate = current_rate * self.decay_factor + (1 - self.decay_factor) * 1.0
            updated_profile[skill] = min(new_rate, 1.0)  # Cap at 1.0
        
        # Update in database
        await user_repository.update_error_profile(
            session, user_id, item_type, updated_profile
        )
        
        return updated_profile
    
    async def get_weak_skills(
        self,
        session: AsyncSession,
        user_id: str,
        item_type: str,
        threshold: float = 0.3
    ) -> List[str]:
        """
        Get skills where user has high error rates.
        
        Args:
            session: Database session
            user_id: User ID
            item_type: "math" or "en"
            threshold: Error rate threshold
            
        Returns:
            List of skills with high error rates
        """
        user = await user_repository.get(session, user_id)
        if not user:
            return []
        
        error_profile = (
            user.error_profile_math if item_type == "math"
            else user.error_profile_en
        ) or {}
        
        weak_skills = [
            skill for skill, error_rate in error_profile.items()
            if error_rate >= threshold
        ]
        
        return weak_skills
    
    def calculate_skill_priority(self, error_profile: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate priority scores for skills based on error rates.
        
        Args:
            error_profile: Skill -> error_rate mapping
            
        Returns:
            Skill -> priority_score mapping
        """
        if not error_profile:
            return {}
        
        # Higher error rate = higher priority
        max_error = max(error_profile.values())
        if max_error == 0:
            return {skill: 0.0 for skill in error_profile}
        
        priorities = {
            skill: error_rate / max_error
            for skill, error_rate in error_profile.items()
        }
        
        return priorities


# Create service instances
irt_service = IRTService()
error_profile_service = ErrorProfileService()