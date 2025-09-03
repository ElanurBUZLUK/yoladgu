"""
Bandit algorithms for adaptive question selection (LinUCB, LinTS, Constrained Bandit).
"""

import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from app.core.config import settings
from app.db.repositories.user import user_repository
from app.db.repositories.attempt import attempt_repository
from app.db.models import Decision


class ContextualBandit:
    """Base class for contextual bandit algorithms."""
    
    def __init__(self, alpha: float = 1.0, lambda_reg: float = 0.01):
        self.alpha = alpha  # Exploration parameter
        self.lambda_reg = lambda_reg  # Regularization parameter
        self.feature_dim = None
        self.arms = {}  # arm_id -> arm_data mapping
    
    def build_context_vector(
        self,
        user_profile: Dict[str, Any],
        item_metadata: Dict[str, Any],
        target_skills: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Build context vector from user profile and item metadata.
        
        Context features:
        - User theta (math/english)
        - Skill match indicators
        - Device type
        - Recency features
        - Difficulty match
        """
        features = []
        
        # User ability features
        theta_math = user_profile.get("theta_math", 0.0)
        theta_en = user_profile.get("theta_en", 0.0)
        features.extend([theta_math, theta_en])
        
        # Item difficulty features
        item_difficulty = item_metadata.get("difficulty_b", 0.0)
        difficulty_match = abs(theta_math - item_difficulty) if item_metadata.get("type") == "math" else 0.0
        features.extend([item_difficulty, difficulty_match])
        
        # Skill match features (one-hot encoding for top skills)
        common_skills = ["linear_equation", "algebra", "geometry", "prepositions", "articles", "grammar"]
        item_skills = item_metadata.get("skills", []) or item_metadata.get("error_tags", [])
        
        for skill in common_skills:
            skill_match = 1.0 if skill in item_skills else 0.0
            features.append(skill_match)
        
        # Target skill match
        target_match = 0.0
        if target_skills and item_skills:
            target_match = len(set(target_skills) & set(item_skills)) / len(target_skills)
        features.append(target_match)
        
        # Device and context features
        preferences = user_profile.get("preferences", {})
        device_mobile = 1.0 if preferences.get("device") == "mobile" else 0.0
        features.append(device_mobile)
        
        # Performance features
        performance = user_profile.get("performance", {})
        item_type = item_metadata.get("type", "math")
        type_performance = performance.get(item_type, {})
        success_rate = type_performance.get("success_rate", 0.5)
        features.append(success_rate)
        
        # Recency features
        recent_attempts = type_performance.get("total_attempts", 0)
        recency_score = min(recent_attempts / 10.0, 1.0)  # Normalize to [0,1]
        features.append(recency_score)
        
        # Error profile match
        error_profile = user_profile.get(f"error_profile_{item_type}", {})
        error_match = 0.0
        if error_profile and item_skills:
            error_rates = [error_profile.get(skill, 0.0) for skill in item_skills]
            error_match = np.mean(error_rates) if error_rates else 0.0
        features.append(error_match)
        
        return np.array(features, dtype=np.float32)
    
    def add_arm(self, arm_id: str, arm_data: Dict[str, Any]) -> None:
        """Add an arm (question candidate) to the bandit."""
        self.arms[arm_id] = arm_data
    
    def select_arm(self, context: np.ndarray) -> Tuple[str, float]:
        """Select arm based on context. To be implemented by subclasses."""
        raise NotImplementedError
    
    def update(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        """Update bandit parameters based on observed reward."""
        raise NotImplementedError


class LinUCB(ContextualBandit):
    """Linear Upper Confidence Bound algorithm."""
    
    def __init__(self, alpha: float = 1.0, lambda_reg: float = 0.01):
        super().__init__(alpha, lambda_reg)
        self.A = {}  # arm_id -> A matrix (d x d)
        self.b = {}  # arm_id -> b vector (d x 1)
        self.theta = {}  # arm_id -> theta vector (d x 1)
    
    def _initialize_arm(self, arm_id: str, feature_dim: int) -> None:
        """Initialize matrices for a new arm."""
        self.A[arm_id] = np.eye(feature_dim) * self.lambda_reg
        self.b[arm_id] = np.zeros(feature_dim)
        self.theta[arm_id] = np.zeros(feature_dim)
    
    def select_arm(self, context: np.ndarray) -> Tuple[str, float]:
        """
        Select arm with highest upper confidence bound.
        
        UCB = theta^T * x + alpha * sqrt(x^T * A^-1 * x)
        """
        if not self.arms:
            raise ValueError("No arms available")
        
        feature_dim = len(context)
        if self.feature_dim is None:
            self.feature_dim = feature_dim
        
        best_arm = None
        best_ucb = -float('inf')
        arm_scores = {}
        
        for arm_id in self.arms:
            # Initialize arm if needed
            if arm_id not in self.A:
                self._initialize_arm(arm_id, feature_dim)
            
            # Calculate UCB
            A_inv = np.linalg.inv(self.A[arm_id])
            theta = A_inv @ self.b[arm_id]
            
            # Expected reward
            expected_reward = theta.T @ context
            
            # Confidence interval
            confidence = self.alpha * np.sqrt(context.T @ A_inv @ context)
            
            # Upper confidence bound
            ucb = expected_reward + confidence
            
            arm_scores[arm_id] = {
                "expected_reward": float(expected_reward),
                "confidence": float(confidence),
                "ucb": float(ucb)
            }
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm_id
        
        return best_arm, arm_scores
    
    def update(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        """Update LinUCB parameters."""
        if arm_id not in self.A:
            self._initialize_arm(arm_id, len(context))
        
        # Update A and b matrices
        self.A[arm_id] += np.outer(context, context)
        self.b[arm_id] += reward * context
        
        # Update theta
        A_inv = np.linalg.inv(self.A[arm_id])
        self.theta[arm_id] = A_inv @ self.b[arm_id]


class LinTS(ContextualBandit):
    """Linear Thompson Sampling algorithm."""
    
    def __init__(self, alpha: float = 1.0, lambda_reg: float = 0.01, noise_var: float = 1.0):
        super().__init__(alpha, lambda_reg)
        self.noise_var = noise_var
        self.A = {}  # arm_id -> A matrix
        self.b = {}  # arm_id -> b vector
    
    def _initialize_arm(self, arm_id: str, feature_dim: int) -> None:
        """Initialize matrices for a new arm."""
        self.A[arm_id] = np.eye(feature_dim) * self.lambda_reg
        self.b[arm_id] = np.zeros(feature_dim)
    
    def select_arm(self, context: np.ndarray) -> Tuple[str, float]:
        """
        Select arm using Thompson Sampling.
        
        Sample theta from posterior and select arm with highest expected reward.
        """
        if not self.arms:
            raise ValueError("No arms available")
        
        feature_dim = len(context)
        if self.feature_dim is None:
            self.feature_dim = feature_dim
        
        best_arm = None
        best_reward = -float('inf')
        arm_scores = {}
        
        for arm_id in self.arms:
            # Initialize arm if needed
            if arm_id not in self.A:
                self._initialize_arm(arm_id, feature_dim)
            
            # Sample theta from posterior
            A_inv = np.linalg.inv(self.A[arm_id])
            mu = A_inv @ self.b[arm_id]
            cov = self.noise_var * A_inv
            
            # Sample from multivariate normal
            sampled_theta = np.random.multivariate_normal(mu, cov)
            
            # Calculate expected reward
            expected_reward = sampled_theta.T @ context
            
            arm_scores[arm_id] = {
                "sampled_reward": float(expected_reward),
                "posterior_mean": mu.tolist(),
                "posterior_var": np.diag(cov).tolist()
            }
            
            if expected_reward > best_reward:
                best_reward = expected_reward
                best_arm = arm_id
        
        return best_arm, arm_scores
    
    def update(self, arm_id: str, context: np.ndarray, reward: float) -> None:
        """Update LinTS parameters."""
        if arm_id not in self.A:
            self._initialize_arm(arm_id, len(context))
        
        # Update A and b matrices
        self.A[arm_id] += np.outer(context, context)
        self.b[arm_id] += reward * context


class ConstrainedBandit:
    """Constrained bandit with success rate and coverage constraints."""
    
    def __init__(
        self,
        base_bandit: ContextualBandit,
        min_success_rate: float = 0.6,
        min_coverage: float = 0.8,
        window_size: int = 100
    ):
        self.base_bandit = base_bandit
        self.min_success_rate = min_success_rate
        self.min_coverage = min_coverage
        self.window_size = window_size
        
        # Constraint tracking
        self.recent_rewards = []
        self.recent_arms = []
        self.skill_coverage = {}
    
    def check_constraints(self) -> Dict[str, bool]:
        """Check if current performance satisfies constraints."""
        constraints = {
            "success_rate": True,
            "coverage": True
        }
        
        # Check success rate constraint
        if len(self.recent_rewards) >= 10:  # Minimum samples
            success_rate = np.mean(self.recent_rewards)
            constraints["success_rate"] = success_rate >= self.min_success_rate
        
        # Check coverage constraint
        if self.skill_coverage:
            total_skills = len(self.skill_coverage)
            covered_skills = sum(1 for count in self.skill_coverage.values() if count > 0)
            coverage = covered_skills / total_skills if total_skills > 0 else 1.0
            constraints["coverage"] = coverage >= self.min_coverage
        
        return constraints
    
    def select_arm(self, context: np.ndarray, available_skills: List[str]) -> Tuple[str, float]:
        """Select arm considering constraints."""
        constraints = self.check_constraints()
        
        # If constraints are violated, use constraint-aware selection
        if not all(constraints.values()):
            return self._constraint_aware_selection(context, available_skills, constraints)
        
        # Otherwise, use base bandit
        return self.base_bandit.select_arm(context)
    
    def _constraint_aware_selection(
        self,
        context: np.ndarray,
        available_skills: List[str],
        constraints: Dict[str, bool]
    ) -> Tuple[str, float]:
        """Select arm to satisfy constraints."""
        # If success rate is low, prefer easier items
        if not constraints["success_rate"]:
            # Filter arms by difficulty (prefer easier)
            easier_arms = {}
            for arm_id, arm_data in self.base_bandit.arms.items():
                difficulty = arm_data.get("difficulty_b", 0.0)
                user_theta = context[0]  # Assuming first feature is theta
                if difficulty <= user_theta + 0.2:  # Slightly easier
                    easier_arms[arm_id] = arm_data
            
            if easier_arms:
                # Temporarily replace arms
                original_arms = self.base_bandit.arms
                self.base_bandit.arms = easier_arms
                result = self.base_bandit.select_arm(context)
                self.base_bandit.arms = original_arms
                return result
        
        # If coverage is low, prefer underrepresented skills
        if not constraints["coverage"]:
            underrepresented_skills = [
                skill for skill, count in self.skill_coverage.items()
                if count < np.mean(list(self.skill_coverage.values()))
            ]
            
            if underrepresented_skills:
                # Filter arms by underrepresented skills
                coverage_arms = {}
                for arm_id, arm_data in self.base_bandit.arms.items():
                    arm_skills = arm_data.get("skills", []) or arm_data.get("error_tags", [])
                    if any(skill in underrepresented_skills for skill in arm_skills):
                        coverage_arms[arm_id] = arm_data
                
                if coverage_arms:
                    original_arms = self.base_bandit.arms
                    self.base_bandit.arms = coverage_arms
                    result = self.base_bandit.select_arm(context)
                    self.base_bandit.arms = original_arms
                    return result
        
        # Fallback to base bandit
        return self.base_bandit.select_arm(context)
    
    def update(self, arm_id: str, context: np.ndarray, reward: float, skills: List[str]) -> None:
        """Update bandit and constraint tracking."""
        # Update base bandit
        self.base_bandit.update(arm_id, context, reward)
        
        # Update constraint tracking
        self.recent_rewards.append(reward)
        self.recent_arms.append(arm_id)
        
        # Maintain window size
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
            self.recent_arms.pop(0)
        
        # Update skill coverage
        for skill in skills:
            self.skill_coverage[skill] = self.skill_coverage.get(skill, 0) + 1


class BanditService:
    """Service for managing bandit algorithms and decisions."""
    
    def __init__(self):
        self.redis_client = None
        self.bandits = {}  # user_id -> bandit instance
        self.policy_configs = {
            "epsilon_greedy": {"epsilon": 0.1},
            "linucb": {"alpha": settings.BANDIT_ALPHA, "lambda": settings.BANDIT_LAMBDA},
            "lints": {"alpha": settings.BANDIT_ALPHA, "lambda": settings.BANDIT_LAMBDA},
            "constrained_linucb": {
                "alpha": settings.BANDIT_ALPHA,
                "lambda": settings.BANDIT_LAMBDA,
                "min_success_rate": settings.MIN_SUCCESS_RATE,
                "min_coverage": settings.MIN_COVERAGE
            }
        }
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client for bandit state persistence."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(settings.REDIS_URL)
        return self.redis_client
    
    def _create_bandit(self, policy_id: str, config: Dict[str, Any]) -> ContextualBandit:
        """Create bandit instance based on policy."""
        if policy_id == "linucb":
            return LinUCB(alpha=config["alpha"], lambda_reg=config["lambda"])
        elif policy_id == "lints":
            return LinTS(alpha=config["alpha"], lambda_reg=config["lambda"])
        elif policy_id == "constrained_linucb":
            base_bandit = LinUCB(alpha=config["alpha"], lambda_reg=config["lambda"])
            return ConstrainedBandit(
                base_bandit=base_bandit,
                min_success_rate=config["min_success_rate"],
                min_coverage=config["min_coverage"]
            )
        else:
            raise ValueError(f"Unknown policy: {policy_id}")
    
    async def select_questions(
        self,
        session: AsyncSession,
        user_id: str,
        candidates: List[Dict[str, Any]],
        policy_id: str = "linucb",
        slate_size: int = 3,
        target_skills: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Select questions using bandit algorithm.
        
        Args:
            session: Database session
            user_id: User ID
            candidates: List of candidate questions
            policy_id: Bandit policy to use
            slate_size: Number of questions to select
            target_skills: Target skills for selection
            
        Returns:
            Selection results with propensity scores
        """
        # Get user profile
        user = await user_repository.get(session, user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Build user profile dict
        user_profile = {
            "theta_math": user.theta_math or 0.0,
            "theta_en": user.theta_en or 0.0,
            "error_profile_math": user.error_profile_math or {},
            "error_profile_en": user.error_profile_en or {},
            "preferences": user.preferences or {},
            "performance": {}  # Would be populated from recent attempts
        }
        
        # Get or create bandit for user
        bandit_key = f"{user_id}_{policy_id}"
        if bandit_key not in self.bandits:
            config = self.policy_configs.get(policy_id, self.policy_configs["linucb"])
            self.bandits[bandit_key] = self._create_bandit(policy_id, config)
        
        bandit = self.bandits[bandit_key]
        
        # Add candidates as arms
        for candidate in candidates:
            arm_id = candidate["item_id"]
            bandit.add_arm(arm_id, candidate)
        
        # Select arms
        selected_arms = []
        arm_scores = {}
        
        for i in range(min(slate_size, len(candidates))):
            # Build context for current selection
            context = bandit.build_context_vector(
                user_profile, 
                candidates[0],  # Use first candidate as reference
                target_skills
            )
            
            # Select arm
            if isinstance(bandit, ConstrainedBandit):
                available_skills = []
                for candidate in candidates:
                    skills = candidate.get("skills", []) or candidate.get("error_tags", [])
                    available_skills.extend(skills)
                chosen_arm, scores = bandit.select_arm(context, available_skills)
            else:
                chosen_arm, scores = bandit.select_arm(context)
            
            if chosen_arm:
                # Find chosen candidate
                chosen_candidate = next(
                    (c for c in candidates if c["item_id"] == chosen_arm), None
                )
                
                if chosen_candidate:
                    selected_arms.append({
                        "arm_id": chosen_arm,
                        "item_id": chosen_arm,
                        "candidate": chosen_candidate,
                        "propensity": 1.0 / len(candidates),  # Uniform for now
                        "scores": scores.get(chosen_arm, {}) if isinstance(scores, dict) else {}
                    })
                    
                    # Remove selected candidate to avoid duplicates
                    candidates = [c for c in candidates if c["item_id"] != chosen_arm]
                    
                    # Update arm scores
                    if isinstance(scores, dict):
                        arm_scores.update(scores)
        
        return {
            "selected_arms": selected_arms,
            "all_arm_scores": arm_scores,
            "policy_id": policy_id,
            "bandit_version": "1.0",
            "context_features": context.tolist() if 'context' in locals() else []
        }
    
    async def update_bandit(
        self,
        session: AsyncSession,
        user_id: str,
        arm_id: str,
        reward_components: Dict[str, float],
        policy_id: str = "linucb",
        item_skills: Optional[List[str]] = None
    ) -> bool:
        """
        Update bandit based on observed reward.
        
        Args:
            session: Database session
            user_id: User ID
            arm_id: Selected arm ID
            reward_components: Reward components (correct, time, hints, etc.)
            policy_id: Bandit policy used
            item_skills: Skills associated with the item
            
        Returns:
            Success status
        """
        try:
            bandit_key = f"{user_id}_{policy_id}"
            if bandit_key not in self.bandits:
                return False
            
            bandit = self.bandits[bandit_key]
            
            # Calculate overall reward
            reward = self._calculate_overall_reward(reward_components)
            
            # Get user profile for context
            user = await user_repository.get(session, user_id)
            if not user:
                return False
            
            user_profile = {
                "theta_math": user.theta_math or 0.0,
                "theta_en": user.theta_en or 0.0,
                "error_profile_math": user.error_profile_math or {},
                "error_profile_en": user.error_profile_en or {},
                "preferences": user.preferences or {},
                "performance": {}
            }
            
            # Build context (use dummy item metadata for context)
            context = bandit.build_context_vector(user_profile, {}, item_skills)
            
            # Update bandit
            if isinstance(bandit, ConstrainedBandit):
                bandit.update(arm_id, context, reward, item_skills or [])
            else:
                bandit.update(arm_id, context, reward)
            
            return True
            
        except Exception as e:
            print(f"Error updating bandit: {e}")
            return False
    
    def _calculate_overall_reward(self, reward_components: Dict[str, float]) -> float:
        """Calculate overall reward from components."""
        # Weighted combination of reward components
        weights = {
            "correct": 0.5,
            "completion": 0.2,
            "time": 0.2,
            "hints": 0.1
        }
        
        total_reward = 0.0
        for component, value in reward_components.items():
            weight = weights.get(component, 0.0)
            total_reward += weight * value
        
        return max(min(total_reward, 1.0), 0.0)  # Clamp to [0, 1]
    
    async def get_bandit_stats(self, user_id: str, policy_id: str) -> Dict[str, Any]:
        """Get bandit statistics for a user."""
        bandit_key = f"{user_id}_{policy_id}"
        if bandit_key not in self.bandits:
            return {}
        
        bandit = self.bandits[bandit_key]
        
        stats = {
            "policy_id": policy_id,
            "num_arms": len(bandit.arms),
            "feature_dim": bandit.feature_dim,
            "algorithm": type(bandit).__name__
        }
        
        if isinstance(bandit, ConstrainedBandit):
            constraints = bandit.check_constraints()
            stats.update({
                "constraints": constraints,
                "recent_rewards": len(bandit.recent_rewards),
                "skill_coverage": len(bandit.skill_coverage)
            })
        
        return stats


# Create service instance
bandit_service = BanditService()