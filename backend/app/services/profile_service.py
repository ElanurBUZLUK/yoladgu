"""
Profile service with caching and comprehensive user management.
"""

import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from app.core.config import settings
from app.db.repositories.user import user_repository
from app.db.repositories.attempt import attempt_repository
from app.services.irt_service import irt_service, error_profile_service


class ProfileService:
    """Profile service with Redis caching and IRT integration."""
    
    def __init__(self):
        self.redis_client = None
        self.cache_ttl = settings.CACHE_TTL_SECONDS
        self.cache_prefix = "profile:"
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client (lazy initialization)."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(settings.REDIS_URL)
        return self.redis_client
    
    def _get_cache_key(self, user_id: str) -> str:
        """Generate cache key for user profile."""
        return f"{self.cache_prefix}{user_id}"
    
    async def _get_from_cache(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get profile from cache."""
        try:
            redis_client = await self._get_redis_client()
            cached_data = await redis_client.get(self._get_cache_key(user_id))
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None
    
    async def _set_cache(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        """Set profile in cache."""
        try:
            redis_client = await self._get_redis_client()
            await redis_client.setex(
                self._get_cache_key(user_id),
                self.cache_ttl,
                json.dumps(profile_data, default=str)
            )
        except Exception as e:
            print(f"Cache set error: {e}")
    
    async def _invalidate_cache(self, user_id: str) -> None:
        """Invalidate user profile cache."""
        try:
            redis_client = await self._get_redis_client()
            await redis_client.delete(self._get_cache_key(user_id))
        except Exception as e:
            print(f"Cache invalidation error: {e}")
    
    async def get_user_profile(
        self,
        session: AsyncSession,
        user_id: str,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive user profile with caching.
        
        Args:
            session: Database session
            user_id: User ID
            use_cache: Whether to use cache
            
        Returns:
            User profile dictionary
        """
        # Try cache first
        if use_cache:
            cached_profile = await self._get_from_cache(user_id)
            if cached_profile:
                return cached_profile
        
        # Get from database
        user = await user_repository.get(session, user_id)
        if not user:
            return None
        
        # Get recent performance stats
        math_stats = await attempt_repository.get_user_performance_stats(
            session, user_id, "math", days=30
        )
        english_stats = await attempt_repository.get_user_performance_stats(
            session, user_id, "en", days=30
        )
        
        # Get weak skills
        weak_math_skills = await error_profile_service.get_weak_skills(
            session, user_id, "math"
        )
        weak_english_skills = await error_profile_service.get_weak_skills(
            session, user_id, "en"
        )
        
        # Build comprehensive profile
        profile = {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "grade": user.grade,
            "lang": user.lang,
            "role": user.role,
            "consent_flag": user.consent_flag,
            
            # Learning state
            "theta_math": user.theta_math,
            "theta_en": user.theta_en,
            "error_profile_math": user.error_profile_math or {},
            "error_profile_en": user.error_profile_en or {},
            
            # Performance stats
            "performance": {
                "math": math_stats,
                "english": english_stats
            },
            
            # Weak skills
            "weak_skills": {
                "math": weak_math_skills,
                "english": weak_english_skills
            },
            
            # User preferences and segments
            "segments": user.segments or [],
            "preferences": user.preferences or {},
            
            # Metadata
            "is_active": user.is_active,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "profile_generated_at": datetime.utcnow()
        }
        
        # Cache the profile
        if use_cache:
            await self._set_cache(user_id, profile)
        
        return profile
    
    async def update_profile_after_attempt(
        self,
        session: AsyncSession,
        user_id: str,
        item_id: str,
        item_type: str,
        answer: str,
        correct: bool,
        time_ms: Optional[int] = None,
        hints_used: int = 0,
        context: Optional[Dict[str, Any]] = None,
        item_skills: Optional[List[str]] = None,
        item_a: float = 1.0,
        item_b: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """
        Update user profile after an attempt (IRT + error profile).
        
        Args:
            session: Database session
            user_id: User ID
            item_id: Item ID
            item_type: "math" or "en"
            answer: User's answer
            correct: Whether answer was correct
            time_ms: Response time
            hints_used: Number of hints used
            context: Additional context
            item_skills: Skills/tags for the item
            item_a: Item discrimination parameter
            item_b: Item difficulty parameter
            
        Returns:
            Updated profile data
        """
        # Record the attempt
        attempt = await attempt_repository.create_attempt(
            session=session,
            user_id=user_id,
            item_id=item_id,
            item_type=item_type,
            answer=answer,
            correct=correct,
            time_ms=time_ms,
            hints_used=hints_used,
            context=context
        )
        
        # Update theta using IRT
        new_theta = await irt_service.update_user_theta(
            session=session,
            user_id=user_id,
            item_type=item_type,
            response=1 if correct else 0,
            item_a=item_a,
            item_b=item_b,
            use_online=True
        )
        
        # Update error profile if incorrect and skills provided
        updated_error_profile = None
        if not correct and item_skills:
            updated_error_profile = await error_profile_service.update_error_profile(
                session=session,
                user_id=user_id,
                item_type=item_type,
                skill_tags=item_skills,
                is_correct=correct
            )
        
        # Invalidate cache
        await self._invalidate_cache(user_id)
        
        # Return update summary
        return {
            "attempt_id": attempt.id,
            "updated_theta": {item_type: new_theta} if new_theta else None,
            "updated_error_profile": updated_error_profile,
            "reward_components": self._calculate_reward_components(
                correct, time_ms, hints_used
            )
        }
    
    def _calculate_reward_components(
        self,
        correct: bool,
        time_ms: Optional[int],
        hints_used: int
    ) -> Dict[str, float]:
        """Calculate reward components for bandit algorithms."""
        components = {
            "correct": 1.0 if correct else 0.0,
            "completion": 1.0,  # Always 1.0 if attempt was made
        }
        
        # Time-based reward (faster = better, up to reasonable limit)
        if time_ms is not None:
            # Normalize time (assume 30 seconds is optimal, 5 minutes is max)
            optimal_time = 30000  # 30 seconds
            max_time = 300000     # 5 minutes
            
            if time_ms <= optimal_time:
                time_reward = 1.0
            elif time_ms >= max_time:
                time_reward = 0.1
            else:
                # Linear decay from optimal to max time
                time_reward = 1.0 - 0.9 * (time_ms - optimal_time) / (max_time - optimal_time)
            
            components["time"] = max(time_reward, 0.1)
        else:
            components["time"] = 0.5  # Default if no time data
        
        # Hint penalty
        hint_penalty = min(hints_used * 0.1, 0.5)  # Max 50% penalty
        components["hints"] = max(1.0 - hint_penalty, 0.1)
        
        return components
    
    async def get_learning_recommendations(
        self,
        session: AsyncSession,
        user_id: str,
        item_type: str
    ) -> Dict[str, Any]:
        """
        Get learning recommendations based on user profile.
        
        Args:
            session: Database session
            user_id: User ID
            item_type: "math" or "en"
            
        Returns:
            Learning recommendations
        """
        profile = await self.get_user_profile(session, user_id)
        if not profile:
            return {}
        
        theta = profile.get(f"theta_{item_type}", 0.0)
        error_profile = profile.get(f"error_profile_{item_type}", {})
        
        # Get optimal difficulty range
        min_diff, max_diff = irt_service.get_optimal_difficulty_range(theta)
        
        # Get skill priorities
        skill_priorities = error_profile_service.calculate_skill_priority(error_profile)
        
        # Get weak skills
        weak_skills = await error_profile_service.get_weak_skills(
            session, user_id, item_type
        )
        
        return {
            "optimal_difficulty_range": {
                "min": min_diff,
                "max": max_diff,
                "target_success_rate": 0.7
            },
            "skill_priorities": skill_priorities,
            "weak_skills": weak_skills,
            "recommended_focus": weak_skills[:3] if weak_skills else [],
            "current_theta": theta,
            "confidence_level": self._calculate_confidence_level(
                profile["performance"].get(item_type, {})
            )
        }
    
    def _calculate_confidence_level(self, performance_stats: Dict[str, Any]) -> str:
        """Calculate confidence level based on performance stats."""
        total_attempts = performance_stats.get("total_attempts", 0)
        success_rate = performance_stats.get("success_rate", 0.0)
        
        if total_attempts < 5:
            return "low"  # Not enough data
        elif total_attempts < 20:
            return "medium"
        else:
            if success_rate >= 0.8:
                return "high"
            elif success_rate >= 0.6:
                return "medium"
            else:
                return "low"
    
    async def get_peer_comparison(
        self,
        session: AsyncSession,
        user_id: str,
        item_type: str
    ) -> Dict[str, Any]:
        """
        Get peer comparison data for user.
        
        Args:
            session: Database session
            user_id: User ID
            item_type: "math" or "en"
            
        Returns:
            Peer comparison data
        """
        user = await user_repository.get(session, user_id)
        if not user:
            return {}
        
        # Get similar users
        similar_users = await user_repository.get_similar_users(
            session, user_id, item_type, limit=20
        )
        
        if not similar_users:
            return {"message": "No similar users found"}
        
        # Calculate peer statistics
        user_theta = getattr(user, f"theta_{item_type}", 0.0)
        peer_thetas = [getattr(u, f"theta_{item_type}", 0.0) for u in similar_users]
        
        if peer_thetas:
            avg_peer_theta = sum(peer_thetas) / len(peer_thetas)
            percentile = sum(1 for theta in peer_thetas if theta < user_theta) / len(peer_thetas)
        else:
            avg_peer_theta = 0.0
            percentile = 0.5
        
        return {
            "user_theta": user_theta,
            "peer_average": avg_peer_theta,
            "percentile": percentile,
            "peer_count": len(similar_users),
            "relative_performance": "above_average" if user_theta > avg_peer_theta else "below_average"
        }
    
    async def bulk_update_profiles(
        self,
        session: AsyncSession,
        user_ids: List[str]
    ) -> Dict[str, bool]:
        """
        Bulk update profiles (useful for batch processing).
        
        Args:
            session: Database session
            user_ids: List of user IDs to update
            
        Returns:
            Success status for each user
        """
        results = {}
        
        for user_id in user_ids:
            try:
                # Invalidate cache and force refresh
                await self._invalidate_cache(user_id)
                profile = await self.get_user_profile(session, user_id, use_cache=False)
                results[user_id] = profile is not None
            except Exception as e:
                print(f"Error updating profile for user {user_id}: {e}")
                results[user_id] = False
        
        return results


# Create service instance
profile_service = ProfileService()