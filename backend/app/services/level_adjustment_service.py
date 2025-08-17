from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, update
from datetime import datetime, timedelta
import uuid

from app.models.user import User
from app.models.question import Question, Subject
from app.models.student_attempt import StudentAttempt
from app.models.error_pattern import ErrorPattern
from app.schemas.answer import LevelAdjustmentRecommendation, PerformanceMetrics
from app.services.answer_evaluation_service import answer_evaluation_service
from app.core.cache import cache_service


class LevelAdjustmentService:
    """Dynamic level adjustment service based on performance"""
    
    def __init__(self):
        # Performance thresholds for level adjustments
        self.PROMOTION_ACCURACY = 85.0  # Minimum accuracy for level up
        self.DEMOTION_ACCURACY = 60.0   # Maximum accuracy for level down
        self.CONSISTENCY_THRESHOLD = 0.8  # Minimum consistency score
        self.MIN_ATTEMPTS = 10  # Minimum attempts needed for adjustment
        self.EVALUATION_PERIOD = 14  # Days to look back for performance
        
        # Confidence thresholds
        self.HIGH_CONFIDENCE = 0.9
        self.MEDIUM_CONFIDENCE = 0.7
        self.LOW_CONFIDENCE = 0.5
    
    async def evaluate_level_adjustment(
        self, 
        db: AsyncSession, 
        user: User,
        subject: Subject,
        force_evaluation: bool = False
    ) -> Optional[LevelAdjustmentRecommendation]:
        """Evaluate if a user needs level adjustment for a subject"""
        
        current_level = self._get_current_level(user, subject)
        
        # Get performance metrics
        metrics = await answer_evaluation_service.get_user_performance_metrics(
            db, str(user.id), subject, self.EVALUATION_PERIOD
        )
        
        # Check if we have enough data
        if metrics.total_attempts < self.MIN_ATTEMPTS and not force_evaluation:
            return None
        
        # Calculate performance indicators
        performance_indicators = await self._calculate_performance_indicators(
            db, user, subject, metrics
        )
        
        # Determine adjustment recommendation
        recommendation = await self._determine_adjustment(
            current_level, metrics, performance_indicators, subject
        )
        
        return recommendation
    
    async def apply_level_adjustment(
        self, 
        db: AsyncSession, 
        user: User,
        subject: Subject,
        new_level: int,
        reason: str,
        applied_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """Apply level adjustment to user"""
        
        old_level = self._get_current_level(user, subject)
        
        if new_level == old_level:
            return {
                "success": False,
                "message": "New level is the same as current level",
                "old_level": old_level,
                "new_level": new_level
            }
        
        if not (1 <= new_level <= 5):
            return {
                "success": False,
                "message": "Level must be between 1 and 5",
                "old_level": old_level,
                "new_level": new_level
            }
        
        # Update user level
        if subject == Subject.MATH:
            user.current_math_level = new_level
        else:
            user.current_english_level = new_level
        
        # Save to database
        await db.commit()
        await db.refresh(user)
        
        # Log the adjustment
        adjustment_log = await self._log_level_adjustment(
            db, user, subject, old_level, new_level, reason, applied_by
        )
        
        # Invalidate caches
        await self._invalidate_level_caches(str(user.id), subject)
        
        return {
            "success": True,
            "message": f"Level adjusted from {old_level} to {new_level}",
            "old_level": old_level,
            "new_level": new_level,
            "subject": subject.value,
            "reason": reason,
            "adjustment_id": str(adjustment_log["id"]) if adjustment_log else None,
            "applied_at": datetime.utcnow().isoformat()
        }
    
    async def get_level_history(
        self, 
        db: AsyncSession, 
        user_id: str,
        subject: Optional[Subject] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get level adjustment history for a user"""
        
        # This would require a level_adjustments table in a real implementation
        # For now, return placeholder data
        return [
            {
                "id": "placeholder",
                "user_id": user_id,
                "subject": subject.value if subject else "math",
                "old_level": 2,
                "new_level": 3,
                "reason": "Performance improvement",
                "applied_by": "system",
                "applied_at": datetime.utcnow().isoformat(),
                "confidence": 0.85
            }
        ]
    
    async def batch_evaluate_adjustments(
        self, 
        db: AsyncSession, 
        subject: Subject,
        min_attempts: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate level adjustments for all users in a subject"""
        
        min_attempts = min_attempts or self.MIN_ATTEMPTS
        
        # Get users with sufficient attempts in the subject
        cutoff_date = datetime.utcnow() - timedelta(days=self.EVALUATION_PERIOD)
        
        # Query to find users with enough attempts
        query = select(
            StudentAttempt.user_id,
            func.count(StudentAttempt.id).label('attempt_count')
        ).join(Question).where(
            and_(
                Question.subject == subject,
                StudentAttempt.attempt_date >= cutoff_date
            )
        ).group_by(StudentAttempt.user_id).having(
            func.count(StudentAttempt.id) >= min_attempts
        )
        
        result = await db.execute(query)
        user_attempts = result.fetchall()
        
        recommendations = []
        
        for user_id, attempt_count in user_attempts:
            # Get user
            user_result = await db.execute(select(User).where(User.id == user_id))
            user = user_result.scalar_one_or_none()
            
            if user:
                # Evaluate adjustment
                recommendation = await self.evaluate_level_adjustment(db, user, subject)
                
                if recommendation:
                    recommendations.append({
                        "user_id": str(user.id),
                        "username": user.username,
                        "current_level": recommendation.current_level,
                        "recommended_level": recommendation.recommended_level,
                        "reason": recommendation.reason,
                        "confidence": recommendation.confidence,
                        "attempt_count": attempt_count,
                        "recommendation": recommendation
                    })
        
        return recommendations
    
    async def get_level_statistics(
        self, 
        db: AsyncSession, 
        subject: Subject
    ) -> Dict[str, Any]:
        """Get level distribution statistics"""
        
        level_field = User.current_math_level if subject == Subject.MATH else User.current_english_level
        
        # Get level distribution
        level_distribution = {}
        for level in range(1, 6):
            result = await db.execute(
                select(func.count(User.id)).where(level_field == level)
            )
            level_distribution[str(level)] = result.scalar() or 0
        
        # Get total users
        total_result = await db.execute(select(func.count(User.id)))
        total_users = total_result.scalar() or 0
        
        # Calculate percentages
        level_percentages = {}
        for level, count in level_distribution.items():
            level_percentages[level] = (count / total_users * 100) if total_users > 0 else 0
        
        # Get average level
        avg_result = await db.execute(select(func.avg(level_field)))
        average_level = float(avg_result.scalar() or 0)
        
        return {
            "subject": subject.value,
            "total_users": total_users,
            "level_distribution": level_distribution,
            "level_percentages": level_percentages,
            "average_level": average_level,
            "most_common_level": max(level_distribution.items(), key=lambda x: x[1])[0] if level_distribution else "1"
        }
    
    async def predict_level_progression(
        self, 
        db: AsyncSession, 
        user: User,
        subject: Subject
    ) -> Dict[str, Any]:
        """Predict when user might be ready for next level"""
        
        current_level = self._get_current_level(user, subject)
        
        if current_level >= 5:
            return {
                "current_level": current_level,
                "next_level": None,
                "prediction": "User is at maximum level",
                "estimated_time": None
            }
        
        # Get recent performance trend
        metrics = await answer_evaluation_service.get_user_performance_metrics(
            db, str(user.id), subject, days=30
        )
        
        # Simple prediction based on current performance
        if metrics.accuracy_rate >= self.PROMOTION_ACCURACY:
            estimated_days = 0  # Ready now
        elif metrics.accuracy_rate >= 75:
            estimated_days = 7  # About a week
        elif metrics.accuracy_rate >= 65:
            estimated_days = 14  # About two weeks
        else:
            estimated_days = 30  # About a month
        
        # Adjust based on improvement trend
        if metrics.improvement_trend == "improving":
            estimated_days = max(0, estimated_days - 7)
        elif metrics.improvement_trend == "declining":
            estimated_days += 14
        
        return {
            "current_level": current_level,
            "next_level": current_level + 1,
            "current_accuracy": metrics.accuracy_rate,
            "required_accuracy": self.PROMOTION_ACCURACY,
            "estimated_days_to_promotion": estimated_days,
            "improvement_trend": metrics.improvement_trend,
            "prediction_confidence": self._calculate_prediction_confidence(metrics)
        }
    
    # Private helper methods
    def _get_current_level(self, user: User, subject: Subject) -> int:
        """Get current level for subject"""
        return user.current_math_level if subject == Subject.MATH else user.current_english_level
    
    async def _calculate_performance_indicators(
        self, 
        db: AsyncSession, 
        user: User, 
        subject: Subject,
        metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Calculate detailed performance indicators"""
        
        # Consistency score (how stable is the performance)
        consistency_score = await self._calculate_consistency_score(db, user, subject)
        
        # Recent performance trend
        recent_accuracy = metrics.recent_accuracy or metrics.accuracy_rate
        
        # Error pattern severity
        error_severity = await self._calculate_error_severity(db, user, subject)
        
        # Time efficiency
        time_efficiency = await self._calculate_time_efficiency(db, user, subject)
        
        return {
            "consistency_score": consistency_score,
            "recent_accuracy": recent_accuracy,
            "error_severity": error_severity,
            "time_efficiency": time_efficiency,
            "improvement_trend": metrics.improvement_trend,
            "current_streak": metrics.current_streak
        }
    
    async def _determine_adjustment(
        self, 
        current_level: int, 
        metrics: PerformanceMetrics,
        indicators: Dict[str, Any],
        subject: Subject
    ) -> Optional[LevelAdjustmentRecommendation]:
        """Determine if level adjustment is needed"""
        
        accuracy = metrics.accuracy_rate
        consistency = indicators["consistency_score"]
        
        # Check for promotion
        if (accuracy >= self.PROMOTION_ACCURACY and 
            consistency >= self.CONSISTENCY_THRESHOLD and 
            current_level < 5 and
            indicators["current_streak"] >= 3):
            
            confidence = min(
                self.HIGH_CONFIDENCE,
                (accuracy / 100) * consistency * (indicators["current_streak"] / 10)
            )
            
            return LevelAdjustmentRecommendation(
                current_level=current_level,
                recommended_level=current_level + 1,
                reason=f"Excellent performance ({accuracy:.1f}% accuracy) with consistent results",
                confidence=confidence,
                supporting_evidence=[
                    f"Accuracy rate: {accuracy:.1f}%",
                    f"Consistency score: {consistency:.2f}",
                    f"Current streak: {indicators['current_streak']}",
                    f"Total attempts: {metrics.total_attempts}"
                ],
                accuracy_threshold=self.PROMOTION_ACCURACY,
                consistency_threshold=self.CONSISTENCY_THRESHOLD
            )
        
        # Check for demotion
        elif (accuracy <= self.DEMOTION_ACCURACY and 
              current_level > 1 and
              indicators["error_severity"] > 0.7):
            
            confidence = min(
                self.MEDIUM_CONFIDENCE,
                (1 - accuracy / 100) * indicators["error_severity"]
            )
            
            return LevelAdjustmentRecommendation(
                current_level=current_level,
                recommended_level=current_level - 1,
                reason=f"Performance below threshold ({accuracy:.1f}% accuracy) with high error rate",
                confidence=confidence,
                supporting_evidence=[
                    f"Accuracy rate: {accuracy:.1f}%",
                    f"Below threshold of {self.DEMOTION_ACCURACY}%",
                    f"Error severity: {indicators['error_severity']:.2f}",
                    f"Total attempts: {metrics.total_attempts}"
                ],
                accuracy_threshold=self.DEMOTION_ACCURACY,
                consistency_threshold=self.CONSISTENCY_THRESHOLD
            )
        
        return None
    
    async def _calculate_consistency_score(
        self, 
        db: AsyncSession, 
        user: User, 
        subject: Subject
    ) -> float:
        """Calculate performance consistency score"""
        
        # Get recent attempts
        cutoff_date = datetime.utcnow() - timedelta(days=self.EVALUATION_PERIOD)
        
        query = select(StudentAttempt.is_correct).join(Question).where(
            and_(
                StudentAttempt.user_id == user.id,
                Question.subject == subject,
                StudentAttempt.attempt_date >= cutoff_date
            )
        ).order_by(StudentAttempt.attempt_date)
        
        result = await db.execute(query)
        attempts = [row[0] for row in result.fetchall()]
        
        if len(attempts) < 5:
            return 0.5  # Default consistency for insufficient data
        
        # Calculate rolling accuracy over windows
        window_size = max(5, len(attempts) // 4)
        accuracies = []
        
        for i in range(len(attempts) - window_size + 1):
            window = attempts[i:i + window_size]
            accuracy = sum(window) / len(window)
            accuracies.append(accuracy)
        
        if not accuracies:
            return 0.5
        
        # Calculate variance (lower variance = higher consistency)
        mean_accuracy = sum(accuracies) / len(accuracies)
        variance = sum((acc - mean_accuracy) ** 2 for acc in accuracies) / len(accuracies)
        
        # Convert variance to consistency score (0-1, higher is better)
        consistency = max(0, 1 - (variance * 4))  # Scale variance
        
        return min(1.0, consistency)
    
    async def _calculate_error_severity(
        self, 
        db: AsyncSession, 
        user: User, 
        subject: Subject
    ) -> float:
        """Calculate error pattern severity"""
        
        # Get error patterns
        result = await db.execute(
            select(ErrorPattern.error_count).where(
                and_(
                    ErrorPattern.user_id == user.id,
                    ErrorPattern.subject == subject
                )
            )
        )
        
        error_counts = [row[0] for row in result.fetchall()]
        
        if not error_counts:
            return 0.0
        
        # Calculate severity based on frequency and distribution
        total_errors = sum(error_counts)
        max_errors = max(error_counts)
        
        # Normalize to 0-1 scale
        severity = min(1.0, (total_errors / 50) * (max_errors / 20))
        
        return severity
    
    async def _calculate_time_efficiency(
        self, 
        db: AsyncSession, 
        user: User, 
        subject: Subject
    ) -> float:
        """Calculate time efficiency score"""
        
        # Get recent attempts with time data
        cutoff_date = datetime.utcnow() - timedelta(days=self.EVALUATION_PERIOD)
        
        query = select(StudentAttempt.time_spent, Question.difficulty_level).join(Question).where(
            and_(
                StudentAttempt.user_id == user.id,
                Question.subject == subject,
                StudentAttempt.attempt_date >= cutoff_date,
                StudentAttempt.time_spent.isnot(None)
            )
        )
        
        result = await db.execute(query)
        time_data = result.fetchall()
        
        if not time_data:
            return 0.5  # Default efficiency
        
        # Calculate average time per difficulty level
        difficulty_times = {}
        for time_spent, difficulty in time_data:
            if difficulty not in difficulty_times:
                difficulty_times[difficulty] = []
            difficulty_times[difficulty].append(time_spent)
        
        # Expected times per difficulty (in seconds)
        expected_times = {1: 30, 2: 45, 3: 60, 4: 90, 5: 120}
        
        efficiency_scores = []
        for difficulty, times in difficulty_times.items():
            avg_time = sum(times) / len(times)
            expected_time = expected_times.get(difficulty, 60)
            
            # Efficiency = expected / actual (capped at 1.0)
            efficiency = min(1.0, expected_time / avg_time)
            efficiency_scores.append(efficiency)
        
        return sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.5
    
    def _calculate_prediction_confidence(self, metrics: PerformanceMetrics) -> float:
        """Calculate confidence in level progression prediction"""
        
        if metrics.total_attempts < 5:
            return 0.3
        elif metrics.total_attempts < 10:
            return 0.5
        elif metrics.total_attempts < 20:
            return 0.7
        else:
            return 0.9
    
    async def _log_level_adjustment(
        self, 
        db: AsyncSession, 
        user: User,
        subject: Subject,
        old_level: int,
        new_level: int,
        reason: str,
        applied_by: Optional[str]
    ) -> Dict[str, Any]:
        """Log level adjustment (placeholder - would need level_adjustments table)"""
        
        # In a real implementation, this would save to a level_adjustments table
        log_entry = {
            "id": uuid.uuid4(),
            "user_id": str(user.id),
            "subject": subject.value,
            "old_level": old_level,
            "new_level": new_level,
            "reason": reason,
            "applied_by": applied_by or "system",
            "applied_at": datetime.utcnow().isoformat()
        }
        
        return log_entry
    
    async def _invalidate_level_caches(self, user_id: str, subject: Subject):
        """Invalidate level-related caches"""
        
        cache_keys = [
            f"user_performance:{user_id}:{subject.value}",
            f"level_recommendation:{user_id}:{subject.value}",
            f"question_recommendations:{user_id}:{subject.value}"
        ]
        
        for key in cache_keys:
            await cache_service.delete(key)


# Global level adjustment service instance
level_adjustment_service = LevelAdjustmentService()