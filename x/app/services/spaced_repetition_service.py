from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, update
from datetime import datetime, timedelta
import uuid
import math

from app.models.user import User
from app.models.question import Question, Subject
from app.models.student_attempt import StudentAttempt
from app.models.spaced_repetition import SpacedRepetition
from app.core.cache import cache_service


class SpacedRepetitionService:
    """Spaced Repetition Service using SM-2 Algorithm"""
    
    def __init__(self):
        # SM-2 Algorithm Constants
        self.INITIAL_EASE_FACTOR = 2.5
        self.MIN_EASE_FACTOR = 1.3
        self.MAX_EASE_FACTOR = 3.5  # Maximum ease factor to prevent excessive intervals
        self.EASE_FACTOR_BONUS = 0.1
        self.EASE_FACTOR_PENALTY = 0.2
        self.MINIMUM_INTERVAL = 1  # 1 day
        self.INITIAL_INTERVAL = 1  # 1 day for first review
        self.SECOND_INTERVAL = 6   # 6 days for second review
        
        # Quality thresholds (0-5 scale)
        self.QUALITY_THRESHOLD = 3  # Below this = failed review
        self.PERFECT_QUALITY = 5
        self.GOOD_QUALITY = 4
        
        # Review scheduling
        self.MAX_DAILY_REVIEWS = 50
        self.REVIEW_BUFFER_HOURS = 2  # Allow reviews 2 hours before scheduled time
    
    async def schedule_question_review(
        self, 
        db: AsyncSession, 
        user_id: str,
        question_id: str,
        quality: int,
        response_time: Optional[int] = None
    ) -> Dict[str, Any]:
        """Schedule next review for a question based on SM-2 algorithm"""
        
        # Get existing spaced repetition record
        result = await db.execute(
            select(SpacedRepetition).where(
                and_(
                    SpacedRepetition.user_id == user_id,
                    SpacedRepetition.question_id == question_id
                )
            )
        )
        
        spaced_rep = result.scalar_one_or_none()
        
        if not spaced_rep:
            # Create new spaced repetition record
            spaced_rep = SpacedRepetition(
                user_id=user_id,
                question_id=question_id,
                ease_factor=self.INITIAL_EASE_FACTOR,
                review_count=0,
                next_review_at=datetime.utcnow() + timedelta(days=self.INITIAL_INTERVAL),
                last_reviewed=datetime.utcnow()
            )
            db.add(spaced_rep)
        
        # Calculate new values using SM-2 algorithm
        new_ease_factor, new_interval = self._calculate_sm2_values(
            spaced_rep.ease_factor,
            spaced_rep.review_count,
            quality
        )
        
        # Update spaced repetition record
        spaced_rep.ease_factor = new_ease_factor
        spaced_rep.review_count += 1
        spaced_rep.last_reviewed = datetime.utcnow()
        spaced_rep.next_review_at = datetime.utcnow() + timedelta(days=new_interval)
        
        await db.commit()
        await db.refresh(spaced_rep)
        
        # Invalidate cache
        await self._invalidate_review_caches(user_id)
        
        return {
            "user_id": user_id,
            "question_id": question_id,
            "quality": quality,
            "ease_factor": new_ease_factor,
            "interval_days": new_interval,
            "review_count": spaced_rep.review_count,
            "next_review_at": spaced_rep.next_review_at.isoformat(),
            "last_reviewed": spaced_rep.last_reviewed.isoformat()
        }
    
    async def get_due_reviews(
        self, 
        db: AsyncSession, 
        user_id: str,
        subject: Optional[Subject] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get questions due for review"""
        
        # Calculate review time with buffer
        review_time = datetime.utcnow() + timedelta(hours=self.REVIEW_BUFFER_HOURS)
        
        # Build query
        query = select(SpacedRepetition, Question).join(Question).where(
            and_(
                SpacedRepetition.user_id == user_id,
                SpacedRepetition.next_review_at <= review_time
            )
        )
        
        if subject:
            query = query.where(Question.subject == subject)
        
        query = query.order_by(SpacedRepetition.next_review_at).limit(limit)
        
        result = await db.execute(query)
        reviews_with_questions = result.fetchall()
        
        due_reviews = []
        for spaced_rep, question in reviews_with_questions:
            # Calculate priority based on overdue time
            overdue_hours = (datetime.utcnow() - spaced_rep.next_review_at).total_seconds() / 3600
            priority = max(1, min(5, int(overdue_hours / 24) + 1))  # 1-5 priority scale
            
            due_reviews.append({
                "question_id": str(question.id),
                "question_content": question.content,
                "question_type": question.question_type.value,
                "subject": question.subject.value,
                "difficulty_level": question.difficulty_level,
                "topic_category": question.topic_category,
                "review_count": spaced_rep.review_count,
                "ease_factor": spaced_rep.ease_factor,
                "next_review_at": spaced_rep.next_review_at.isoformat(),
                "overdue_hours": max(0, overdue_hours),
                "priority": priority,
                "last_reviewed": spaced_rep.last_reviewed.isoformat() if spaced_rep.last_reviewed else None
            })
        
        return due_reviews
    
    async def get_review_statistics(
        self, 
        db: AsyncSession, 
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get review statistics for a user"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get all spaced repetition records for user
        result = await db.execute(
            select(SpacedRepetition).where(
                and_(
                    SpacedRepetition.user_id == user_id,
                    SpacedRepetition.last_reviewed >= cutoff_date
                )
            )
        )
        
        reviews = result.scalars().all()
        
        if not reviews:
            return {
                "total_reviews": 0,
                "average_ease_factor": self.INITIAL_EASE_FACTOR,
                "due_today": 0,
                "overdue": 0,
                "upcoming_7_days": 0,
                "retention_rate": 0.0,
                "average_interval": 0.0
            }
        
        # Calculate statistics
        total_reviews = len(reviews)
        average_ease_factor = sum(r.ease_factor for r in reviews) / total_reviews
        
        # Count due and overdue reviews
        now = datetime.utcnow()
        today_end = now.replace(hour=23, minute=59, second=59)
        week_end = now + timedelta(days=7)
        
        due_today = len([r for r in reviews if r.next_review_at <= today_end])
        overdue = len([r for r in reviews if r.next_review_at < now])
        upcoming_7_days = len([r for r in reviews if now < r.next_review_at <= week_end])
        
        # Calculate retention rate (ease factor >= initial = good retention)
        good_retention = len([r for r in reviews if r.ease_factor >= self.INITIAL_EASE_FACTOR])
        retention_rate = (good_retention / total_reviews) * 100 if total_reviews > 0 else 0
        
        # Calculate average interval
        intervals = []
        for review in reviews:
            if review.last_reviewed:
                interval = (review.next_review_at - review.last_reviewed).days
                intervals.append(interval)
        
        average_interval = sum(intervals) / len(intervals) if intervals else 0
        
        return {
            "total_reviews": total_reviews,
            "average_ease_factor": round(average_ease_factor, 2),
            "due_today": due_today,
            "overdue": overdue,
            "upcoming_7_days": upcoming_7_days,
            "retention_rate": round(retention_rate, 1),
            "average_interval": round(average_interval, 1),
            "analysis_period_days": days
        }
    
    async def get_review_calendar(
        self, 
        db: AsyncSession, 
        user_id: str,
        days_ahead: int = 30
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get review calendar for upcoming days"""
        
        start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=days_ahead)
        
        # Get scheduled reviews
        result = await db.execute(
            select(SpacedRepetition, Question).join(Question).where(
                and_(
                    SpacedRepetition.user_id == user_id,
                    SpacedRepetition.next_review_at >= start_date,
                    SpacedRepetition.next_review_at <= end_date
                )
            ).order_by(SpacedRepetition.next_review_at)
        )
        
        reviews_with_questions = result.fetchall()
        
        # Group by date
        calendar = {}
        
        for spaced_rep, question in reviews_with_questions:
            review_date = spaced_rep.next_review_at.date().isoformat()
            
            if review_date not in calendar:
                calendar[review_date] = []
            
            calendar[review_date].append({
                "question_id": str(question.id),
                "question_content": question.content[:100] + "..." if len(question.content) > 100 else question.content,
                "subject": question.subject.value,
                "difficulty_level": question.difficulty_level,
                "review_count": spaced_rep.review_count,
                "ease_factor": spaced_rep.ease_factor,
                "scheduled_time": spaced_rep.next_review_at.isoformat()
            })
        
        return calendar
    
    async def reset_question_progress(
        self, 
        db: AsyncSession, 
        user_id: str,
        question_id: str
    ) -> Dict[str, Any]:
        """Reset spaced repetition progress for a question"""
        
        result = await db.execute(
            select(SpacedRepetition).where(
                and_(
                    SpacedRepetition.user_id == user_id,
                    SpacedRepetition.question_id == question_id
                )
            )
        )
        
        spaced_rep = result.scalar_one_or_none()
        
        if not spaced_rep:
            return {
                "success": False,
                "message": "No spaced repetition record found for this question"
            }
        
        # Reset to initial values
        spaced_rep.ease_factor = self.INITIAL_EASE_FACTOR
        spaced_rep.review_count = 0
        spaced_rep.next_review_at = datetime.utcnow() + timedelta(days=self.INITIAL_INTERVAL)
        spaced_rep.last_reviewed = datetime.utcnow()
        
        await db.commit()
        
        # Invalidate cache
        await self._invalidate_review_caches(user_id)
        
        return {
            "success": True,
            "message": "Question progress reset successfully",
            "question_id": question_id,
            "new_ease_factor": spaced_rep.ease_factor,
            "new_review_count": spaced_rep.review_count,
            "next_review_at": spaced_rep.next_review_at.isoformat()
        }
    
    async def bulk_schedule_reviews(
        self, 
        db: AsyncSession, 
        user_id: str,
        question_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Bulk schedule reviews for multiple questions"""
        
        scheduled_count = 0
        failed_count = 0
        results = []
        
        for result in question_results:
            try:
                question_id = result["question_id"]
                quality = result["quality"]
                response_time = result.get("response_time")
                
                schedule_result = await self.schedule_question_review(
                    db, user_id, question_id, quality, response_time
                )
                
                results.append(schedule_result)
                scheduled_count += 1
                
            except Exception as e:
                failed_count += 1
                results.append({
                    "question_id": result.get("question_id", "unknown"),
                    "error": str(e),
                    "success": False
                })
        
        return {
            "total_processed": len(question_results),
            "scheduled_count": scheduled_count,
            "failed_count": failed_count,
            "results": results
        }
    
    async def get_learning_progress(
        self, 
        db: AsyncSession, 
        user_id: str,
        subject: Optional[Subject] = None
    ) -> Dict[str, Any]:
        """Get learning progress based on spaced repetition data"""
        
        # Build query
        query = select(SpacedRepetition, Question).join(Question).where(
            SpacedRepetition.user_id == user_id
        )
        
        if subject:
            query = query.where(Question.subject == subject)
        
        result = await db.execute(query)
        reviews_with_questions = result.fetchall()
        
        if not reviews_with_questions:
            return {
                "total_questions": 0,
                "mastery_levels": {"learning": 0, "reviewing": 0, "mastered": 0},
                "average_ease_factor": self.INITIAL_EASE_FACTOR,
                "progress_percentage": 0.0
            }
        
        # Categorize questions by mastery level
        learning = 0      # review_count < 3
        reviewing = 0     # 3 <= review_count < 8
        mastered = 0      # review_count >= 8 and ease_factor >= 2.5
        
        ease_factors = []
        
        for spaced_rep, question in reviews_with_questions:
            ease_factors.append(spaced_rep.ease_factor)
            
            if spaced_rep.review_count < 3:
                learning += 1
            elif spaced_rep.review_count < 8:
                reviewing += 1
            elif spaced_rep.ease_factor >= self.INITIAL_EASE_FACTOR:
                mastered += 1
            else:
                reviewing += 1  # Still reviewing if ease factor is low
        
        total_questions = len(reviews_with_questions)
        average_ease_factor = sum(ease_factors) / len(ease_factors)
        
        # Calculate progress percentage (weighted: mastered=100%, reviewing=50%, learning=25%)
        progress_score = (mastered * 100 + reviewing * 50 + learning * 25) / total_questions
        progress_percentage = min(100.0, progress_score)
        
        return {
            "total_questions": total_questions,
            "mastery_levels": {
                "learning": learning,
                "reviewing": reviewing,
                "mastered": mastered
            },
            "average_ease_factor": round(average_ease_factor, 2),
            "progress_percentage": round(progress_percentage, 1),
            "subject": subject.value if subject else "all"
        }
    
    # Private helper methods
    def _calculate_sm2_values(self, ease_factor: float, review_count: int, quality: int) -> Tuple[float, int]:
        """Calculate new ease factor and interval using SM-2 algorithm"""
        
        # Ensure quality is in valid range (0-5)
        quality = max(0, min(5, quality))
        
        # Calculate new ease factor
        new_ease_factor = ease_factor
        
        if quality >= self.QUALITY_THRESHOLD:
            # Successful review - increase ease factor slightly
            if quality == self.PERFECT_QUALITY:
                new_ease_factor += self.EASE_FACTOR_BONUS
            elif quality == self.GOOD_QUALITY:
                new_ease_factor += self.EASE_FACTOR_BONUS * 0.5
            
            # Cap the maximum ease factor
            new_ease_factor = min(self.MAX_EASE_FACTOR, new_ease_factor)
        else:
            # Failed review - decrease ease factor and reset interval
            new_ease_factor -= self.EASE_FACTOR_PENALTY
            new_ease_factor = max(self.MIN_EASE_FACTOR, new_ease_factor)
        
        # Calculate new interval
        if quality < self.QUALITY_THRESHOLD:
            # Failed review - start over
            new_interval = self.INITIAL_INTERVAL
        elif review_count == 0:
            # First review
            new_interval = self.INITIAL_INTERVAL
        elif review_count == 1:
            # Second review
            new_interval = self.SECOND_INTERVAL
        else:
            # Subsequent reviews - use ease factor
            previous_interval = self._estimate_previous_interval(review_count, ease_factor)
            new_interval = max(self.MINIMUM_INTERVAL, int(previous_interval * new_ease_factor))
        
        return new_ease_factor, new_interval
    
    def _estimate_previous_interval(self, review_count: int, ease_factor: float) -> int:
        """Estimate previous interval based on review count and ease factor"""
        
        if review_count <= 1:
            return self.INITIAL_INTERVAL
        elif review_count == 2:
            return self.SECOND_INTERVAL
        else:
            # Estimate based on exponential growth
            interval = self.SECOND_INTERVAL
            for _ in range(review_count - 2):
                interval = int(interval * ease_factor)
            return interval
    
    def _calculate_quality_from_performance(
        self, 
        is_correct: bool, 
        response_time: Optional[int] = None,
        expected_time: int = 60
    ) -> int:
        """Convert performance metrics to SM-2 quality score (0-5)"""
        
        if not is_correct:
            return 0  # Complete failure
        
        # Base quality for correct answer
        base_quality = 3
        
        if response_time and expected_time:
            # Adjust based on response time
            time_ratio = response_time / expected_time
            
            if time_ratio <= 0.5:
                # Very fast - excellent
                base_quality = 5
            elif time_ratio <= 0.8:
                # Fast - good
                base_quality = 4
            elif time_ratio <= 1.2:
                # Normal - satisfactory
                base_quality = 3
            else:
                # Slow - but still correct
                base_quality = 2
        
        return base_quality
    
    async def _invalidate_review_caches(self, user_id: str):
        """Invalidate spaced repetition related caches"""
        
        cache_keys = [
            f"due_reviews:{user_id}",
            f"review_statistics:{user_id}",
            f"learning_progress:{user_id}",
            f"review_calendar:{user_id}"
        ]
        
        for key in cache_keys:
            await cache_service.delete(key)
    
    async def process_answer_for_spaced_repetition(
        self, 
        db: AsyncSession, 
        user_id: str,
        question_id: str,
        is_correct: bool,
        response_time: Optional[int] = None,
        expected_time: int = 60
    ) -> Dict[str, Any]:
        """Process an answer and update spaced repetition schedule"""
        
        # Convert performance to quality score
        quality = self._calculate_quality_from_performance(
            is_correct, response_time, expected_time
        )
        
        # Schedule next review
        schedule_result = await self.schedule_question_review(
            db, user_id, question_id, quality, response_time
        )
        
        return {
            "spaced_repetition": schedule_result,
            "quality_score": quality,
            "is_correct": is_correct,
            "response_time": response_time
        }


# Global spaced repetition service instance
spaced_repetition_service = SpacedRepetitionService()