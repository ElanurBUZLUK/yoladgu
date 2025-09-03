"""
Attempt repository for tracking student performance.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, func, desc

from app.db.models import Attempt
from app.db.repositories.base import BaseRepository


class AttemptRepository(BaseRepository[Attempt]):
    """Attempt repository for performance tracking."""
    
    def __init__(self):
        super().__init__(Attempt)
    
    async def create_attempt(
        self,
        session: AsyncSession,
        user_id: str,
        item_id: str,
        item_type: str,
        answer: str,
        correct: bool,
        time_ms: Optional[int] = None,
        hints_used: int = 0,
        context: Optional[Dict[str, Any]] = None
    ) -> Attempt:
        """Create a new attempt record."""
        attempt_data = {
            "user_id": user_id,
            "item_id": item_id,
            "item_type": item_type,
            "answer": answer,
            "correct": correct,
            "time_ms": time_ms,
            "hints_used": hints_used,
            "context": context or {}
        }
        
        return await self.create(session, obj_in=attempt_data)
    
    async def get_user_attempts(
        self,
        session: AsyncSession,
        user_id: str,
        item_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Attempt]:
        """Get user's attempts with optional filtering by item type."""
        conditions = [Attempt.user_id == user_id]
        
        if item_type:
            conditions.append(Attempt.item_type == item_type)
        
        statement = select(Attempt).where(
            and_(*conditions)
        ).order_by(desc(Attempt.created_at)).offset(offset).limit(limit)
        
        result = await session.exec(statement)
        return result.all()
    
    async def get_item_attempts(
        self,
        session: AsyncSession,
        item_id: str,
        limit: int = 100
    ) -> List[Attempt]:
        """Get all attempts for a specific item."""
        statement = select(Attempt).where(
            Attempt.item_id == item_id
        ).order_by(desc(Attempt.created_at)).limit(limit)
        
        result = await session.exec(statement)
        return result.all()
    
    async def get_user_performance_stats(
        self,
        session: AsyncSession,
        user_id: str,
        item_type: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get user performance statistics."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        conditions = [
            Attempt.user_id == user_id,
            Attempt.created_at >= since_date
        ]
        
        if item_type:
            conditions.append(Attempt.item_type == item_type)
        
        # Total attempts
        total_statement = select(func.count(Attempt.id)).where(and_(*conditions))
        total_result = await session.exec(total_statement)
        total_attempts = total_result.one()
        
        # Correct attempts
        correct_conditions = conditions + [Attempt.correct == True]
        correct_statement = select(func.count(Attempt.id)).where(and_(*correct_conditions))
        correct_result = await session.exec(correct_statement)
        correct_attempts = correct_result.one()
        
        # Average time
        time_statement = select(func.avg(Attempt.time_ms)).where(
            and_(*conditions, Attempt.time_ms.is_not(None))
        )
        time_result = await session.exec(time_statement)
        avg_time_ms = time_result.one()
        
        # Success rate
        success_rate = correct_attempts / total_attempts if total_attempts > 0 else 0.0
        
        return {
            "total_attempts": total_attempts,
            "correct_attempts": correct_attempts,
            "success_rate": success_rate,
            "avg_time_ms": avg_time_ms,
            "period_days": days
        }
    
    async def get_recent_attempts(
        self,
        session: AsyncSession,
        user_id: str,
        hours: int = 24,
        item_type: Optional[str] = None
    ) -> List[Attempt]:
        """Get user's recent attempts."""
        since_time = datetime.utcnow() - timedelta(hours=hours)
        
        conditions = [
            Attempt.user_id == user_id,
            Attempt.created_at >= since_time
        ]
        
        if item_type:
            conditions.append(Attempt.item_type == item_type)
        
        statement = select(Attempt).where(
            and_(*conditions)
        ).order_by(desc(Attempt.created_at))
        
        result = await session.exec(statement)
        return result.all()
    
    async def get_error_patterns(
        self,
        session: AsyncSession,
        user_id: str,
        item_type: str,
        days: int = 30
    ) -> List[Tuple[str, int]]:
        """Get error patterns for IRT/error profile updates."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        # Get incorrect attempts
        statement = select(Attempt).where(
            and_(
                Attempt.user_id == user_id,
                Attempt.item_type == item_type,
                Attempt.correct == False,
                Attempt.created_at >= since_date
            )
        )
        
        result = await session.exec(statement)
        incorrect_attempts = result.all()
        
        # This would typically join with item tables to get skill/error information
        # For now, return item_ids and their error counts
        error_counts = {}
        for attempt in incorrect_attempts:
            item_id = attempt.item_id
            error_counts[item_id] = error_counts.get(item_id, 0) + 1
        
        return list(error_counts.items())
    
    async def get_learning_curve_data(
        self,
        session: AsyncSession,
        user_id: str,
        item_type: Optional[str] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get data for learning curve analysis."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        conditions = [
            Attempt.user_id == user_id,
            Attempt.created_at >= since_date
        ]
        
        if item_type:
            conditions.append(Attempt.item_type == item_type)
        
        statement = select(Attempt).where(
            and_(*conditions)
        ).order_by(Attempt.created_at)
        
        result = await session.exec(statement)
        attempts = result.all()
        
        # Group by day and calculate daily success rates
        daily_stats = {}
        for attempt in attempts:
            day = attempt.created_at.date()
            if day not in daily_stats:
                daily_stats[day] = {"total": 0, "correct": 0}
            
            daily_stats[day]["total"] += 1
            if attempt.correct:
                daily_stats[day]["correct"] += 1
        
        # Convert to list format
        curve_data = []
        for day, stats in sorted(daily_stats.items()):
            success_rate = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            curve_data.append({
                "date": day,
                "total_attempts": stats["total"],
                "correct_attempts": stats["correct"],
                "success_rate": success_rate
            })
        
        return curve_data
    
    async def get_item_difficulty_data(
        self,
        session: AsyncSession,
        item_id: str,
        min_attempts: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Get item difficulty calibration data."""
        # Get all attempts for this item
        statement = select(Attempt).where(Attempt.item_id == item_id)
        result = await session.exec(statement)
        attempts = result.all()
        
        if len(attempts) < min_attempts:
            return None
        
        total_attempts = len(attempts)
        correct_attempts = sum(1 for a in attempts if a.correct)
        success_rate = correct_attempts / total_attempts
        
        # Calculate average response time
        times = [a.time_ms for a in attempts if a.time_ms is not None]
        avg_time = sum(times) / len(times) if times else None
        
        return {
            "item_id": item_id,
            "total_attempts": total_attempts,
            "correct_attempts": correct_attempts,
            "success_rate": success_rate,
            "avg_time_ms": avg_time,
            "sample_size": total_attempts
        }


# Create repository instance
attempt_repository = AttemptRepository()