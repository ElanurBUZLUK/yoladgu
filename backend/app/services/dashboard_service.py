from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from datetime import datetime, timedelta
from app.models.user import User, LearningStyle
from app.models.question import Subject
from app.models.student_attempt import StudentAttempt
from app.models.error_pattern import ErrorPattern
from app.models.spaced_repetition import SpacedRepetition
from app.schemas.dashboard import (
    DashboardData, DashboardStats, SubjectProgress, 
    SubjectSelectionResponse, LearningStyleAdaptation,
    PerformanceSummary, WeeklyProgress, RecommendationItem,
    UserPreferences
)
from app.core.cache import cache_service
import json


class DashboardService:
    """Dashboard and subject selection service"""
    
    def __init__(self):
        pass
    
    async def get_subject_selection_data(self, db: AsyncSession, user: User) -> SubjectSelectionResponse:
        """Get data for subject selection screen"""
        
        # Get user's last selected subject from cache
        cache_key = f"last_subject:{user.id}"
        last_selected_str = await cache_service.get(cache_key)
        last_selected = None
        
        if last_selected_str:
            try:
                last_selected = Subject(last_selected_str)
            except ValueError:
                pass
        
        # Generate recommendations based on user profile
        recommendations = await self._generate_subject_recommendations(db, user)
        
        return SubjectSelectionResponse(
            available_subjects=[Subject.MATH, Subject.ENGLISH],
            user_levels={
                "math": user.current_math_level,
                "english": user.current_english_level
            },
            recommendations=recommendations,
            last_selected=last_selected
        )
    
    async def select_subject(self, db: AsyncSession, user: User, subject: Subject) -> Dict[str, Any]:
        """Handle subject selection and return initial dashboard data"""
        
        # Cache the selected subject
        cache_key = f"last_subject:{user.id}"
        await cache_service.set(cache_key, subject.value, expire=86400)  # 24 hours
        
        # Get dashboard data for the selected subject
        dashboard_data = await self.get_dashboard_data(db, user, subject)
        
        return {
            "selected_subject": subject.value,
            "dashboard_data": dashboard_data,
            "message": f"{subject.value.title()} subject selected successfully"
        }
    
    async def get_dashboard_data(self, db: AsyncSession, user: User, subject: Optional[Subject] = None) -> DashboardData:
        """Get comprehensive dashboard data"""
        
        # Get overall stats
        overall_stats = await self._get_overall_stats(db, user)
        
        # Get subject-specific progress
        math_progress = None
        english_progress = None
        
        if subject is None or subject == Subject.MATH:
            math_progress = await self._get_subject_progress(db, user, Subject.MATH)
        
        if subject is None or subject == Subject.ENGLISH:
            english_progress = await self._get_subject_progress(db, user, Subject.ENGLISH)
        
        # Get recent activity
        recent_activity = await self._get_recent_activity(db, user, limit=10)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(db, user, subject)
        
        # Get achievements (placeholder)
        achievements = await self._get_achievements(db, user)
        
        return DashboardData(
            user_info={
                "id": str(user.id),
                "username": user.username,
                "learning_style": user.learning_style.value,
                "math_level": user.current_math_level,
                "english_level": user.current_english_level
            },
            math_progress=math_progress,
            english_progress=english_progress,
            overall_stats=overall_stats,
            recent_activity=recent_activity,
            recommendations=recommendations,
            achievements=achievements
        )
    
    async def get_learning_style_adaptation(self, user: User) -> LearningStyleAdaptation:
        """Get learning style specific adaptations"""
        
        adaptations = {
            LearningStyle.VISUAL: {
                "use_colors": True,
                "highlight_keywords": True,
                "include_diagrams": True,
                "visual_progress_bars": True,
                "color_coded_difficulty": True
            },
            LearningStyle.AUDITORY: {
                "text_to_speech": True,
                "audio_feedback": True,
                "verbal_instructions": True,
                "sound_effects": True,
                "audio_progress_updates": True
            },
            LearningStyle.KINESTHETIC: {
                "interactive_elements": True,
                "drag_and_drop": True,
                "step_by_step_guidance": True,
                "hands_on_activities": True,
                "gesture_controls": True
            },
            LearningStyle.MIXED: {
                "multiple_formats": True,
                "flexible_presentation": True,
                "choice_of_interaction": True,
                "adaptive_interface": True,
                "personalized_experience": True
            }
        }
        
        recommended_features = {
            LearningStyle.VISUAL: [
                "Color-coded difficulty levels",
                "Visual progress tracking",
                "Diagram-based explanations",
                "Highlighted key concepts"
            ],
            LearningStyle.AUDITORY: [
                "Audio question reading",
                "Verbal feedback",
                "Sound-based notifications",
                "Audio explanations"
            ],
            LearningStyle.KINESTHETIC: [
                "Interactive question elements",
                "Drag-and-drop activities",
                "Step-by-step problem solving",
                "Touch-based interactions"
            ],
            LearningStyle.MIXED: [
                "Multiple interaction methods",
                "Customizable interface",
                "Adaptive content presentation",
                "Flexible learning paths"
            ]
        }
        
        ui_preferences = {
            LearningStyle.VISUAL: {
                "theme": "high_contrast",
                "font_size": "large",
                "animations": True,
                "progress_visualization": "detailed"
            },
            LearningStyle.AUDITORY: {
                "audio_enabled": True,
                "sound_feedback": True,
                "voice_navigation": True,
                "audio_descriptions": True
            },
            LearningStyle.KINESTHETIC: {
                "touch_friendly": True,
                "large_buttons": True,
                "gesture_support": True,
                "interactive_elements": "enhanced"
            },
            LearningStyle.MIXED: {
                "customizable": True,
                "adaptive_ui": True,
                "multiple_options": True,
                "flexible_layout": True
            }
        }
        
        return LearningStyleAdaptation(
            learning_style=user.learning_style.value,
            adaptations=adaptations.get(user.learning_style, {}),
            recommended_features=recommended_features.get(user.learning_style, []),
            ui_preferences=ui_preferences.get(user.learning_style, {})
        )
    
    async def get_performance_summary(
        self, 
        db: AsyncSession, 
        user: User, 
        subject: Subject, 
        period: str = "week"
    ) -> PerformanceSummary:
        """Get performance summary for a specific period"""
        
        # Calculate date range based on period
        now = datetime.utcnow()
        if period == "today":
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            start_date = now - timedelta(days=7)
        elif period == "month":
            start_date = now - timedelta(days=30)
        else:  # all_time
            start_date = datetime.min
        
        # Get attempts for the period
        query = select(StudentAttempt).where(
            and_(
                StudentAttempt.user_id == user.id,
                StudentAttempt.attempt_date >= start_date
            )
        ).join(StudentAttempt.question).where(
            StudentAttempt.question.has(subject=subject)
        )
        
        result = await db.execute(query)
        attempts = result.scalars().all()
        
        if not attempts:
            return PerformanceSummary(
                subject=subject,
                period=period,
                questions_attempted=0,
                correct_answers=0,
                accuracy_rate=0.0,
                average_time_per_question=0.0,
                improvement_rate=0.0,
                difficulty_distribution={},
                topic_performance={}
            )
        
        # Calculate metrics
        total_attempts = len(attempts)
        correct_attempts = sum(1 for attempt in attempts if attempt.is_correct)
        accuracy_rate = (correct_attempts / total_attempts) * 100 if total_attempts > 0 else 0
        
        total_time = sum(attempt.time_spent or 0 for attempt in attempts)
        avg_time = total_time / total_attempts if total_attempts > 0 else 0
        
        # Calculate improvement rate (placeholder - would need historical data)
        improvement_rate = 0.0  # TODO: Implement based on previous period comparison
        
        # Difficulty distribution
        difficulty_dist = {}
        topic_performance = {}
        
        # These would be calculated from actual question data
        # For now, return placeholder data
        
        return PerformanceSummary(
            subject=subject,
            period=period,
            questions_attempted=total_attempts,
            correct_answers=correct_attempts,
            accuracy_rate=accuracy_rate,
            average_time_per_question=avg_time,
            improvement_rate=improvement_rate,
            difficulty_distribution=difficulty_dist,
            topic_performance=topic_performance
        )
    
    async def get_weekly_progress(self, db: AsyncSession, user: User) -> WeeklyProgress:
        """Get weekly progress data"""
        
        # Calculate week boundaries
        now = datetime.utcnow()
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        
        # Get daily stats for the week
        daily_stats = []
        for i in range(7):
            day = week_start + timedelta(days=i)
            day_end = day + timedelta(hours=23, minutes=59, seconds=59)
            
            # Get attempts for this day
            query = select(StudentAttempt).where(
                and_(
                    StudentAttempt.user_id == user.id,
                    StudentAttempt.attempt_date >= day,
                    StudentAttempt.attempt_date <= day_end
                )
            )
            
            result = await db.execute(query)
            day_attempts = result.scalars().all()
            
            total_attempts = len(day_attempts)
            correct_attempts = sum(1 for attempt in day_attempts if attempt.is_correct)
            accuracy = (correct_attempts / total_attempts * 100) if total_attempts > 0 else 0
            time_spent = sum(attempt.time_spent or 0 for attempt in day_attempts)
            
            daily_stats.append({
                "date": day.strftime("%Y-%m-%d"),
                "questions_answered": total_attempts,
                "correct_answers": correct_attempts,
                "accuracy_rate": accuracy,
                "time_spent": time_spent
            })
        
        # Calculate weekly totals
        total_questions = sum(day["questions_answered"] for day in daily_stats)
        total_time = sum(day["time_spent"] for day in daily_stats)
        accuracy_trend = [day["accuracy_rate"] for day in daily_stats]
        
        return WeeklyProgress(
            week_start=week_start.strftime("%Y-%m-%d"),
            week_end=week_end.strftime("%Y-%m-%d"),
            daily_stats=daily_stats,
            weekly_goals={
                "target_questions": 50,
                "target_accuracy": 80,
                "target_time": 300  # 5 hours in minutes
            },
            achievements_unlocked=[],  # Placeholder
            total_time_spent=total_time,
            questions_answered=total_questions,
            accuracy_trend=accuracy_trend
        )
    
    # Private helper methods
    async def _get_overall_stats(self, db: AsyncSession, user: User) -> DashboardStats:
        """Get overall user statistics"""
        
        # Get all attempts
        result = await db.execute(
            select(StudentAttempt).where(StudentAttempt.user_id == user.id)
        )
        all_attempts = result.scalars().all()
        
        # Get today's attempts
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_attempts = [a for a in all_attempts if a.attempt_date >= today]
        
        # Calculate stats
        total_questions = len(all_attempts)
        correct_answers = sum(1 for a in all_attempts if a.is_correct)
        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        
        questions_today = len(today_attempts)
        time_today = sum(a.time_spent or 0 for a in today_attempts)
        time_total = sum(a.time_spent or 0 for a in all_attempts)
        
        # Calculate streaks (placeholder)
        current_streak = 0
        best_streak = 0
        
        return DashboardStats(
            total_questions_answered=total_questions,
            correct_answers=correct_answers,
            accuracy_percentage=accuracy,
            current_level=max(user.current_math_level, user.current_english_level),
            questions_today=questions_today,
            current_streak=current_streak,
            best_streak=best_streak,
            time_spent_today=time_today,
            time_spent_total=time_total
        )
    
    async def _get_subject_progress(self, db: AsyncSession, user: User, subject: Subject) -> SubjectProgress:
        """Get progress for a specific subject"""
        
        current_level = user.current_math_level if subject == Subject.MATH else user.current_english_level
        
        # Get attempts for this subject
        # This would join with questions table to filter by subject
        # For now, return placeholder data
        
        # Get spaced repetition queue count
        sr_result = await db.execute(
            select(func.count(SpacedRepetition.question_id)).where(
                and_(
                    SpacedRepetition.user_id == user.id,
                    SpacedRepetition.next_review_at <= datetime.utcnow()
                )
            )
        )
        next_review_count = sr_result.scalar() or 0
        
        # Get error patterns for weak areas
        error_result = await db.execute(
            select(ErrorPattern.error_type).where(
                and_(
                    ErrorPattern.user_id == user.id,
                    ErrorPattern.subject == subject
                )
            ).order_by(desc(ErrorPattern.error_count)).limit(3)
        )
        weak_areas = [row[0] for row in error_result.fetchall()]
        
        return SubjectProgress(
            subject=subject,
            current_level=current_level,
            progress_percentage=75.0,  # Placeholder
            total_questions=0,  # Will be calculated from actual data
            correct_answers=0,  # Will be calculated from actual data
            accuracy_rate=0.0,  # Will be calculated from actual data
            last_activity=None,  # Will be calculated from actual data
            next_review_count=next_review_count,
            weak_areas=weak_areas,
            strong_areas=[]  # Placeholder
        )
    
    async def _get_recent_activity(self, db: AsyncSession, user: User, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent user activity"""
        
        result = await db.execute(
            select(StudentAttempt).where(
                StudentAttempt.user_id == user.id
            ).order_by(desc(StudentAttempt.attempt_date)).limit(limit)
        )
        
        attempts = result.scalars().all()
        
        activity = []
        for attempt in attempts:
            activity.append({
                "id": str(attempt.id),
                "type": "question_attempt",
                "timestamp": attempt.attempt_date.isoformat(),
                "is_correct": attempt.is_correct,
                "time_spent": attempt.time_spent,
                "question_id": str(attempt.question_id)
            })
        
        return activity
    
    async def _generate_subject_recommendations(self, db: AsyncSession, user: User) -> List[str]:
        """Generate subject selection recommendations"""
        
        recommendations = []
        
        # Based on learning style
        if user.learning_style == LearningStyle.VISUAL:
            recommendations.append("Mathematics offers great visual problem-solving opportunities")
        elif user.learning_style == LearningStyle.AUDITORY:
            recommendations.append("English practice can improve your listening and speaking skills")
        elif user.learning_style == LearningStyle.KINESTHETIC:
            recommendations.append("Try interactive math problems for hands-on learning")
        
        # Based on levels
        if user.current_math_level < user.current_english_level:
            recommendations.append("Consider focusing on Mathematics to balance your skills")
        elif user.current_english_level < user.current_math_level:
            recommendations.append("English practice could help improve your overall performance")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("Both subjects offer unique learning opportunities")
        
        return recommendations
    
    async def _generate_recommendations(self, db: AsyncSession, user: User, subject: Optional[Subject]) -> List[str]:
        """Generate personalized recommendations"""
        
        recommendations = []
        
        # Get error patterns
        error_result = await db.execute(
            select(ErrorPattern).where(ErrorPattern.user_id == user.id)
            .order_by(desc(ErrorPattern.error_count)).limit(3)
        )
        error_patterns = error_result.scalars().all()
        
        for pattern in error_patterns:
            if pattern.subject == Subject.MATH:
                recommendations.append(f"Practice {pattern.error_type} problems to improve your math skills")
            else:
                recommendations.append(f"Focus on {pattern.error_type} to enhance your English")
        
        # Learning style recommendations
        if user.learning_style == LearningStyle.VISUAL:
            recommendations.append("Try using diagrams and visual aids while studying")
        elif user.learning_style == LearningStyle.AUDITORY:
            recommendations.append("Consider reading questions aloud or using audio features")
        elif user.learning_style == LearningStyle.KINESTHETIC:
            recommendations.append("Use interactive elements and hands-on practice")
        
        # Default recommendations
        if not recommendations:
            recommendations.extend([
                "Practice regularly to maintain your learning momentum",
                "Review your mistakes to learn from them",
                "Set daily learning goals to track your progress"
            ])
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    async def _get_achievements(self, db: AsyncSession, user: User) -> List[Dict[str, Any]]:
        """Get user achievements (placeholder)"""
        
        achievements = []
        
        # Get total attempts for achievement calculation
        result = await db.execute(
            select(func.count(StudentAttempt.id)).where(StudentAttempt.user_id == user.id)
        )
        total_attempts = result.scalar() or 0
        
        # Basic achievements
        if total_attempts >= 10:
            achievements.append({
                "id": "first_10",
                "title": "Getting Started",
                "description": "Answered your first 10 questions",
                "icon": "🎯",
                "unlocked_at": "2024-01-01T00:00:00Z"
            })
        
        if total_attempts >= 50:
            achievements.append({
                "id": "half_century",
                "title": "Half Century",
                "description": "Answered 50 questions",
                "icon": "🏆",
                "unlocked_at": "2024-01-01T00:00:00Z"
            })
        
        return achievements


# Global dashboard service instance
dashboard_service = DashboardService()