from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.core.database import get_async_session
from app.services.dashboard_service import dashboard_service
from app.schemas.dashboard import (
    DashboardData, SubjectSelectionResponse, LearningStyleAdaptation,
    PerformanceSummary, WeeklyProgress, SubjectChoice
)
from app.middleware.auth import get_current_active_user, get_current_student
from app.models.user import User
from app.models.question import Subject

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/subject-selection", response_model=SubjectSelectionResponse)
async def get_subject_selection_data(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get data for subject selection screen"""
    
    return await dashboard_service.get_subject_selection_data(db, current_user)


@router.post("/select-subject")
async def select_subject(
    subject_choice: SubjectChoice,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Select a subject and get initial dashboard data"""
    
    result = await dashboard_service.select_subject(db, current_user, subject_choice.subject)
    return result


@router.get("/", response_model=DashboardData)
async def get_dashboard_data(
    subject: Optional[Subject] = Query(None, description="Filter by specific subject"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get comprehensive dashboard data"""
    
    return await dashboard_service.get_dashboard_data(db, current_user, subject)


@router.get("/math", response_model=DashboardData)
async def get_math_dashboard(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get mathematics-specific dashboard data"""
    
    return await dashboard_service.get_dashboard_data(db, current_user, Subject.MATH)


@router.get("/english", response_model=DashboardData)
async def get_english_dashboard(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get English-specific dashboard data"""
    
    return await dashboard_service.get_dashboard_data(db, current_user, Subject.ENGLISH)


@router.get("/learning-style", response_model=LearningStyleAdaptation)
async def get_learning_style_adaptation(
    current_user: User = Depends(get_current_active_user)
):
    """Get learning style specific adaptations and UI preferences"""
    
    return await dashboard_service.get_learning_style_adaptation(current_user)


@router.get("/performance/{subject}", response_model=PerformanceSummary)
async def get_performance_summary(
    subject: Subject,
    period: str = Query("week", regex="^(today|week|month|all_time)$"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get performance summary for a specific subject and period"""
    
    return await dashboard_service.get_performance_summary(db, current_user, subject, period)


@router.get("/weekly-progress", response_model=WeeklyProgress)
async def get_weekly_progress(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get weekly progress data"""
    
    return await dashboard_service.get_weekly_progress(db, current_user)


@router.get("/stats/overview")
async def get_stats_overview(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get quick stats overview for dashboard widgets"""
    
    dashboard_data = await dashboard_service.get_dashboard_data(db, current_user)
    
    return {
        "overall_stats": dashboard_data.overall_stats,
        "math_level": current_user.current_math_level,
        "english_level": current_user.current_english_level,
        "learning_style": current_user.learning_style.value,
        "recent_activity_count": len(dashboard_data.recent_activity),
        "recommendations_count": len(dashboard_data.recommendations),
        "achievements_count": len(dashboard_data.achievements)
    }


@router.get("/recommendations")
async def get_recommendations(
    subject: Optional[Subject] = Query(None, description="Filter by specific subject"),
    limit: int = Query(5, ge=1, le=20, description="Number of recommendations to return"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get personalized recommendations"""
    
    dashboard_data = await dashboard_service.get_dashboard_data(db, current_user, subject)
    
    return {
        "recommendations": dashboard_data.recommendations[:limit],
        "learning_style_adaptations": await dashboard_service.get_learning_style_adaptation(current_user)
    }


@router.get("/progress-tracking")
async def get_progress_tracking(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get detailed progress tracking data"""
    
    dashboard_data = await dashboard_service.get_dashboard_data(db, current_user)
    
    return {
        "math_progress": dashboard_data.math_progress,
        "english_progress": dashboard_data.english_progress,
        "overall_stats": dashboard_data.overall_stats,
        "weekly_progress": await dashboard_service.get_weekly_progress(db, current_user)
    }


@router.get("/subject-performance-comparison")
async def get_subject_performance_comparison(
    period: str = Query("week", regex="^(today|week|month|all_time)$"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Compare performance between subjects"""
    
    math_performance = await dashboard_service.get_performance_summary(
        db, current_user, Subject.MATH, period
    )
    english_performance = await dashboard_service.get_performance_summary(
        db, current_user, Subject.ENGLISH, period
    )
    
    return {
        "period": period,
        "math_performance": math_performance,
        "english_performance": english_performance,
        "comparison": {
            "better_subject": "math" if math_performance.accuracy_rate > english_performance.accuracy_rate else "english",
            "accuracy_difference": abs(math_performance.accuracy_rate - english_performance.accuracy_rate),
            "total_questions": math_performance.questions_attempted + english_performance.questions_attempted,
            "overall_accuracy": (
                (math_performance.correct_answers + english_performance.correct_answers) /
                max(1, math_performance.questions_attempted + english_performance.questions_attempted) * 100
            )
        }
    }


@router.get("/health")
async def dashboard_health():
    """Dashboard module health check"""
    return {"status": "ok", "module": "dashboard"}