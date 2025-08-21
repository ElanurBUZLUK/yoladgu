from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Any, Dict
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from app.database_enhanced import enhanced_database_manager
from app.services.answer_evaluation_service import answer_evaluation_service
from app.services.error_pattern_service import error_pattern_service
from app.services.level_adjustment_service import level_adjustment_service
from app.services.spaced_repetition_service import spaced_repetition_service
from app.schemas.answer import (
    AnswerSubmission, AnswerEvaluation, AnswerEvaluationRequest,
    StudentAttemptResponse, ErrorAnalysisResult, PerformanceMetrics,
    DetailedErrorAnalysis, LevelAdjustmentRecommendation, FeedbackGeneration
)
from app.schemas.spaced_repetition import (
    ScheduleReviewRequest, ReviewScheduleResult, DueReview, ReviewStatistics,
    ReviewCalendar, LearningProgress, BulkReviewRequest, BulkReviewResponse
)
from app.middleware.auth import get_current_student, get_current_teacher, get_current_admin
from app.models.user import User
from app.models.question import Subject
from app.utils.distlock_idem import idempotency_decorator, IdempotencyConfig # Added import

router = APIRouter(prefix="/api/v1/answers", tags=["answers"])

# New Response Models
class AnswerSubmissionResponse(BaseModel):
    success: bool
    evaluation: AnswerEvaluation
    attempt: StudentAttemptResponse
    spaced_repetition: Optional[Any] # This might need a more specific schema later
    message: str

class ReviewCalendarResponse(BaseModel):
    calendar: ReviewCalendar
    total_days: int
    total_reviews: int
    days_ahead: int

class ResetQuestionProgressResponse(BaseModel):
    success: bool
    message: str

class ProcessAnswerForReviewResponse(BaseModel):
    success: bool
    message: str

class UserReviewStatisticsAdminResponse(BaseModel):
    user_id: str
    statistics: ReviewStatistics
    requested_by: str

class UserLearningProgressAdminResponse(BaseModel):
    user_id: str
    progress: LearningProgress
    requested_by: str

class ResetUserQuestionProgressAdminResponse(BaseModel):
    user_id: str
    reset_result: ResetQuestionProgressResponse
    applied_by: str

class PerformanceSummary(BaseModel):
    accuracy_rate: float
    current_level: int
    total_attempts: int
    current_streak: int
    progress_percentage: float
    overdue_reviews: int
    due_today: int

class RecommendationItem(BaseModel):
    type: str
    priority: str
    title: str
    description: str
    action: str
    confidence: float

class PerformanceBasedRecommendationsResponse(BaseModel):
    subject: str
    user_id: str
    performance_summary: PerformanceSummary
    recommendations: List[RecommendationItem]
    total_recommendations: int
    generated_at: str

class ActivityItem(BaseModel):
    type: str
    title: str
    duration_minutes: int
    priority: str
    questions: Optional[int] = None

class StudyPlanDay(BaseModel):
    date: str
    day_name: str
    total_time_minutes: int
    activities: List[ActivityItem]
    scheduled_reviews: int
    focus_area: str

class StudyPlanRecommendationsResponse(BaseModel):
    subject: str
    user_id: str
    study_plan: List[StudyPlanDay]
    total_days: int
    average_daily_time: float
    total_scheduled_reviews: int
    generated_at: str

class RecentPerformance(BaseModel):
    attempts_analyzed: int
    accuracy: float
    average_difficulty: float
    average_time_seconds: float
    current_streak: int

class AdaptiveRecommendationItem(BaseModel):
    type: str
    title: str
    description: str
    immediate_action: str
    confidence: float

class AdaptiveRecommendationsResponse(BaseModel):
    subject: str
    message: Optional[str] = None
    user_id: Optional[str] = None
    recent_performance: Optional[RecentPerformance] = None
    recommendations: List[AdaptiveRecommendationItem]
    next_suggested_action: Optional[str] = None
    generated_at: Optional[str] = None

class AnswersHealthResponse(BaseModel):
    status: str
    module: str

class QuestionDetails(BaseModel):
    id: str
    content: str
    correct_answer: str
    options: Optional[List[str]] = None
    topic_category: Optional[str] = None
    source_type: str

class MathErrorDetailItem(BaseModel):
    operation: str
    math_concept: str
    error_step: str

class ErrorDetails(BaseModel):
    grammar_errors: Optional[List[str]] = None
    vocabulary_errors: Optional[List[str]] = None
    math_error_details: Optional[List[MathErrorDetailItem]] = None

class AttemptDetailsResponse(BaseModel):
    attempt: StudentAttemptResponse
    question_details: QuestionDetails
    error_details: ErrorDetails

class FeedbackResponse(BaseModel):
    question_id: str
    attempt_id: str
    is_correct: bool
    personalized_feedback: str
    generated_at: str

class OverallSummary(BaseModel):
    total_attempts: int
    accuracy_rate: float
    current_streak: int
    best_streak: int

class SubjectSummary(BaseModel):
    attempts: int
    accuracy: float
    current_level: int

class RecentErrorItem(BaseModel):
    error_type: str
    frequency: int
    subject: str

class StatisticsSummaryResponse(BaseModel):
    overall: OverallSummary
    by_subject: Dict[str, SubjectSummary]
    recent_errors: List[RecentErrorItem]
    improvement_areas: List[str]

class SimilarStudentItem(BaseModel):
    user_id: str
    similarity_score: float

class SimilarStudentsResponse(BaseModel):
    similar_students: List[SimilarStudentItem]
    total_found: int
    subject: str
    user_id: str

class InterventionItem(BaseModel):
    type: str
    description: str
    action: str

class InterventionRecommendationsResponse(BaseModel):
    interventions: List[InterventionItem]
    total_recommendations: int
    subject: str
    generated_at: str

class TrackedErrorPattern(BaseModel):
    id: str
    error_type: str
    error_count: int
    subject: str
    last_occurrence: str

class TrackErrorPatternResponse(BaseModel):
    success: bool
    error_pattern: TrackedErrorPattern
    message: str

class CommonErrorDetail(BaseModel):
    error_type: str
    frequency: int
    student_count: Optional[int] = None

class ClassErrorAnalyticsResponse(BaseModel):
    total_error_types: int
    affected_students: int
    most_common_errors: List[CommonErrorDetail]

class UserErrorPatternsAdminResponse(BaseModel):
    user_id: str
    error_patterns: List[TrackedErrorPattern]
    total_patterns: int
    subject_filter: str

class OverviewSummary(BaseModel):
    total_error_types: int
    affected_students: int
    analysis_period: str

class SubjectErrorAnalytics(BaseModel):
    error_types: int
    affected_students: int
    top_errors: List[CommonErrorDetail]

class CriticalErrorItem(BaseModel):
    error_type: str
    frequency: int
    subject: str
    student_count: Optional[int] = None

class ErrorAnalyticsOverviewResponse(BaseModel):
    overview: OverviewSummary
    by_subject: Dict[str, SubjectErrorAnalytics]
    critical_errors: List[CriticalErrorItem]
    recommendations: List[str]

class BatchRecommendationItem(BaseModel):
    user_id: str
    current_level: int
    recommended_level: int
    confidence: float

class BatchEvaluateLevelAdjustmentsResponse(BaseModel):
    recommendations: List[BatchRecommendationItem]
    total_users_evaluated: int
    subject: str
    min_attempts_threshold: int
    evaluation_date: str

class AdminApplyLevelAdjustmentResponse(BaseModel):
    success: bool
    adjustment: dict
    target_user_id: str
    applied_by: str
    message: str

class LevelHistoryItem(BaseModel):
    old_level: int
    new_level: int
    reason: str
    applied_at: str

class UserLevelHistoryAdminResponse(BaseModel):
    level_history: List[LevelHistoryItem]
    total_adjustments: int
    user_id: str
    subject_filter: str
    requested_by: str

class LevelDistribution(BaseModel):
    data: Dict[str, int]

class SubjectPendingRecommendations(BaseModel):
    total: int
    promotions: int
    demotions: int

class RecommendationsSummaryItem(BaseModel):
    data: List[BatchRecommendationItem]

class LevelAnalyticsOverviewResponse(BaseModel):
    level_distribution: Dict[str, LevelDistribution]
    pending_recommendations: Dict[str, SubjectPendingRecommendations]
    recommendations_summary: Dict[str, RecommendationsSummaryItem]
    generated_at: str

class DifficultyPerformance(BaseModel):
    attempts: int
    accuracy: float

class CommonError(BaseModel):
    error_type: str
    frequency: int

class ClassPerformanceAnalyticsResponse(BaseModel):
    total_attempts: int
    class_accuracy: float
    active_students: int
    difficulty_performance: Dict[str, DifficultyPerformance]
    common_errors: List[CommonError]
    analysis_period: str
    subject_filter: str


@router.post("/submit", response_model=AnswerSubmissionResponse)
@idempotency_decorator(
    key_builder=lambda submission, current_user: f"submit:{current_user.id}:{submission.question_id}:{hash(submission.student_answer)}",
    config=IdempotencyConfig(scope="answer_submission", ttl_seconds=3600)
)
async def submit_answer(
    submission: AnswerSubmission,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Submit and evaluate a student answer"""
    
    try:
        evaluation, attempt_response = await answer_evaluation_service.submit_answer(
            db, current_user, submission
        )
        
        # Update user level based on performance
        await _update_user_level(db, current_user, evaluation, submission)
        
        # Automatically schedule spaced repetition review
        spaced_repetition_result = None
        try:
            spaced_repetition_result = await spaced_repetition_service.process_answer_for_spaced_repetition(
                db, 
                str(current_user.id), 
                submission.question_id, 
                evaluation.is_correct,
                submission.time_spent,
                expected_time=60  # Default expected time
            )
        except Exception as sr_error:
            # Don't fail the main submission if spaced repetition fails
            print(f"Spaced repetition scheduling failed: {sr_error}")
        
        return {
            "success": True,
            "evaluation": evaluation,
            "attempt": attempt_response,
            "spaced_repetition": spaced_repetition_result,
            "message": "Answer submitted and evaluated successfully"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer submission failed: {str(e)}"
        )

async def _update_user_level(db: AsyncSession, user: User, evaluation: AnswerEvaluation, submission: AnswerSubmission):
    """Update user level based on answer evaluation"""
    try:
        # Get the question to determine subject
        result = await db.execute(
            select(Question).where(Question.id == submission.question_id)
        )
        question = result.scalar_one_or_none()
        
        if not question:
            return
        
        # Get recent performance for this subject
        recent_attempts = await db.execute(
            select(StudentAttempt)
            .where(
                and_(
                    StudentAttempt.user_id == user.id,
                    StudentAttempt.subject == question.subject
                )
            )
            .order_by(desc(StudentAttempt.attempt_date))
            .limit(20)  # Last 20 attempts
        )
        
        attempts = recent_attempts.scalars().all()
        if len(attempts) >= 10:  # Need at least 10 attempts for level adjustment
            recent_accuracy = sum(1 for a in attempts if a.is_correct) / len(attempts)
            
            # Update level based on performance
            if question.subject == Subject.MATH:
                if recent_accuracy > 0.8 and user.current_math_level < 5:
                    user.current_math_level += 1
                elif recent_accuracy < 0.3 and user.current_math_level > 1:
                    user.current_math_level -= 1
            elif question.subject == Subject.ENGLISH:
                if recent_accuracy > 0.8 and user.current_english_level < 5:
                    user.current_english_level += 1
                elif recent_accuracy < 0.3 and user.current_english_level > 1:
                    user.current_english_level -= 1
            
            await db.commit()
            logger.info(f"Updated user {user.id} levels: Math={user.current_math_level}, English={user.current_english_level}")
            
    except Exception as e:
        logger.error(f"Error updating user level: {e}")
        # Don't fail the main submission if level update fails


@router.post("/evaluate", response_model=AnswerEvaluation)
async def evaluate_answer(
    evaluation_request: AnswerEvaluationRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Evaluate an answer without saving to database (for practice/preview)"""
    
    try:
        evaluation = await answer_evaluation_service.evaluate_answer(
            db, current_user, evaluation_request
        )
        
        return evaluation
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer evaluation failed: {str(e)}"
        )


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get comprehensive performance metrics for the current user"""
    
    metrics = await answer_evaluation_service.get_user_performance_metrics(
        db, str(current_user.id), subject, days
    )
    
    return metrics


@router.get("/error-analysis", response_model=List[ErrorAnalysisResult])
async def get_error_analysis(
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get detailed error analysis for the current user"""
    
    error_analysis = await answer_evaluation_service.get_error_analysis(
        db, str(current_user.id), subject
    )
    
    return error_analysis


@router.get("/detailed-error-analysis/{subject}", response_model=DetailedErrorAnalysis)
async def get_detailed_error_analysis(
    subject: Subject,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get detailed subject-specific error analysis"""
    
    detailed_analysis = await answer_evaluation_service.get_detailed_error_analysis(
        db, str(current_user.id), subject
    )
    
    return detailed_analysis


@router.get("/level-recommendation/{subject}")
async def get_level_adjustment_recommendation(
    subject: Subject,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get level adjustment recommendation for a subject"""
    
    recommendation = await level_adjustment_service.evaluate_level_adjustment(
        db, current_user, subject
    )
    
    if recommendation:
        return {
            "has_recommendation": True,
            "recommendation": recommendation,
            "message": f"Level adjustment recommended for {subject.value}"
        }
    else:
        return {
            "has_recommendation": False,
            "message": f"No level adjustment needed for {subject.value} at this time",
            "reason": "Insufficient data or performance within acceptable range"
        }


@router.post("/level-adjustment/{subject}")
@idempotency_decorator(
    key_builder=lambda subject, new_level, reason, current_user: f"level_adj:{current_user.id}:{subject.value}:{new_level}",
    config=IdempotencyConfig(scope="level_adjustment", ttl_seconds=3600)
)
async def apply_level_adjustment(
    subject: Subject,
    new_level: int = Query(..., ge=1, le=5, description="New level (1-5)"),
    reason: str = Query(..., description="Reason for adjustment"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Apply level adjustment for a subject"""
    
    result = await level_adjustment_service.apply_level_adjustment(
        db, current_user, subject, new_level, reason, applied_by=str(current_user.id)
    )
    
    if result["success"]:
        return {
            "success": True,
            "adjustment": result,
            "message": f"Level successfully adjusted to {new_level}"
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )


@router.get("/level-history")
async def get_level_adjustment_history(
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    limit: int = Query(20, ge=1, le=100, description="Number of adjustments to return"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get level adjustment history for the current user"""
    
    history = await level_adjustment_service.get_level_history(
        db, str(current_user.id), subject, limit
    )
    
    return {
        "level_history": history,
        "total_adjustments": len(history),
        "user_id": str(current_user.id),
        "subject_filter": subject.value if subject else "all"
    }


@router.get("/level-progression/{subject}")
async def get_level_progression_prediction(
    subject: Subject,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get level progression prediction for a subject"""
    
    prediction = await level_adjustment_service.predict_level_progression(
        db, current_user, subject
    )
    
    return {
        "progression_prediction": prediction,
        "subject": subject.value,
        "user_id": str(current_user.id),
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/level-statistics/{subject}")
async def get_level_statistics(
    subject: Subject,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get level distribution statistics for a subject"""
    
    statistics = await level_adjustment_service.get_level_statistics(db, subject)
    
    return {
        "statistics": statistics,
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/attempts")
async def get_user_attempts(
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    limit: int = Query(20, ge=1, le=100, description="Number of attempts to return"),
    skip: int = Query(0, ge=0, description="Number of attempts to skip"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get user's answer attempts with pagination"""
    
    from sqlalchemy import select, desc, and_
    from app.models.student_attempt import StudentAttempt
    from app.models.question import Question
    
    # Build query
    query = select(StudentAttempt).join(Question).where(
        StudentAttempt.user_id == current_user.id
    )
    
    if subject:
        query = query.where(Question.subject == subject)
    
    query = query.order_by(desc(StudentAttempt.attempt_date)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    attempts_with_questions = result.fetchall()
    
    # Format response
    attempts = []
    for attempt, question in attempts_with_questions:
        attempts.append(StudentAttemptResponse(
            id=str(attempt.id),
            question_id=str(attempt.question_id),
            student_answer=attempt.student_answer,
            is_correct=attempt.is_correct,
            time_spent=attempt.time_spent,
            attempt_date=attempt.attempt_date.isoformat(),
            error_category=attempt.error_category,
            question_content=question.content,
            question_type=question.question_type.value,
            subject=question.subject.value,
            difficulty_level=question.difficulty_level
        ))
    
    return {
        "attempts": attempts,
        "total_returned": len(attempts),
        "skip": skip,
        "limit": limit,
        "filters": {
            "subject": subject.value if subject else None
        }
    }


@router.get("/attempts/{attempt_id}", response_model=AttemptDetailsResponse)
async def get_attempt_details(
    attempt_id: str,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get detailed information about a specific attempt"""
    
    from sqlalchemy import select
    from app.models.student_attempt import StudentAttempt
    from app.models.question import Question
    from app.models.math_error_detail import MathErrorDetail
    
    # Get attempt with question details
    result = await db.execute(
        select(StudentAttempt, Question).join(Question).where(
            and_(
                StudentAttempt.id == attempt_id,
                StudentAttempt.user_id == current_user.id
            )
        )
    )
    
    attempt_with_question = result.first()
    
    if not attempt_with_question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Attempt not found"
        )
    
    attempt, question = attempt_with_question
    
    # Get math error details if applicable
    math_error_details = None
    if question.subject == Subject.MATH and not attempt.is_correct:
        math_result = await db.execute(
            select(MathErrorDetail).where(MathErrorDetail.attempt_id == attempt.id)
        )
        math_error_details = math_result.scalars().all()
    
    # Format response
    response = {
        "attempt": StudentAttemptResponse(
            id=str(attempt.id),
            question_id=str(attempt.question_id),
            student_answer=attempt.student_answer,
            is_correct=attempt.is_correct,
            time_spent=attempt.time_spent,
            attempt_date=attempt.attempt_date.isoformat(),
            error_category=attempt.error_category,
            question_content=question.content,
            question_type=question.question_type.value,
            subject=question.subject.value,
            difficulty_level=question.difficulty_level
        ),
        "question_details": {
            "id": str(question.id),
            "content": question.content,
            "correct_answer": question.correct_answer,
            "options": question.options,
            "topic_category": question.topic_category,
            "source_type": question.source_type.value
        },
        "error_details": {
            "grammar_errors": attempt.grammar_errors,
            "vocabulary_errors": attempt.vocabulary_errors,
            "math_error_details": [
                {
                    "operation": detail.operation,
                    "math_concept": detail.math_concept,
                    "error_step": detail.error_step
                }
                for detail in (math_error_details or [])
            ]
        }
    }
    
    return response


@router.get("/feedback/{question_id}", response_model=FeedbackResponse)
async def get_personalized_feedback(
    question_id: str,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get personalized feedback for a question (based on latest attempt)"""
    
    from sqlalchemy import select, desc
    from app.models.student_attempt import StudentAttempt
    from app.models.question import Question
    
    # Get latest attempt for this question
    result = await db.execute(
        select(StudentAttempt, Question).join(Question).where(
            and_(
                StudentAttempt.user_id == current_user.id,
                StudentAttempt.question_id == question_id
            )
        ).order_by(desc(StudentAttempt.attempt_date)).limit(1)
    )
    
    attempt_with_question = result.first()
    
    if not attempt_with_question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No attempts found for this question"
        )
    
    attempt, question = attempt_with_question
    
    # Create a mock evaluation for feedback generation
    mock_evaluation = AnswerEvaluation(
        is_correct=attempt.is_correct,
        score=100.0 if attempt.is_correct else 0.0,
        feedback="",
        error_analysis={}
    )
    
    # Generate personalized feedback
    feedback = await answer_evaluation_service.generate_personalized_feedback(
        db, current_user, mock_evaluation, question
    )
    
    return {
        "question_id": question_id,
        "attempt_id": str(attempt.id),
        "is_correct": attempt.is_correct,
        "personalized_feedback": feedback,
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/statistics/summary", response_model=StatisticsSummaryResponse)
async def get_statistics_summary(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get a summary of user's answer statistics"""
    
    # Get performance metrics for both subjects
    math_metrics = await answer_evaluation_service.get_user_performance_metrics(
        db, str(current_user.id), Subject.MATH, 30
    )
    
    english_metrics = await answer_evaluation_service.get_user_performance_metrics(
        db, str(current_user.id), Subject.ENGLISH, 30
    )
    
    overall_metrics = await answer_evaluation_service.get_user_performance_metrics(
        db, str(current_user.id), None, 30
    )
    
    # Get recent error analysis
    recent_errors = await answer_evaluation_service.get_error_analysis(
        db, str(current_user.id), None
    )
    
    return {
        "overall": {
            "total_attempts": overall_metrics.total_attempts,
            "accuracy_rate": overall_metrics.accuracy_rate,
            "current_streak": overall_metrics.current_streak,
            "best_streak": overall_metrics.best_streak
        },
        "by_subject": {
            "math": {
                "attempts": math_metrics.total_attempts,
                "accuracy": math_metrics.accuracy_rate,
                "current_level": current_user.current_math_level
            },
            "english": {
                "attempts": english_metrics.total_attempts,
                "accuracy": english_metrics.accuracy_rate,
                "current_level": current_user.current_english_level
            }
        },
        "recent_errors": [
            {
                "error_type": error.error_type,
                "frequency": error.frequency,
                "subject": error.subject.value
            }
            for error in recent_errors[:5]  # Top 5 errors
        ],
        "improvement_areas": [
            error.error_type for error in recent_errors[:3]
        ]
    }


# Teacher/Admin endpoints
@router.get("/user/{user_id}/performance", response_model=PerformanceMetrics)
async def get_user_performance_metrics_admin(
    user_id: str,
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get performance metrics for any user (Teacher/Admin only)"""
    
    metrics = await answer_evaluation_service.get_user_performance_metrics(
        db, user_id, subject, days
    )
    
    return metrics


@router.get("/user/{user_id}/error-analysis", response_model=List[ErrorAnalysisResult])
async def get_user_error_analysis_admin(
    user_id: str,
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get error analysis for any user (Teacher/Admin only)"""
    
    error_analysis = await answer_evaluation_service.get_error_analysis(
        db, user_id, subject
    )
    
    return error_analysis


@router.get("/analytics/class-performance", response_model=ClassPerformanceAnalyticsResponse)
async def get_class_performance_analytics(
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get class-wide performance analytics (Teacher/Admin only)"""
    
    from sqlalchemy import select, func
    from app.models.student_attempt import StudentAttempt
    from app.models.question import Question
    from app.models.user import User as UserModel
    
    # Date filter
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Base query for attempts
    query = select(StudentAttempt).join(Question).where(
        StudentAttempt.attempt_date >= cutoff_date
    )
    
    if subject:
        query = query.where(Question.subject == subject)
    
    result = await db.execute(query)
    attempts = result.scalars().all()
    
    if not attempts:
        return {
            "total_attempts": 0,
            "class_accuracy": 0.0,
            "active_students": 0,
            "subject_filter": subject.value if subject else "all"
        }
    
    # Calculate class metrics
    total_attempts = len(attempts)
    correct_attempts = sum(1 for a in attempts if a.is_correct)
    class_accuracy = (correct_attempts / total_attempts) * 100
    
    # Get unique students
    unique_students = len(set(str(a.user_id) for a in attempts))
    
    # Performance by difficulty
    difficulty_performance = {}
    for level in range(1, 6):
        level_attempts = [a for a in attempts if hasattr(a, 'question') and a.question.difficulty_level == level]
        if level_attempts:
            level_correct = sum(1 for a in level_attempts if a.is_correct)
            difficulty_performance[str(level)] = {
                "attempts": len(level_attempts),
                "accuracy": (level_correct / len(level_attempts)) * 100
            }
    
    # Most common errors
    error_counts = {}
    for attempt in attempts:
        if not attempt.is_correct and attempt.error_category:
            error_counts[attempt.error_category] = error_counts.get(attempt.error_category, 0) + 1
    
    common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "total_attempts": total_attempts,
        "class_accuracy": class_accuracy,
        "active_students": unique_students,
        "difficulty_performance": difficulty_performance,
        "common_errors": [
            {"error_type": error, "frequency": count}
            for error, count in common_errors
        ],
        "analysis_period": f"{days} days",
        "subject_filter": subject.value if subject else "all"
    }


# Error Pattern Analytics Endpoints
@router.get("/error-patterns/similar-students", response_model=SimilarStudentsResponse)
async def get_similar_students(
    subject: Subject,
    limit: int = Query(10, ge=1, le=50, description="Number of similar students to return"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Find students with similar error patterns"""
    
    similar_students = await error_pattern_service.get_similar_students(
        db, str(current_user.id), subject, limit
    )
    
    return {
        "similar_students": similar_students,
        "total_found": len(similar_students),
        "subject": subject.value,
        "user_id": str(current_user.id)
    }


@router.get("/error-patterns/trend-analysis/{subject}")
async def get_error_trend_analysis(
    subject: Subject,
    days: int = Query(30, ge=7, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get error trend analysis for a subject"""
    
    trend_analysis = await error_pattern_service.get_error_trend_analysis(
        db, str(current_user.id), subject, days
    )
    
    return trend_analysis


@router.get("/error-patterns/interventions/{subject}", response_model=InterventionRecommendationsResponse)
async def get_intervention_recommendations(
    subject: Subject,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get personalized intervention recommendations"""
    
    interventions = await error_pattern_service.recommend_interventions(
        db, str(current_user.id), subject
    )
    
    return {
        "interventions": interventions,
        "total_recommendations": len(interventions),
        "subject": subject.value,
        "generated_at": datetime.utcnow().isoformat()
    }


@router.post("/error-patterns/track", response_model=TrackErrorPatternResponse)
@idempotency_decorator(
    key_builder=lambda error_data, current_user: f"track_error:{current_user.id}:{error_data.get('subject')}:{error_data.get('error_type')}:{hash(frozenset(error_data.items()))}",
    config=IdempotencyConfig(scope="error_pattern_track", ttl_seconds=3600)
)
async def track_error_pattern(
    error_data: dict,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Manually track an error pattern (for testing/admin purposes)"""
    
    required_fields = ["subject", "error_type"]
    for field in required_fields:
        if field not in error_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required field: {field}"
            )
    
    try:
        subject = Subject(error_data["subject"])
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid subject"
        )
    
    error_pattern = await error_pattern_service.track_error_pattern(
        db=db,
        user_id=str(current_user.id),
        subject=subject,
        error_type=error_data["error_type"],
        topic_category=error_data.get("topic_category"),
        difficulty_level=error_data.get("difficulty_level"),
        additional_context=error_data.get("additional_context")
    )
    
    return {
        "success": True,
        "error_pattern": {
            "id": str(error_pattern.id),
            "error_type": error_pattern.error_type,
            "error_count": error_pattern.error_count,
            "subject": error_pattern.subject.value,
            "last_occurrence": error_pattern.last_occurrence.isoformat()
        },
        "message": "Error pattern tracked successfully"
    }


# Teacher/Admin Error Analytics Endpoints
@router.get("/error-patterns/class-analytics/{subject}", response_model=ClassErrorAnalyticsResponse)
async def get_class_error_analytics(
    subject: Subject,
    days: int = Query(30, ge=7, le=365, description="Number of days to analyze"),
    min_students: int = Query(3, ge=1, le=20, description="Minimum students affected for significance"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get class-wide error analytics (Teacher/Admin only)"""
    
    analytics = await error_pattern_service.get_class_error_analytics(
        db, subject, days, min_students
    )
    
    return analytics


@router.get("/error-patterns/user/{user_id}/patterns", response_model=UserErrorPatternsAdminResponse)
async def get_user_error_patterns_admin(
    user_id: str,
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    limit: int = Query(20, ge=1, le=100, description="Number of patterns to return"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get error patterns for any user (Teacher/Admin only)"""
    
    error_patterns = await error_pattern_service.get_user_error_patterns(
        db, user_id, subject, limit
    )
    
    return {
        "user_id": user_id,
        "error_patterns": error_patterns,
        "total_patterns": len(error_patterns),
        "subject_filter": subject.value if subject else "all"
    }


@router.get("/error-patterns/analytics/overview", response_model=ErrorAnalyticsOverviewResponse)
async def get_error_analytics_overview(
    days: int = Query(30, ge=7, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get overview of error analytics across all subjects (Teacher/Admin only)"""
    
    # Get analytics for both subjects
    math_analytics = await error_pattern_service.get_class_error_analytics(
        db, Subject.MATH, days, min_students=2
    )
    
    english_analytics = await error_pattern_service.get_class_error_analytics(
        db, Subject.ENGLISH, days, min_students=2
    )
    
    # Combine and summarize
    total_error_types = (
        math_analytics.get("total_error_types", 0) + 
        english_analytics.get("total_error_types", 0)
    )
    
    total_affected_students = max(
        math_analytics.get("affected_students", 0),
        english_analytics.get("affected_students", 0)
    )
    
    # Most critical errors (affecting most students)
    all_errors = []
    
    for error in math_analytics.get("most_common_errors", [])[:3]:
        error["subject"] = "math"
        all_errors.append(error)
    
    for error in english_analytics.get("most_common_errors", [])[:3]:
        error["subject"] = "english"
        all_errors.append(error)
    
    # Sort by student count
    critical_errors = sorted(all_errors, key=lambda x: x["student_count"], reverse=True)[:5]
    
    return {
        "overview": {
            "total_error_types": total_error_types,
            "affected_students": total_affected_students,
            "analysis_period": f"{days} days"
        },
        "by_subject": {
            "math": {
                "error_types": math_analytics.get("total_error_types", 0),
                "affected_students": math_analytics.get("affected_students", 0),
                "top_errors": math_analytics.get("most_common_errors", [])[:3]
            },
            "english": {
                "error_types": english_analytics.get("total_error_types", 0),
                "affected_students": english_analytics.get("affected_students", 0),
                "top_errors": english_analytics.get("most_common_errors", [])[:3]
            }
        },
        "critical_errors": critical_errors,
        "recommendations": [
            "Focus on the most common errors affecting multiple students",
            "Provide targeted practice for high-frequency error patterns",
            "Consider class-wide interventions for widespread issues"
        ]
    }


# Admin/Teacher Level Management Endpoints
@router.get("/admin/level-evaluations/{subject}", response_model=BatchEvaluateLevelAdjustmentsResponse)
async def batch_evaluate_level_adjustments(
    subject: Subject,
    min_attempts: int = Query(10, ge=5, le=50, description="Minimum attempts required"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Batch evaluate level adjustments for all users (Teacher/Admin only)"""
    
    recommendations = await level_adjustment_service.batch_evaluate_adjustments(
        db, subject, min_attempts
    )
    
    return {
        "recommendations": recommendations,
        "total_users_evaluated": len(recommendations),
        "subject": subject.value,
        "min_attempts_threshold": min_attempts,
        "evaluation_date": datetime.utcnow().isoformat()
    }


@router.post("/admin/level-adjustment/{user_id}/{subject}", response_model=AdminApplyLevelAdjustmentResponse)
@idempotency_decorator(
    key_builder=lambda user_id, subject, new_level, reason, current_user: f"admin_level_adj:{user_id}:{subject.value}:{new_level}",
    config=IdempotencyConfig(scope="admin_level_adjustment", ttl_seconds=3600)
)
async def admin_apply_level_adjustment(
    user_id: str,
    subject: Subject,
    new_level: int = Query(..., ge=1, le=5, description="New level (1-5)"),
    reason: str = Query(..., description="Reason for adjustment"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Apply level adjustment for any user (Teacher/Admin only)"""
    
    from sqlalchemy import select
    from app.models.user import User as UserModel
    
    # Get target user
    result = await db.execute(select(UserModel).where(UserModel.id == user_id))
    target_user = result.scalar_one_or_none()
    
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    result = await level_adjustment_service.apply_level_adjustment(
        db, target_user, subject, new_level, reason, applied_by=str(current_user.id)
    )
    
    if result["success"]:
        return {
            "success": True,
            "adjustment": result,
            "target_user_id": user_id,
            "applied_by": str(current_user.id),
            "message": f"Level successfully adjusted to {new_level} for user {target_user.username}"
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )


@router.get("/admin/level-history/{user_id}", response_model=UserLevelHistoryAdminResponse)
async def get_user_level_history_admin(
    user_id: str,
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    limit: int = Query(20, ge=1, le=100, description="Number of adjustments to return"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get level adjustment history for any user (Teacher/Admin only)"""
    
    history = await level_adjustment_service.get_level_history(
        db, user_id, subject, limit
    )
    
    return {
        "level_history": history,
        "total_adjustments": len(history),
        "user_id": user_id,
        "subject_filter": subject.value if subject else "all",
        "requested_by": str(current_user.id)
    }


@router.get("/admin/level-analytics", response_model=LevelAnalyticsOverviewResponse)
async def get_level_analytics_overview(
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get comprehensive level analytics overview (Teacher/Admin only)"""
    
    # Get statistics for both subjects
    math_stats = await level_adjustment_service.get_level_statistics(db, Subject.MATH)
    english_stats = await level_adjustment_service.get_level_statistics(db, Subject.ENGLISH)
    
    # Get recent batch evaluations
    math_recommendations = await level_adjustment_service.batch_evaluate_adjustments(
        db, Subject.MATH, min_attempts=5
    )
    english_recommendations = await level_adjustment_service.batch_evaluate_adjustments(
        db, Subject.ENGLISH, min_attempts=5
    )
    
    return {
        "level_distribution": {
            "math": math_stats,
            "english": english_stats
        },
        "pending_recommendations": {
            "math": {
                "total": len(math_recommendations),
                "promotions": len([r for r in math_recommendations if r["recommended_level"] > r["current_level"]]),
                "demotions": len([r for r in math_recommendations if r["recommended_level"] < r["current_level"]])
            },
            "english": {
                "total": len(english_recommendations),
                "promotions": len([r for r in english_recommendations if r["recommended_level"] > r["current_level"]]),
                "demotions": len([r for r in english_recommendations if r["recommended_level"] < r["current_level"]])
            }
        },
        "recommendations_summary": {
            "math_recommendations": math_recommendations[:10],  # Top 10
            "english_recommendations": english_recommendations[:10]  # Top 10
        },
        "generated_at": datetime.utcnow().isoformat()
    }


# Spaced Repetition Endpoints
@router.get("/reviews/due", response_model=List[DueReview])
async def get_due_reviews(
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    limit: int = Query(20, ge=1, le=100, description="Number of reviews to return"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get questions due for review"""
    
    due_reviews = await spaced_repetition_service.get_due_reviews(
        db, str(current_user.id), subject, limit
    )
    
    return due_reviews


@router.post("/reviews/schedule", response_model=ReviewScheduleResult)
@idempotency_decorator(
    key_builder=lambda request, current_user: f"schedule_review:{current_user.id}:{request.question_id}:{request.quality}",
    config=IdempotencyConfig(scope="schedule_review", ttl_seconds=3600)
)
async def schedule_review(
    request: ScheduleReviewRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Schedule next review for a question"""
    
    result = await spaced_repetition_service.schedule_question_review(
        db, str(current_user.id), request.question_id, request.quality, request.response_time
    )
    
    return ReviewScheduleResult(**result)


@router.get("/reviews/statistics", response_model=ReviewStatistics)
async def get_review_statistics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get review statistics for the current user"""
    
    statistics = await spaced_repetition_service.get_review_statistics(
        db, str(current_user.id), days
    )
    
    return ReviewStatistics(**statistics)


@router.get("/reviews/calendar", response_model=ReviewCalendarResponse)
async def get_review_calendar(
    days_ahead: int = Query(30, ge=1, le=90, description="Number of days ahead to show"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get review calendar for upcoming days"""
    
    calendar = await spaced_repetition_service.get_review_calendar(
        db, str(current_user.id), days_ahead
    )
    
    return {
        "calendar": calendar,
        "total_days": len(calendar),
        "total_reviews": sum(len(reviews) for reviews in calendar.values()),
        "days_ahead": days_ahead
    }


@router.get("/reviews/progress", response_model=LearningProgress)
async def get_learning_progress(
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get learning progress based on spaced repetition data"""
    
    progress = await spaced_repetition_service.get_learning_progress(
        db, str(current_user.id), subject
    )
    
    return LearningProgress(**progress)


@router.post("/reviews/reset/{question_id}", response_model=ResetQuestionProgressResponse)
@idempotency_decorator(
    key_builder=lambda question_id, current_user: f"reset_review:{current_user.id}:{question_id}",
    config=IdempotencyConfig(scope="reset_review", ttl_seconds=3600)
)
async def reset_question_progress(
    question_id: str,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Reset spaced repetition progress for a question"""
    
    result = await spaced_repetition_service.reset_question_progress(
        db, str(current_user.id), question_id
    )
    
    return result


@router.post("/reviews/bulk-schedule", response_model=BulkReviewResponse)
@idempotency_decorator(
    key_builder=lambda request, current_user: f"bulk_schedule:{current_user.id}:{hash(frozenset(r.dict().items() for r in request.question_results))}",
    config=IdempotencyConfig(scope="bulk_schedule", ttl_seconds=3600)
)
async def bulk_schedule_reviews(
    request: BulkReviewRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Bulk schedule reviews for multiple questions"""
    
    result = await spaced_repetition_service.bulk_schedule_reviews(
        db, str(current_user.id), request.question_results
    )
    
    return BulkReviewResponse(**result)


@router.post("/reviews/process-answer/{question_id}", response_model=ProcessAnswerForReviewResponse)
@idempotency_decorator(
    key_builder=lambda question_id, is_correct, response_time, expected_time, current_user: f"process_answer:{current_user.id}:{question_id}:{is_correct}",
    config=IdempotencyConfig(scope="process_answer", ttl_seconds=3600)
)
async def process_answer_for_review(
    question_id: str,
    is_correct: bool = Query(..., description="Whether the answer was correct"),
    response_time: Optional[int] = Query(None, ge=0, description="Response time in seconds"),
    expected_time: int = Query(60, ge=1, description="Expected response time in seconds"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Process an answer and automatically schedule next review"""
    
    result = await spaced_repetition_service.process_answer_for_spaced_repetition(
        db, str(current_user.id), question_id, is_correct, response_time, expected_time
    )
    
    return result


# Admin/Teacher Spaced Repetition Endpoints
@router.get("/admin/reviews/user/{user_id}/statistics", response_model=UserReviewStatisticsAdminResponse)
async def get_user_review_statistics_admin(
    user_id: str,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get review statistics for any user (Teacher/Admin only)"""
    
    statistics = await spaced_repetition_service.get_review_statistics(
        db, user_id, days
    )
    
    return {
        "user_id": user_id,
        "statistics": statistics,
        "requested_by": str(current_user.id)
    }


@router.get("/admin/reviews/user/{user_id}/progress", response_model=UserLearningProgressAdminResponse)
async def get_user_learning_progress_admin(
    user_id: str,
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get learning progress for any user (Teacher/Admin only)"""
    
    progress = await spaced_repetition_service.get_learning_progress(
        db, user_id, subject
    )
    
    return {
        "user_id": user_id,
        "progress": progress,
        "requested_by": str(current_user.id)
    }


@router.post("/admin/reviews/user/{user_id}/reset/{question_id}", response_model=ResetUserQuestionProgressAdminResponse)
@idempotency_decorator(
    key_builder=lambda user_id, question_id, current_user: f"admin_reset_review:{user_id}:{question_id}",
    config=IdempotencyConfig(scope="admin_reset_review", ttl_seconds=3600)
)
async def reset_user_question_progress_admin(
    user_id: str,
    question_id: str,
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Reset spaced repetition progress for any user's question (Teacher/Admin only)"""
    
    result = await spaced_repetition_service.reset_question_progress(
        db, user_id, question_id
    )
    
    return {
        "user_id": user_id,
        "reset_result": result,
        "applied_by": str(current_user.id)
    }


# Performance-based Recommendations API
@router.get("/recommendations/performance/{subject}", response_model=PerformanceBasedRecommendationsResponse)
async def get_performance_based_recommendations(
    subject: Subject,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get performance-based recommendations for a subject"""
    
    # Get user performance metrics
    performance_metrics = await answer_evaluation_service.get_user_performance_metrics(
        db, str(current_user.id), subject, 30
    )
    
    # Get level adjustment recommendation
    level_recommendation = await level_adjustment_service.evaluate_level_adjustment(
        db, current_user, subject
    )
    
    # Get spaced repetition statistics
    sr_statistics = await spaced_repetition_service.get_review_statistics(
        db, str(current_user.id), 30
    )
    
    # Get learning progress
    learning_progress = await spaced_repetition_service.get_learning_progress(
        db, str(current_user.id), subject
    )
    
    # Generate recommendations based on performance
    recommendations = []
    
    # Level-based recommendations
    if level_recommendation:
        if level_recommendation.recommended_level > level_recommendation.current_level:
            recommendations.append({
                "type": "level_promotion",
                "priority": "high",
                "title": f"Ready for Level {level_recommendation.recommended_level}!",
                "description": f"Your performance shows you're ready to advance from level {level_recommendation.current_level} to {level_recommendation.recommended_level}",
                "action": "Consider taking more challenging questions",
                "confidence": level_recommendation.confidence
            })
        elif level_recommendation.recommended_level < level_recommendation.current_level:
            recommendations.append({
                "type": "level_review",
                "priority": "medium",
                "title": "Review Previous Concepts",
                "description": f"Consider reviewing level {level_recommendation.recommended_level} concepts to strengthen your foundation",
                "action": "Practice easier questions to build confidence",
                "confidence": level_recommendation.confidence
            })
    
    # Accuracy-based recommendations
    if performance_metrics.accuracy_rate < 70:
        recommendations.append({
            "type": "accuracy_improvement",
            "priority": "high",
            "title": "Focus on Accuracy",
            "description": f"Your accuracy rate is {performance_metrics.accuracy_rate:.1f}%. Let's work on getting more answers correct",
            "action": "Take your time and double-check answers",
            "confidence": 0.9
        })
    elif performance_metrics.accuracy_rate > 90:
        recommendations.append({
            "type": "challenge_increase",
            "priority": "medium",
            "title": "Ready for More Challenge",
            "description": f"Excellent accuracy of {performance_metrics.accuracy_rate:.1f}%! You might be ready for harder questions",
            "action": "Try questions from the next difficulty level",
            "confidence": 0.8
        })
    
    # Spaced repetition recommendations
    if sr_statistics["overdue"] > 5:
        recommendations.append({
            "type": "review_overdue",
            "priority": "high",
            "title": "Catch Up on Reviews",
            "description": f"You have {sr_statistics['overdue']} overdue reviews. Regular review helps retention",
            "action": "Complete overdue reviews first",
            "confidence": 1.0
        })
    elif sr_statistics["due_today"] > 0:
        recommendations.append({
            "type": "daily_review",
            "priority": "medium",
            "title": "Daily Review Available",
            "description": f"{sr_statistics['due_today']} questions are ready for review today",
            "action": "Complete today's reviews to maintain progress",
            "confidence": 0.9
        })
    
    # Learning progress recommendations
    if learning_progress["progress_percentage"] < 25:
        recommendations.append({
            "type": "learning_start",
            "priority": "medium",
            "title": "Build Your Foundation",
            "description": "You're just getting started! Focus on learning new concepts",
            "action": "Practice regularly and don't worry about speed yet",
            "confidence": 0.8
        })
    elif learning_progress["progress_percentage"] > 75:
        recommendations.append({
            "type": "mastery_focus",
            "priority": "low",
            "title": "Excellent Progress!",
            "description": f"You've mastered {learning_progress['progress_percentage']:.1f}% of the material",
            "action": "Focus on maintaining mastery and exploring advanced topics",
            "confidence": 0.9
        })
    
    # Streak-based recommendations
    if performance_metrics.current_streak >= 10:
        recommendations.append({
            "type": "streak_celebration",
            "priority": "low",
            "title": "Amazing Streak!",
            "description": f"You're on a {performance_metrics.current_streak}-question streak!",
            "action": "Keep up the excellent work",
            "confidence": 1.0
        })
    elif performance_metrics.current_streak == 0 and performance_metrics.total_attempts > 5:
        recommendations.append({
            "type": "streak_recovery",
            "priority": "medium",
            "title": "Let's Start a New Streak",
            "description": "Focus on getting the next few questions right to build momentum",
            "action": "Take your time with the next questions",
            "confidence": 0.7
        })
    
    # Sort recommendations by priority
    priority_order = {"high": 3, "medium": 2, "low": 1}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)
    
    return {
        "subject": subject.value,
        "user_id": str(current_user.id),
        "performance_summary": {
            "accuracy_rate": performance_metrics.accuracy_rate,
            "current_level": getattr(current_user, f"current_{subject.value}_level"),
            "total_attempts": performance_metrics.total_attempts,
            "current_streak": performance_metrics.current_streak,
            "progress_percentage": learning_progress["progress_percentage"],
            "overdue_reviews": sr_statistics["overdue"],
            "due_today": sr_statistics["due_today"]
        },
        "recommendations": recommendations,
        "total_recommendations": len(recommendations),
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/recommendations/study-plan/{subject}", response_model=StudyPlanRecommendationsResponse)
async def get_study_plan_recommendations(
    subject: Subject,
    days_ahead: int = Query(7, ge=1, le=30, description="Number of days to plan ahead"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get personalized study plan recommendations"""
    
    # Get performance data
    performance_metrics = await answer_evaluation_service.get_user_performance_metrics(
        db, str(current_user.id), subject, 30
    )
    
    # Get due reviews
    due_reviews = await spaced_repetition_service.get_due_reviews(
        db, str(current_user.id), subject, 50
    )
    
    # Get review calendar
    review_calendar = await spaced_repetition_service.get_review_calendar(
        db, str(current_user.id), days_ahead
    )
    
    # Generate study plan
    study_plan = []
    
    for day in range(days_ahead):
        date = datetime.utcnow() + timedelta(days=day)
        date_str = date.date().isoformat()
        
        # Get scheduled reviews for this day
        scheduled_reviews = review_calendar.get(date_str, [])
        
        # Calculate recommended study time
        base_time = 15  # Base 15 minutes
        review_time = len(scheduled_reviews) * 2  # 2 minutes per review
        new_questions_time = 10 if day % 2 == 0 else 0  # New questions every other day
        
        total_time = base_time + review_time + new_questions_time
        
        # Generate activities
        activities = []
        
        if scheduled_reviews:
            activities.append({
                "type": "spaced_repetition",
                "title": f"Review {len(scheduled_reviews)} questions",
                "duration_minutes": review_time,
                "priority": "high",
                "questions": len(scheduled_reviews)
            })
        
        if new_questions_time > 0:
            activities.append({
                "type": "new_learning",
                "title": "Learn new concepts",
                "duration_minutes": new_questions_time,
                "priority": "medium",
                "questions": 3
            })
        
        if performance_metrics.accuracy_rate < 80:
            activities.append({
                "type": "practice",
                "title": "Practice weak areas",
                "duration_minutes": 10,
                "priority": "medium",
                "questions": 5
            })
        
        study_plan.append({
            "date": date_str,
            "day_name": date.strftime("%A"),
            "total_time_minutes": total_time,
            "activities": activities,
            "scheduled_reviews": len(scheduled_reviews),
            "focus_area": "Review" if scheduled_reviews else "New Learning"
        })
    
    return {
        "subject": subject.value,
        "user_id": str(current_user.id),
        "study_plan": study_plan,
        "total_days": days_ahead,
        "average_daily_time": sum(day["total_time_minutes"] for day in study_plan) / len(study_plan),
        "total_scheduled_reviews": sum(len(review_calendar.get(day["date"], [])) for day in study_plan),
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/recommendations/adaptive/{subject}", response_model=AdaptiveRecommendationsResponse)
async def get_adaptive_recommendations(
    subject: Subject,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get adaptive recommendations based on real-time performance"""
    
    # Get recent performance (last 10 attempts)
    from sqlalchemy import select, desc
    from app.models.student_attempt import StudentAttempt
    from app.models.question import Question
    
    recent_attempts_query = select(StudentAttempt, Question).join(Question).where(
        and_(
            StudentAttempt.user_id == current_user.id,
            Question.subject == subject
        )
    ).order_by(desc(StudentAttempt.attempt_date)).limit(10)
    
    result = await db.execute(recent_attempts_query)
    recent_attempts = result.fetchall()
    
    if not recent_attempts:
        return {
            "subject": subject.value,
            "message": "No recent attempts found. Start practicing to get personalized recommendations!",
            "recommendations": []
        }
    
    # Analyze recent performance
    correct_count = sum(1 for attempt, _ in recent_attempts if attempt.is_correct)
    accuracy = (correct_count / len(recent_attempts)) * 100
    
    # Analyze difficulty distribution
    difficulties = [question.difficulty_level for _, question in recent_attempts]
    avg_difficulty = sum(difficulties) / len(difficulties)
    
    # Analyze time performance
    times = [attempt.time_spent for attempt, _ in recent_attempts if attempt.time_spent]
    avg_time = sum(times) / len(times) if times else 60
    
    # Generate adaptive recommendations
    recommendations = []
    
    # Difficulty adjustment recommendations
    if accuracy > 85 and avg_difficulty < 4:
        recommendations.append({
            "type": "increase_difficulty",
            "title": "Ready for Harder Questions",
            "description": f"You're getting {accuracy:.1f}% correct. Try difficulty level {min(5, int(avg_difficulty) + 1)}",
            "immediate_action": f"Practice {min(5, int(avg_difficulty) + 1)} difficulty questions",
            "confidence": 0.8
        })
    elif accuracy < 60 and avg_difficulty > 2:
        recommendations.append({
            "type": "decrease_difficulty",
            "title": "Build Confidence with Easier Questions",
            "description": f"Accuracy is {accuracy:.1f}%. Try easier questions to build confidence",
            "immediate_action": f"Practice level {max(1, int(avg_difficulty) - 1)} questions",
            "confidence": 0.9
        })
    
    # Speed recommendations
    if avg_time > 90:
        recommendations.append({
            "type": "speed_improvement",
            "title": "Work on Speed",
            "description": f"Average time is {avg_time:.0f} seconds. Try to answer more quickly",
            "immediate_action": "Set a 60-second timer for next questions",
            "confidence": 0.7
        })
    elif avg_time < 30 and accuracy < 80:
        recommendations.append({
            "type": "slow_down",
            "title": "Take Your Time",
            "description": f"You're fast ({avg_time:.0f}s) but accuracy is {accuracy:.1f}%. Slow down a bit",
            "immediate_action": "Double-check your answers before submitting",
            "confidence": 0.8
        })
    
    # Pattern-based recommendations
    recent_errors = [attempt.error_category for attempt, _ in recent_attempts 
                    if not attempt.is_correct and attempt.error_category]
    
    if recent_errors:
        from collections import Counter
        error_counts = Counter(recent_errors)
        most_common_error = error_counts.most_common(1)[0]
        
        recommendations.append({
            "type": "error_pattern",
            "title": f"Focus on {most_common_error[0]}",
            "description": f"You've made {most_common_error[1]} errors in {most_common_error[0]} recently",
            "immediate_action": f"Review {most_common_error[0]} concepts and practice similar questions",
            "confidence": 0.9
        })
    
    # Streak recommendations
    current_streak = 0
    for attempt, _ in recent_attempts:
        if attempt.is_correct:
            current_streak += 1
        else:
            break
    
    if current_streak >= 5:
        recommendations.append({
            "type": "streak_momentum",
            "title": f"Great Streak of {current_streak}!",
            "description": "You're on fire! Keep the momentum going",
            "immediate_action": "Try a slightly harder question to challenge yourself",
            "confidence": 1.0
        })
    
    return {
        "subject": subject.value,
        "user_id": str(current_user.id),
        "recent_performance": {
            "attempts_analyzed": len(recent_attempts),
            "accuracy": accuracy,
            "average_difficulty": avg_difficulty,
            "average_time_seconds": avg_time,
            "current_streak": current_streak
        },
        "recommendations": recommendations,
        "next_suggested_action": recommendations[0]["immediate_action"] if recommendations else "Keep practicing!",
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/health", response_model=AnswersHealthResponse)
async def answers_health():
    """Answer evaluation module health check"""
    return {"status": "ok", "module": "answers"}

# Progress Save Models
class QuizResult(BaseModel):
    userId: int = Field(..., description="User ID")
    subject: str = Field(..., description="Subject (math/english)")
    score: float = Field(..., ge=0, le=100, description="Score percentage")
    totalQuestions: int = Field(..., ge=1, description="Total questions in quiz")
    correctAnswers: int = Field(..., ge=0, description="Number of correct answers")
    timeSpent: int = Field(..., ge=0, description="Time spent in seconds")
    difficulty: str = Field(..., description="Difficulty level")
    timestamp: str = Field(..., description="ISO timestamp")

class ProgressSaveResponse(BaseModel):
    success: bool
    message: str
    progress_id: Optional[str] = None
    saved_at: str

@router.post("/progress/save", response_model=ProgressSaveResponse)
async def save_progress(
    progress: QuizResult,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Save quiz progress to database"""
    
    try:
        # Validate subject
        if progress.subject.lower() not in ['math', 'english']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid subject. Must be 'math' or 'english'"
            )
        
        # Convert subject to enum
        subject_enum = Subject.MATH if progress.subject.lower() == 'math' else Subject.ENGLISH
        
        # Create progress record
        from app.models.student_progress import StudentProgress
        
        progress_record = StudentProgress(
            student_id=str(current_user.id),
            subject=subject_enum,
            score=progress.score,
            total_questions=progress.totalQuestions,
            correct_answers=progress.correctAnswers,
            time_spent_seconds=progress.timeSpent,
            difficulty_level=progress.difficulty,
            completed_at=datetime.fromisoformat(progress.timestamp.replace('Z', '+00:00')),
            created_at=datetime.utcnow()
        )
        
        # Save to database
        db.add(progress_record)
        await db.commit()
        await db.refresh(progress_record)
        
        # Log the progress save
        logger = structlog.get_logger()
        logger.info(
            "Progress saved successfully",
            user_id=str(current_user.id),
            subject=progress.subject,
            score=progress.score,
            progress_id=str(progress_record.id)
        )
        
        return ProgressSaveResponse(
            success=True,
            message="Progress saved successfully",
            progress_id=str(progress_record.id),
            saved_at=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger = structlog.get_logger()
        logger.error(
            "Failed to save progress",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save progress: {str(e)}"
        )