from fastapi import APIRouter, Depends, HTTPException, status, Query, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging
import time
from datetime import datetime, timedelta

from app.core.database import get_async_session
from app.middleware.auth import get_current_student, get_current_teacher
from app.models.user import User
from app.models.question import Subject, QuestionType
from app.services.math_selector import math_selector
from app.services.math_profile_manager import math_profile_manager
from app.services.advanced_math_algorithms import advanced_math_algorithms
from app.services.math_analytics_service import math_analytics_service
from app.services.math_quality_assurance import math_quality_assurance
from app.services.math_performance_monitoring import math_performance_monitoring
from app.services.math_personalization import math_personalization, PersonalizationContext
from app.services.math_advanced_retrieval import math_advanced_retrieval, RetrievalQuery
from app.services.math_ab_testing import math_ab_testing
from app.services.mcp_service import mcp_service
from app.core.cache import cache_service
from app.services.rag_validation_service import rag_validation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/math/rag", tags=["math-rag"])


# Request/Response Models
class MathQuestionRequest(BaseModel):
    exclude_question_ids: Optional[List[str]] = Field(None, description="Dışlanacak soru ID'leri")
    topic_category: Optional[str] = Field(None, description="Belirli konu kategorisi")
    force_recovery: Optional[bool] = Field(False, description="Kurtarma modunu zorla")
    force_srs: Optional[bool] = Field(False, description="SRS modunu zorla")


class MathQuestionResponse(BaseModel):
    question: Dict[str, Any]
    selection_rationale: Dict[str, Any]
    profile_stats: Dict[str, Any]
    learning_path: Dict[str, Any]
    latency_ms: int


class MathAnswerSubmission(BaseModel):
    question_id: str = Field(..., description="Soru ID'si")
    student_answer: str = Field(..., description="Öğrenci cevabı")
    time_taken: float = Field(..., description="Harcanan süre (saniye)")
    partial_credit: Optional[float] = Field(None, description="Kısmi puan (0.0-1.0)")


class MathAnswerResponse(BaseModel):
    is_correct: bool
    score: float
    feedback: str
    explanation: str
    updated_profile: Dict[str, Any]
    next_recommendations: List[Dict[str, Any]]
    latency_ms: int


class MathProfileResponse(BaseModel):
    profile: Dict[str, Any]
    statistics: Dict[str, Any]
    recommendations: Dict[str, Any]
    learning_path: Dict[str, Any]


# ---------- New RAG Endpoints Schemas ----------
class GenerateRequest(BaseModel):
    topic: Optional[str] = Field(default=None, description="Konu/alt başlık (ör. 'quadratic equations')")
    difficulty_level: Optional[int] = Field(default=1, ge=1, le=10)
    question_type: QuestionType = Field(default=QuestionType.MULTIPLE_CHOICE)
    n: int = Field(default=1, ge=1, le=10)


class GeneratedQuestion(BaseModel):
    kind: str
    stem: str
    options: Optional[List[str]] = None
    answer: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None


class GenerateResponse(BaseModel):
    items: List[GeneratedQuestion]
    usage: Dict[str, Any]


class SolveRequest(BaseModel):
    problem: str
    show_steps: bool = True


class SolveResponse(BaseModel):
    solution: str
    steps: Optional[str] = None
    usage: Dict[str, Any]


class CheckRequest(BaseModel):
    question: str
    user_answer: str
    answer_key: Optional[str] = None  # MCQ anahtarı varsa
    require_explanation: bool = True


class CheckResponse(BaseModel):
    correct: bool
    explanation: Optional[str] = None
    usage: Dict[str, Any]


@router.post("/next-question", response_model=MathQuestionResponse)
async def get_next_math_question(
    request: MathQuestionRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Öğrenci için bir sonraki matematik sorusunu seç"""
    
    start_time = time.time()
    
    try:
        # Öğrenci profilini al veya oluştur
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Seçim modunu belirle
        if request.force_recovery:
            # Kurtarma modunu zorla
            selected_question, rationale = await math_selector._pick_recovery_question(
                db, profile, request.exclude_question_ids
            )
            rationale["mode"] = "forced_recovery"
        elif request.force_srs:
            # SRS modunu zorla
            selected_question, rationale = await math_selector._pick_srs_question(
                db, profile, request.exclude_question_ids
            )
            rationale["mode"] = "forced_srs"
        else:
            # Normal seçim
            selected_question, rationale = await math_selector.select_question(
                db, profile, request.exclude_question_ids
            )
        
        # Kalite kontrolü
        # Son soruları al
        recent_questions = await math_profile_manager.get_recent_questions(profile)
        quality_result = math_quality_assurance.validate_question_quality(
            selected_question, profile, recent_questions
        )
        
        # Performans izleme
        await math_performance_monitoring.track_question_selection(
            str(current_user.id), selected_question, profile, rationale, latency_ms
        )
        
        # Soru verilerini hazırla
        question_data = {
            "id": str(selected_question.id),
            "content": selected_question.content,
            "question_type": selected_question.question_type.value,
            "difficulty_level": selected_question.difficulty_level,
            "estimated_difficulty": selected_question.estimated_difficulty,
            "topic_category": selected_question.topic_category,
            "correct_answer": selected_question.correct_answer,
            "options": selected_question.options,
            "source_type": selected_question.source_type.value,
            "created_at": selected_question.created_at.isoformat(),
        }
        
        # Profil istatistiklerini al
        profile_stats = await math_profile_manager.get_profile_statistics(profile)
        
        # Öğrenme yolunu al
        learning_path = await math_profile_manager.get_learning_path(profile)
        
        # Latency hesapla
        latency_ms = int((time.time() - start_time) * 1000)
        
        return MathQuestionResponse(
            question=question_data,
            selection_rationale=rationale,
            profile_stats=profile_stats,
            learning_path=learning_path,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"❌ Error in math question selection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question selection failed: {str(e)}"
        )


@router.post("/submit-answer", response_model=MathAnswerResponse)
async def submit_math_answer(
    submission: MathAnswerSubmission,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Matematik sorusu cevabını gönder ve değerlendir"""
    
    start_time = time.time()
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Soruyu veritabanından al
        from app.models.question import Question
        from sqlalchemy import select
        
        stmt = select(Question).where(Question.id == submission.question_id)
        result = await db.execute(stmt)
        question = result.scalar_one_or_none()
        
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found"
            )
        
        # Cevabı değerlendir (MCP math evaluator ile)
        evaluation_result = await mcp_service.evaluate_math_answer(
            question_content=question.content,
            correct_answer=question.correct_answer or "",
            student_answer=submission.student_answer,
            question_type=question.question_type.value,
            difficulty_level=question.difficulty_level,
            partial_credit=True
        )
        
        # Değerlendirme sonuçlarını al
        is_correct = evaluation_result.get("is_correct", False)
        score = evaluation_result.get("score", 0.0)
        
        # Kalite kontrolü
        answer_quality_result = math_quality_assurance.validate_answer_quality(
            submission.student_answer, question, submission.time_taken, {}
        )
        
        # Kısmi puan hesaplama
        partial_credit = math_quality_assurance.calculate_partial_credit(
            submission.student_answer, question.correct_answer or "", question, submission.time_taken
        )
        
        # Performans izleme
        await math_performance_monitoring.track_answer_submission(
            str(current_user.id), question, profile, is_correct, submission.time_taken, partial_credit
        )
        feedback = evaluation_result.get("feedback", "No feedback available")
        explanation = evaluation_result.get("explanation", "No explanation available")
        
        # Kısmi puan varsa kullan
        if submission.partial_credit is not None:
            score = submission.partial_credit
            is_correct = submission.partial_credit >= 0.5
        
        # Profili güncelle
        question_difficulty = question.estimated_difficulty or question.difficulty_level
        updated_profile = await math_profile_manager.update_profile_after_answer(
            db=db,
            profile=profile,
            question_difficulty=question_difficulty,
            is_correct=is_correct,
            time_taken=submission.time_taken,
            partial_credit=submission.partial_credit
        )
        
        # Önerileri al
        recommendations = await math_profile_manager.get_profile_statistics(updated_profile)
        next_recommendations = recommendations.get("recommendations", {}).get("recommendations", [])
        
        # Latency hesapla
        latency_ms = int((time.time() - start_time) * 1000)
        
        return MathAnswerResponse(
            is_correct=is_correct,
            score=score,
            feedback=feedback,
            explanation=explanation,
            updated_profile=updated_profile.to_dict(),
            next_recommendations=next_recommendations,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"❌ Error in math answer submission: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer submission failed: {str(e)}"
        )


@router.get("/profile", response_model=MathProfileResponse)
async def get_math_profile(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Öğrencinin matematik profilini al"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Profil istatistiklerini al
        statistics = await math_profile_manager.get_profile_statistics(profile)
        
        # Öğrenme yolunu al
        learning_path = await math_profile_manager.get_learning_path(profile)
        
        return MathProfileResponse(
            profile=profile.to_dict(),
            statistics=statistics,
            recommendations=statistics.get("recommendations", {}),
            learning_path=learning_path
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting math profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get profile: {str(e)}"
        )


@router.post("/reset-profile")
async def reset_math_profile(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Öğrencinin matematik profilini sıfırla (test amaçlı)"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Profili sıfırla
        reset_profile = await math_profile_manager.reset_profile(db, profile)
        
        return {
            "message": "Math profile reset successfully",
            "profile": reset_profile.to_dict()
        }
        
    except Exception as e:
        logger.error(f"❌ Error resetting math profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset profile: {str(e)}"
        )


@router.get("/selection-stats")
async def get_selection_statistics(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Seçim istatistiklerini al"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Seçim istatistiklerini al
        selection_stats = math_selector.get_selection_statistics(profile)
        
        return {
            "user_id": str(current_user.id),
            "selection_statistics": selection_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting selection statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get selection statistics: {str(e)}"
        )


@router.get("/health")
async def math_rag_health():
    """Matematik RAG sistemi health check"""
    
    return {
        "status": "healthy",
        "service": "math-rag",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "adaptive_difficulty",
            "thompson_sampling",
            "srs_spaced_repetition",
            "recovery_mode",
            "profile_management",
            "advanced_algorithms",
            "analytics_service",
            "mcp_integration",
            "quality_assurance",
            "performance_monitoring",
            "real_time_alerts",
            "system_health_monitoring",
            "personalization",
            "advanced_retrieval",
            "ab_testing"
        ]
    }


@router.get("/analytics/learning-progress")
async def get_learning_progress(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Öğrenme ilerlemesi analizi"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Performans geçmişini al
        performance_history = await math_profile_manager.get_performance_history(db, current_user.id)
        
        # Analitik servisi ile analiz
        analysis_result = math_analytics_service.analyze_learning_progress(profile, performance_history)
        
        return {
            "user_id": str(current_user.id),
            "analysis": analysis_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in learning progress analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Learning progress analysis failed: {str(e)}"
        )


@router.get("/analytics/algorithm-performance")
async def get_algorithm_performance(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Algoritma performansı analizi"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Seçim geçmişini al (mock data)
        selection_history = []
        outcomes = profile.last_k_outcomes or []
        for i, is_correct in enumerate(outcomes[-10:]):
            selection_history.append({
                "mode": "normal_thompson" if i > 5 else "normal_epsilon_greedy",
                "difficulty_match": 0.7 + (i * 0.02),
                "selection_accuracy": 0.8 if is_correct else 0.3,
                "timestamp": datetime.utcnow() - timedelta(hours=i)
            })
        
        # Analitik servisi ile analiz
        analysis_result = math_analytics_service.analyze_algorithm_performance(profile, selection_history)
        
        return {
            "user_id": str(current_user.id),
            "analysis": analysis_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in algorithm performance analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Algorithm performance analysis failed: {str(e)}"
        )


@router.get("/analytics/performance-prediction")
async def get_performance_prediction(
    prediction_horizon: int = Query(30, description="Prediction horizon in days"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Performans tahmini"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Performans geçmişini al (mock data)
        performance_history = []
        outcomes = profile.last_k_outcomes or []
        for i, is_correct in enumerate(outcomes[-20:]):
            performance_history.append({
                "accuracy": 1.0 if is_correct else 0.0,
                "speed": 0.8 - (i * 0.02),
                "difficulty": profile.global_skill + (i * 0.1),
                "timestamp": datetime.utcnow() - timedelta(hours=i)
            })
        
        # Analitik servisi ile tahmin
        prediction_result = math_analytics_service.predict_performance_trajectory(
            profile, performance_history, prediction_horizon
        )
        
        return {
            "user_id": str(current_user.id),
            "prediction": prediction_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in performance prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance prediction failed: {str(e)}"
        )


@router.post("/recommendations/adaptive")
async def get_adaptive_recommendations(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Adaptif öneriler"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Performans geçmişini al (mock data)
        performance_history = []
        outcomes = profile.last_k_outcomes or []
        for i, is_correct in enumerate(outcomes[-20:]):
            performance_history.append({
                "accuracy": 1.0 if is_correct else 0.0,
                "speed": 0.8 - (i * 0.02),
                "difficulty": profile.global_skill + (i * 0.1),
                "timestamp": datetime.utcnow() - timedelta(hours=i)
            })
        
        # Mevcut context
        current_context = {
            "learning_session": "active",
            "time_of_day": datetime.utcnow().hour,
            "recent_activity": len(outcomes),
            "current_difficulty": profile.global_skill * profile.difficulty_factor
        }
        
        # Analitik servisi ile öneriler
        recommendations_result = math_analytics_service.generate_adaptive_recommendations(
            profile, performance_history, current_context
        )
        
        return {
            "user_id": str(current_user.id),
            "recommendations": recommendations_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in adaptive recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Adaptive recommendations failed: {str(e)}"
        )


@router.get("/monitoring/performance-metrics")
async def get_performance_metrics(
    metric_name: Optional[str] = Query(None, description="Specific metric name"),
    time_window_hours: int = Query(24, description="Time window in hours"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Performans metriklerini al"""
    
    try:
        time_window = timedelta(hours=time_window_hours)
        
        metrics = math_performance_monitoring.get_performance_metrics(
            metric_name=metric_name,
            user_id=str(current_user.id),
            time_window=time_window
        )
        
        return {
            "user_id": str(current_user.id),
            "metrics": metrics,
            "time_window_hours": time_window_hours,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance metrics failed: {str(e)}"
        )


@router.get("/monitoring/alerts")
async def get_alerts(
    level: Optional[str] = Query(None, description="Alert level (info, warning, error, critical)"),
    time_window_hours: int = Query(24, description="Time window in hours"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Uyarıları al"""
    
    try:
        time_window = timedelta(hours=time_window_hours)
        
        alerts = math_performance_monitoring.get_alerts(
            level=level,
            time_window=time_window
        )
        
        return {
            "user_id": str(current_user.id),
            "alerts": alerts,
            "level_filter": level,
            "time_window_hours": time_window_hours,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alerts failed: {str(e)}"
        )


@router.get("/monitoring/system-health")
async def get_system_health(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Sistem sağlığını kontrol et"""
    
    try:
        health = math_performance_monitoring.get_system_health()
        
        return {
            "user_id": str(current_user.id),
            "system_health": health,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in system health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"System health check failed: {str(e)}"
        )


@router.get("/quality/question-validation")
async def validate_question_quality(
    question_id: str = Query(..., description="Question ID to validate"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Soru kalitesini doğrula"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Soruyu veritabanından al
        from app.models.question import Question
        from sqlalchemy import select
        
        stmt = select(Question).where(Question.id == question_id)
        result = await db.execute(stmt)
        question = result.scalar_one_or_none()
        
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found"
            )
        
        # Son soruları al (mock data)
        recent_questions = []
        
        # Kalite kontrolü
        quality_result = math_quality_assurance.validate_question_quality(
            question, profile, recent_questions
        )
        
        return {
            "user_id": str(current_user.id),
            "question_id": question_id,
            "quality_check": {
                "passed": quality_result.passed,
                "score": quality_result.score,
                "issues": quality_result.issues,
                "warnings": quality_result.warnings,
                "recommendations": quality_result.recommendations
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in question quality validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question quality validation failed: {str(e)}"
        )


@router.post("/quality/session-validation")
async def validate_session_quality(
    session_data: Dict[str, Any],
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Oturum kalitesini doğrula"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Oturum kalite kontrolü
        session_result = math_quality_assurance.validate_user_session(profile, session_data)
        
        return {
            "user_id": str(current_user.id),
            "session_validation": {
                "passed": session_result.passed,
                "score": session_result.score,
                "issues": session_result.issues,
                "warnings": session_result.warnings,
                "recommendations": session_result.recommendations
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in session quality validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session quality validation failed: {str(e)}"
        )


# ============================================================================
# PERSONALIZATION ENDPOINTS
# ============================================================================

@router.post("/personalization/learn-preferences")
async def learn_user_preferences(
    question_id: str = Form(...),
    user_answer: str = Form(...),
    is_correct: bool = Form(...),
    response_time: float = Form(...),
    time_of_day: int = Form(...),
    day_of_week: int = Form(...),
    session_duration: int = Form(...),
    current_mood: Optional[str] = Form(None),
    learning_goals: Optional[List[str]] = Form(None),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Kullanıcı tercihlerini öğren"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Soruyu veritabanından al
        from app.models.question import Question
        from sqlalchemy import select
        
        stmt = select(Question).where(Question.id == question_id)
        result = await db.execute(stmt)
        question = result.scalar_one_or_none()
        
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found"
            )
        
        # Personalization context oluştur
        context = PersonalizationContext(
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            session_duration=session_duration,
            recent_performance=[1.0 if is_correct else 0.0],  # Basit performans
            current_mood=current_mood,
            learning_goals=learning_goals or []
        )
        
        # Tercihleri öğren
        learning_result = await math_personalization.learn_user_preferences(
            str(current_user.id), profile, question, user_answer, is_correct, response_time, context
        )
        
        return {
            "user_id": str(current_user.id),
            "learning_result": learning_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in learning preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Learning preferences failed: {str(e)}"
        )


@router.post("/personalization/personalize-selection")
async def personalize_question_selection(
    candidate_questions: List[str] = Form(...),
    target_difficulty: float = Form(...),
    topic_category: Optional[str] = Form(None),
    question_type: Optional[str] = Form(None),
    time_of_day: int = Form(...),
    session_duration: int = Form(...),
    current_mood: Optional[str] = Form(None),
    learning_goals: Optional[List[str]] = Form(None),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Soru seçimini kişiselleştir"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Aday soruları veritabanından al
        from app.models.question import Question
        from sqlalchemy import select
        
        stmt = select(Question).where(Question.id.in_(candidate_questions))
        result = await db.execute(stmt)
        questions = result.scalars().all()
        
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No candidate questions found"
            )
        
        # Gerçek performans verisi ile doldur
        recent_performance = await math_profile_manager.get_recent_performance(profile)
        context = PersonalizationContext(
            time_of_day=time_of_day,
            day_of_week=datetime.utcnow().weekday(),
            session_duration=session_duration,
            recent_performance=recent_performance,
            current_mood=current_mood,
            learning_goals=learning_goals or []
        )
        
        # Kişiselleştirilmiş seçim
        personalized_results = await math_personalization.personalize_question_selection(
            str(current_user.id), profile, questions, context
        )
        
        # Sonuçları formatla
        formatted_results = []
        for question, score in personalized_results:
            formatted_results.append({
                "question_id": str(question.id),
                "content": question.content,
                "difficulty": question.estimated_difficulty or question.difficulty_level,
                "topic_category": question.topic_category,
                "question_type": question.question_type.value,
                "personalization_score": score
            })
        
        return {
            "user_id": str(current_user.id),
            "personalized_questions": formatted_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in personalized selection for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Personalized selection failed for user {current_user.id}: {str(e)}"
        )


@router.post("/personalization/adapt-difficulty")
async def adapt_difficulty(
    current_difficulty: float = Form(...),
    recent_performance: List[float] = Form(...),
    time_of_day: int = Form(...),
    session_duration: int = Form(...),
    current_mood: Optional[str] = Form(None),
    learning_goals: Optional[List[str]] = Form(None),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Zorluğu kişiselleştirilmiş şekilde adapte et"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # recent_performance None ise boş listeye çevir
        rp = recent_performance if recent_performance is not None else []
        context = PersonalizationContext(
            time_of_day=time_of_day,
            day_of_week=datetime.utcnow().weekday(),
            session_duration=session_duration,
            recent_performance=rp,
            current_mood=current_mood,
            learning_goals=learning_goals or []
        )
        
        # Zorluk adaptasyonu
        new_difficulty = await math_personalization.adapt_difficulty(
            str(current_user.id), profile, current_difficulty, recent_performance, context
        )
        
        return {
            "user_id": str(current_user.id),
            "current_difficulty": current_difficulty,
            "adapted_difficulty": new_difficulty,
            "adaptation_delta": new_difficulty - current_difficulty,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in difficulty adaptation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Difficulty adaptation failed: {str(e)}"
        )


@router.post("/personalization/generate-feedback")
async def generate_personalized_feedback(
    question_id: str = Form(...),
    user_answer: str = Form(...),
    is_correct: bool = Form(...),
    time_of_day: int = Form(...),
    session_duration: int = Form(...),
    current_mood: Optional[str] = Form(None),
    learning_goals: Optional[List[str]] = Form(None),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Kişiselleştirilmiş feedback oluştur"""
    
    try:
        # Soruyu veritabanından al
        from app.models.question import Question
        from sqlalchemy import select
        
        stmt = select(Question).where(Question.id == question_id)
        result = await db.execute(stmt)
        question = result.scalar_one_or_none()
        
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found"
            )
        
        # Personalization context oluştur
        context = PersonalizationContext(
            time_of_day=time_of_day,
            day_of_week=datetime.utcnow().weekday(),
            session_duration=session_duration,
            recent_performance=[1.0 if is_correct else 0.0],
            current_mood=current_mood,
            learning_goals=learning_goals or []
        )
        
        # Kişiselleştirilmiş feedback
        feedback_result = await math_personalization.generate_personalized_feedback(
            str(current_user.id), question, user_answer, is_correct, context
        )
        
        return {
            "user_id": str(current_user.id),
            "question_id": question_id,
            "feedback": feedback_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in personalized feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Personalized feedback failed: {str(e)}"
        )


@router.get("/personalization/learning-recommendations")
async def get_learning_recommendations(
    time_of_day: int = Query(...),
    session_duration: int = Query(...),
    current_mood: Optional[str] = Query(None),
    learning_goals: Optional[List[str]] = Query(None),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Kişiselleştirilmiş öğrenme önerileri al"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Gerçek performans verisi ile doldur
        recent_performance = await math_profile_manager.get_recent_performance(profile)
        context = PersonalizationContext(
            time_of_day=time_of_day,
            day_of_week=datetime.utcnow().weekday(),
            session_duration=session_duration,
            recent_performance=recent_performance,
            current_mood=current_mood,
            learning_goals=learning_goals or []
        )
        
        # Öğrenme önerileri
        recommendations = await math_personalization.get_learning_recommendations(
            str(current_user.id), profile, context
        )
        
        return {
            "user_id": str(current_user.id),
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in learning recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Learning recommendations failed: {str(e)}"
        )


# ============================================================================
# ADVANCED RETRIEVAL ENDPOINTS
# ============================================================================

@router.post("/retrieval/advanced-search")
async def advanced_retrieve_questions(
    target_difficulty: float = Form(...),
    topic_category: Optional[str] = Form(None),
    question_type: Optional[str] = Form(None),
    exclude_question_ids: Optional[List[str]] = Form(None),
    max_results: int = Form(10),
    time_of_day: int = Form(...),
    session_duration: int = Form(...),
    current_mood: Optional[str] = Form(None),
    learning_goals: Optional[List[str]] = Form(None),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Gelişmiş soru retrieval"""
    
    try:
        # Öğrenci profilini al
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Tüm soruları al (gerçek implementasyonda filtreleme yapılacak)
        from app.models.question import Question
        from sqlalchemy import select
        
        stmt = select(Question).limit(100)  # Limit for demo
        result = await db.execute(stmt)
        candidate_questions = result.scalars().all()
        
        if not candidate_questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No questions available"
            )
        
        # Context oluştur
        context = {
            "recent_performance": await self._get_recent_performance(current_user, db),
            "preferences": await self._get_user_preferences(current_user, db),
            "learning_goals": learning_goals or []
        }
        
        # Retrieval query oluştur
        query = RetrievalQuery(
            user_id=str(current_user.id),
            profile=profile,
            target_difficulty=target_difficulty,
            topic_category=topic_category,
            question_type=question_type,
            exclude_question_ids=exclude_question_ids,
            context=context
        )
        
        # Gelişmiş retrieval
        retrieval_results = await math_advanced_retrieval.advanced_retrieve_questions(
            query, candidate_questions, max_results
        )
        
        # Sonuçları formatla
        formatted_results = []
        for result in retrieval_results:
            formatted_results.append({
                "question_id": str(result.question.id),
                "content": result.question.content,
                "difficulty": result.question.estimated_difficulty or result.question.difficulty_level,
                "topic_category": result.question.topic_category,
                "question_type": result.question.question_type.value,
                "relevance_score": result.relevance_score,
                "diversity_score": result.diversity_score,
                "freshness_score": result.freshness_score,
                "difficulty_match": result.difficulty_match,
                "overall_score": result.overall_score,
                "retrieval_method": result.retrieval_method
            })
        
        return {
            "user_id": str(current_user.id),
            "retrieval_results": formatted_results,
            "total_results": len(formatted_results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in advanced retrieval: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced retrieval failed: {str(e)}"
        )

    async def _get_recent_performance(self, user: User, db: AsyncSession) -> List[float]:
        """Get user's recent performance data"""
        try:
            from app.models.student_attempt import StudentAttempt
            from app.models.question import Question, Subject
            
            # Get last 10 math attempts
            query = select(StudentAttempt).join(Question).where(
                and_(
                    StudentAttempt.user_id == user.id,
                    Question.subject == Subject.MATH
                )
            ).order_by(desc(StudentAttempt.attempt_date)).limit(10)
            
            result = await db.execute(query)
            attempts = result.scalars().all()
            
            # Calculate performance scores
            performance = []
            for attempt in attempts:
                if attempt.is_correct:
                    performance.append(1.0)
                else:
                    performance.append(0.0)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting recent performance: {e}")
            return []

    async def _get_user_preferences(self, user: User, db: AsyncSession) -> Dict[str, Any]:
        """Get user's learning preferences"""
        try:
            return {
                "learning_style": user.learning_style.value,
                "preferred_difficulty": user.current_math_level,
                "preferred_topics": [],  # Could be extended with user preferences
                "session_duration": 30,  # Default 30 minutes
                "time_of_day": datetime.utcnow().hour
            }
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}


@router.post("/retrieval/query-expansion")
async def expand_query(
    base_query: str = Form(...),
    topic_category: Optional[str] = Form(None),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Query expansion"""
    
    try:
        # Query expansion
        expanded_terms = await math_advanced_retrieval.query_expansion(base_query, topic_category)
        
        return {
            "user_id": str(current_user.id),
            "base_query": base_query,
            "expanded_terms": expanded_terms,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in query expansion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query expansion failed: {str(e)}"
        )


# ============================================================================
# A/B TESTING ENDPOINTS
# ============================================================================

@router.post("/ab-testing/create-experiment")
async def create_experiment(
    name: str = Form(...),
    description: str = Form(...),
    variants: List[Dict[str, Any]] = Form(...),
    target_metrics: List[str] = Form(...),
    statistical_test: str = Form("t_test"),
    significance_level: float = Form(0.05),
    minimum_sample_size: int = Form(100),
    duration_days: int = Form(14),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Yeni A/B test deneyi oluştur"""
    
    try:
        # Deney oluştur
        experiment = await math_ab_testing.create_experiment(
            name=name,
            description=description,
            variants=variants,
            target_metrics=target_metrics,
            statistical_test=statistical_test,
            significance_level=significance_level,
            minimum_sample_size=minimum_sample_size,
            duration_days=duration_days
        )
        
        return {
            "user_id": str(current_user.id),
            "experiment": {
                "id": experiment.id,
                "name": experiment.name,
                "description": experiment.description,
                "status": experiment.status.value,
                "variants": [{"name": v.name, "description": v.description} for v in experiment.variants],
                "target_metrics": experiment.target_metrics,
                "created_at": experiment.created_at.isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in creating experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Creating experiment failed: {str(e)}"
        )


@router.post("/ab-testing/assign-variant")
async def assign_variant(
    experiment_id: str = Form(...),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Kullanıcıya varyant ata"""
    
    try:
        # Varyant atama
        variant_name = await math_ab_testing.assign_variant(experiment_id, str(current_user.id))
        
        if not variant_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to assign variant"
            )
        
        return {
            "user_id": str(current_user.id),
            "experiment_id": experiment_id,
            "assigned_variant": variant_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in variant assignment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Variant assignment failed: {str(e)}"
        )


@router.post("/ab-testing/record-event")
async def record_experiment_event(
    experiment_id: str = Form(...),
    variant_name: str = Form(...),
    event_type: str = Form(...),
    event_data: Dict[str, Any] = Form(...),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Deney olayını kaydet"""
    
    try:
        # Olay kaydetme
        success = await math_ab_testing.record_experiment_event(
            experiment_id, str(current_user.id), variant_name, event_type, event_data
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to record event"
            )
        
        return {
            "user_id": str(current_user.id),
            "experiment_id": experiment_id,
            "variant_name": variant_name,
            "event_type": event_type,
            "recorded": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in recording event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recording event failed: {str(e)}"
        )


@router.get("/ab-testing/experiment-status/{experiment_id}")
async def get_experiment_status(
    experiment_id: str,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Deney durumunu al"""
    
    try:
        # Deney durumu
        status = await math_ab_testing.get_experiment_status(experiment_id)
        
        if "error" in status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=status["error"]
            )
        
        return {
            "user_id": str(current_user.id),
            "experiment_status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in getting experiment status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Getting experiment status failed: {str(e)}"
        )


@router.post("/ab-testing/analyze-experiment")
async def analyze_experiment(
    experiment_id: str = Form(...),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Deneyi analiz et"""
    
    try:
        # Deney analizi
        analysis = await math_ab_testing.analyze_experiment(experiment_id)
        
        if "error" in analysis:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=analysis["error"]
            )
        
        return {
            "user_id": str(current_user.id),
            "experiment_analysis": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in analyzing experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analyzing experiment failed: {str(e)}"
        )


@router.get("/ab-testing/list-experiments")
async def list_experiments(
    status_filter: Optional[str] = Query(None),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Deneyleri listele"""
    
    try:
        # Deney listesi
        experiments = await math_ab_testing.list_experiments(status_filter)
        
        return {
            "user_id": str(current_user.id),
            "experiments": experiments,
            "total_count": len(experiments),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in listing experiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Listing experiments failed: {str(e)}"
        )


# ---------- New RAG Endpoints ----------

@router.get("/health")
async def health():
    return {"status": "ok", "module": "math-rag"}


@router.post("/generate", response_model=GenerateResponse, status_code=status.HTTP_200_OK)
async def generate_questions(
    body: GenerateRequest,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_teacher),  # öğretmen yetkisi varsayıldı; istersen öğrenciye de açabiliriz
):
    """
    Math sorusu üretir (LLM'yi JSON şemasıyla çağırır).
    """
    # LLM'ye verilecek çok basit JSON şeması — provider tarafında 'valid JSON only' modu var.
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "stem": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "answer": {"type": "object"},
                        "meta": {"type": "object"},
                    },
                    "required": ["kind", "stem"]
                }
            },
            "usage": {"type": "object"}
        },
        "required": ["items"]
    }

    sys_prompt = (
        "You are a math item writer. Produce exam-quality questions strictly as JSON.\n"
        f"Question type: {body.question_type.value}. Difficulty: {body.difficulty_level or 1} / 10."
        + (f" Topic: {body.topic}." if body.topic else "")
    )

    user_prompt = (
        f"Generate {body.n} {body.question_type.value} math question(s). "
        "Return JSON matching the given schema only."
    )

    try:
        from app.services.llm_gateway import llm_gateway
        
        result = await llm_gateway.generate_json(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            schema=schema,
            temperature=0.2,
            max_tokens=900,
        )
        if not result.success:
            raise HTTPException(status_code=502, detail=f"LLM error: {result.error}")

        data = result.parsed_json
        usage = getattr(result, "usage", {}) or {}
        # koruma: items yoksa boş liste
        items = data.get("items", [])
        return GenerateResponse(items=items, usage=usage)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("math-rag generate failed: %s", e)
        raise HTTPException(status_code=500, detail="math-rag generate failed")


@router.post("/solve", response_model=SolveResponse, status_code=status.HTTP_200_OK)
async def solve_problem(
    body: SolveRequest,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_student),
):
    """
    Serbest biçimli bir matematik problemi için çözüm/izahat üretir.
    """
    schema = {
        "type": "object",
        "properties": {
            "solution": {"type": "string"},
            "steps": {"type": "string"}
        },
        "required": ["solution"]
    }
    sys_prompt = "You are a rigorous math solver. Return only JSON."
    user_prompt = (
        "Solve the following problem. "
        + ("Show steps as well." if body.show_steps else "Do not include steps.")
        + f"\n\nProblem:\n{body.problem}"
    )

    try:
        from app.services.llm_gateway import llm_gateway
        
        result = await llm_gateway.generate_json(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            schema=schema,
            temperature=0.0,
            max_tokens=800,
        )
        if not result.success:
            raise HTTPException(status_code=502, detail=f"LLM error: {result.error}")

        data = result.parsed_json
        usage = getattr(result, "usage", {}) or {}
        return SolveResponse(
            solution=data.get("solution", ""),
            steps=data.get("steps"),
            usage=usage
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("math-rag solve failed: %s", e)
        raise HTTPException(status_code=500, detail="math-rag solve failed")


@router.post("/check", response_model=CheckResponse, status_code=status.HTTP_200_OK)
async def check_answer(
    body: CheckRequest,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_student),
):
    """
    Kullanıcı cevabını değerlendirir (anahtar varsa ona göre; yoksa LLM kıyaslar).
    """
    schema = {
        "type": "object",
        "properties": {
            "correct": {"type": "boolean"},
            "explanation": {"type": "string"}
        },
        "required": ["correct"]
    }

    sys_prompt = (
        "You are a math answer checker. Return STRICT JSON only. "
        "If an answer key is provided, compare strictly; otherwise judge correctness."
    )
    key_line = f"\nAnswerKey: {body.answer_key}" if body.answer_key else ""
    explain_line = "\nExplain briefly." if body.require_explanation else ""
    user_prompt = f"Question:\n{body.question}\nUserAnswer: {body.user_answer}{key_line}{explain_line}"

    try:
        from app.services.llm_gateway import llm_gateway
        
        result = await llm_gateway.generate_json(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            schema=schema,
            temperature=0.0,
            max_tokens=400,
        )
        if not result.success:
            raise HTTPException(status_code=502, detail=f"LLM error: {result.error}")

        data = result.parsed_json
        usage = getattr(result, "usage", {}) or {}
        return CheckResponse(
            correct=bool(data.get("correct")),
            explanation=data.get("explanation"),
            usage=usage
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("math-rag check failed: %s", e)
        raise HTTPException(status_code=500, detail="math-rag check failed")


# ---------- Grounded RAG Endpoints ----------
class GroundedQueryRequest(BaseModel):
    query: str = Field(..., description="Soru veya sorgu")
    namespace: Optional[str] = Field(default="math", description="Namespace")
    max_results: Optional[int] = Field(default=5, ge=1, le=20, description="Maksimum sonuç sayısı")
    min_similarity: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="Minimum benzerlik skoru")


class GroundedQueryResponse(BaseModel):
    answer: str
    citations: List[str]
    answerable: bool
    confidence: float
    validation: Dict[str, Any]
    retrieved_docs: List[Dict[str, Any]]
    latency_ms: int


@router.post("/grounded-query", response_model=GroundedQueryResponse, status_code=status.HTTP_200_OK)
async def grounded_query(
    body: GroundedQueryRequest,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_student),
):
    """
    Kaynak tabanlı RAG sorgusu - sadece mevcut dokümanlara dayalı yanıt üretir.
    """
    start_time = time.time()
    
    try:
        # Generate query embedding
        from app.services.embedding_service import embedding_service
        query_embedding = await embedding_service.get_embedding(body.query)
        
        # Retrieve relevant documents
        from app.services.vector_index_manager import vector_index_manager
        retrieved_docs = await vector_index_manager.perform_vector_search(
            query_embedding=query_embedding,
            table_name="questions",
            filters={"subject": "math"},
            limit=body.max_results,
            namespace=body.namespace,
            min_similarity=body.min_similarity
        )
        
        # Generate grounded response
        response = await rag_validation_service.generate_grounded_response(
            query=body.query,
            retrieved_docs=retrieved_docs
        )
        
        # Add retrieved documents and latency
        response["retrieved_docs"] = retrieved_docs
        response["latency_ms"] = int((time.time() - start_time) * 1000)
        
        return GroundedQueryResponse(**response)
        
    except Exception as e:
        logger.error(f"Error in grounded query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grounded query failed: {str(e)}"
        )


@router.get("/validation-stats", status_code=status.HTTP_200_OK)
async def get_validation_stats(
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_teacher),
):
    """
    RAG validasyon istatistiklerini döndürür.
    """
    try:
        stats = await rag_validation_service.get_retrieval_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting validation stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get validation stats: {str(e)}"
        )
