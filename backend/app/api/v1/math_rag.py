from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging
import time
from datetime import datetime

from app.core.database import get_async_session
from app.middleware.auth import get_current_student, get_current_teacher
from app.models.user import User
from app.models.question import Subject, QuestionType
from app.services.math_selector import math_selector
from app.services.math_profile_manager import math_profile_manager
from app.services.mcp_service import mcp_service
from app.core.cache import cache_service

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
        
        # Cevabı değerlendir (MCP service ile)
        evaluation_result = await mcp_service.evaluate_student_answer(
            question_content=question.content,
            correct_answer=question.correct_answer or "",
            student_answer=submission.student_answer,
            subject="math",
            question_type=question.question_type.value,
            difficulty_level=question.difficulty_level
        )
        
        # Değerlendirme sonuçlarını al
        is_correct = evaluation_result.get("is_correct", False)
        score = evaluation_result.get("score", 0.0)
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
            "profile_management"
        ]
    }
