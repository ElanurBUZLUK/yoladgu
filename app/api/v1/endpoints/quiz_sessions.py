from typing import Any
from datetime import datetime

from app.crud.quiz_session import (
    create_quiz_session,
    get_quiz_session_by_id,
    get_quiz_session_count_by_student,
    get_quiz_sessions_by_student,
    get_student_quiz_stats,
)
from app.crud.user import get_current_user
from app.db.database import get_db
from app.db.models import User as UserModel
from app.schemas.quiz_session import QuizSession, QuizSessionCreate, QuizSessionList, StudentResponse, StudentResponseCreate
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.core.dependencies import get_skill_graph_service
from app.services.skill_graph_service import SkillGraphService

router = APIRouter(prefix="/quiz-sessions", tags=["quiz-sessions"])


@router.post("/", response_model=QuizSession)
def create_quiz_session_endpoint(
    *,
    db: Session = Depends(get_db),
    quiz_data: QuizSessionCreate,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Create a new quiz session for the current user
    """
    try:
        quiz_session = create_quiz_session(
            db=db, quiz_data=quiz_data, student_id=getattr(current_user, "id", 1)
        )
        return quiz_session
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create quiz session: {str(e)}",
        )


@router.get("/", response_model=QuizSessionList)
def get_quiz_sessions(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
) -> Any:
    """
    Get quiz sessions for the current user
    """
    sessions = get_quiz_sessions_by_student(
        db=db, student_id=getattr(current_user, "id", 1), skip=skip, limit=limit
    )
    total_count = get_quiz_session_count_by_student(
        db=db, student_id=getattr(current_user, "id", 1)
    )

    return QuizSessionList(sessions=sessions, total_count=total_count)  # type: ignore


@router.get("/stats")
def get_quiz_stats(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Get quiz statistics for the current user
    """
    stats = get_student_quiz_stats(db=db, student_id=getattr(current_user, "id", 1))
    return stats


@router.get("/{quiz_session_id}", response_model=QuizSession)
def get_quiz_session(
    quiz_session_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Get a specific quiz session by ID
    """
    quiz_session = get_quiz_session_by_id(db=db, quiz_session_id=quiz_session_id)

    if not quiz_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Quiz session not found"
        )

    # Check if the quiz session belongs to the current user
    if getattr(quiz_session, "student_id", 0) != getattr(current_user, "id", 0):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    return quiz_session


@router.post(
    "/{session_id}/responses",
    response_model=StudentResponse,
    summary="Öğrencinin bir soruya verdiği yanıtı kaydeder"
)
def submit_response(
    session_id: int,
    in_resp: StudentResponseCreate,
    db: Session = Depends(get_db),
    skill_svc: SkillGraphService = Depends(get_skill_graph_service),
):
    """
    Öğrencinin bir soruya verdiği yanıtı kaydeder ve skill graph cache'ini temizler
    """
    try:
        # 1) DB'ye kaydet (bu kısım crud fonksiyonu ile yapılacak)
        # resp = crud.student_response.create(db, obj_in=in_resp)
        
        # Şimdilik basit bir response döndürüyoruz
        # Gerçek implementasyonda crud fonksiyonu kullanılacak
        resp = StudentResponse(
            id=1,
            question_id=in_resp.question_id,
            user_id=in_resp.user_id,
            topic_id=in_resp.topic_id,
            answer_text=in_resp.answer_text,
            is_correct=in_resp.is_correct,
            time_taken_seconds=in_resp.time_taken_seconds,
            confidence_level=in_resp.confidence_level,
            quiz_session_id=session_id,
            created_at=datetime.now(),
            updated_at=None
        )
        
        # 2) SkillGraphService'in cache'ini temizle
        #    invalidate student+topic bazında, böylece weak-skills yeniden hesaplanır
        skill_svc.invalidate(student_id=in_resp.user_id, topic_id=in_resp.topic_id)
        
        return resp
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit response: {str(e)}",
        )
