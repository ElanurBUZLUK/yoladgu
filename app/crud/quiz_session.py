import structlog
from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.models import QuizSession
from app.schemas.quiz_session import QuizSessionCreate

logger = structlog.get_logger()


def create_quiz_session(
    db: Session, quiz_data: QuizSessionCreate, student_id: int
) -> QuizSession:
    """Create quiz session"""
    try:
        # Session tarihlerini al
        started_at = quiz_data.get_started_at() or datetime.utcnow()
        completed_at = quiz_data.get_completed_at() or datetime.utcnow()

        db_quiz = QuizSession(
            student_id=student_id,
            total_questions=quiz_data.total_questions,
            correct_answers=quiz_data.correct_answers,
            wrong_answers=quiz_data.wrong_answers,
            accuracy_percentage=quiz_data.accuracy_percentage,
            total_time_seconds=quiz_data.total_time_seconds,
            started_at=started_at,
            completed_at=completed_at,
        )
        db.add(db_quiz)
        db.commit()
        db.refresh(db_quiz)
        
        logger.info("quiz_session_created", 
                   session_id=db_quiz.id,
                   student_id=student_id)
        return db_quiz
        
    except Exception as e:
        db.rollback()
        logger.error("create_quiz_session_error", 
                    student_id=student_id,
                    error=str(e))
        raise


def get_quiz_session_by_id(db: Session, quiz_session_id: int) -> Optional[QuizSession]:
    """Get quiz session by ID"""
    return db.query(QuizSession).filter(QuizSession.id == quiz_session_id).first()


def get_quiz_sessions_by_student(
    db: Session, student_id: int, skip: int = 0, limit: int = 10
) -> List[QuizSession]:
    """Get quiz sessions for student"""
    return db.query(QuizSession).filter(
        QuizSession.student_id == student_id
    ).offset(skip).limit(limit).all()


def get_quiz_session_count_by_student(db: Session, student_id: int) -> int:
    """Get quiz session count for student"""
    return db.query(QuizSession).filter(
        QuizSession.student_id == student_id
    ).count()


def get_student_quiz_stats(db: Session, student_id: int) -> dict:
    """Get quiz statistics for student"""
    try:
        # Total sessions
        total_sessions = db.query(QuizSession).filter(
            QuizSession.student_id == student_id
        ).count()
        
        # Average accuracy
        avg_accuracy = db.query(func.avg(QuizSession.accuracy_percentage)).filter(
            QuizSession.student_id == student_id
        ).scalar() or 0
        
        # Total questions answered
        total_questions = db.query(func.sum(QuizSession.total_questions)).filter(
            QuizSession.student_id == student_id
        ).scalar() or 0
        
        # Total correct answers
        total_correct = db.query(func.sum(QuizSession.correct_answers)).filter(
            QuizSession.student_id == student_id
        ).scalar() or 0
        
        # Total time spent
        total_time = db.query(func.sum(QuizSession.total_time_seconds)).filter(
            QuizSession.student_id == student_id
        ).scalar() or 0
        
        return {
            "total_sessions": total_sessions,
            "average_accuracy": round(avg_accuracy, 2),
            "total_questions": total_questions,
            "total_correct_answers": total_correct,
            "total_time_seconds": total_time,
            "success_rate": round((total_correct / total_questions * 100), 2) if total_questions > 0 else 0
        }
        
    except Exception as e:
        logger.error("get_student_quiz_stats_error", 
                    student_id=student_id,
                    error=str(e))
        return {
            "total_sessions": 0,
            "average_accuracy": 0,
            "total_questions": 0,
            "total_correct_answers": 0,
            "total_time_seconds": 0,
            "success_rate": 0
        }
