from datetime import datetime
from typing import List, Optional

from app.db.models import QuizSession
from app.schemas.quiz_session import QuizSessionCreate
from sqlalchemy.orm import Session


def create_quiz_session(
    db: Session, quiz_data: QuizSessionCreate, student_id: int
) -> QuizSession:
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
    return db_quiz


def get_quiz_sessions_by_student(
    db: Session, student_id: int, skip: int = 0, limit: int = 10
) -> List[QuizSession]:
    return (
        db.query(QuizSession)
        .filter(QuizSession.student_id == student_id)
        .order_by(QuizSession.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_quiz_session_count_by_student(db: Session, student_id: int) -> int:
    return db.query(QuizSession).filter(QuizSession.student_id == student_id).count()


def get_quiz_session_by_id(db: Session, quiz_session_id: int) -> Optional[QuizSession]:
    return db.query(QuizSession).filter(QuizSession.id == quiz_session_id).first()


def get_student_quiz_stats(db: Session, student_id: int) -> dict:
    """Öğrencinin quiz istatistiklerini getir"""
    sessions = db.query(QuizSession).filter(QuizSession.student_id == student_id).all()

    if not sessions:
        return {
            "total_sessions": 0,
            "total_questions": 0,
            "total_correct": 0,
            "average_accuracy": 0.0,
            "average_time_per_question": 0.0,
        }

    total_sessions = len(sessions)
    total_questions = sum(s.total_questions for s in sessions)
    total_correct = sum(s.correct_answers for s in sessions)
    total_time = sum(s.total_time_seconds for s in sessions)

    average_accuracy = (
        (total_correct / total_questions * 100) if total_questions > 0 else 0.0
    )
    average_time_per_question = (
        (total_time / total_questions) if total_questions > 0 else 0.0
    )

    return {
        "total_sessions": total_sessions,
        "total_questions": total_questions,
        "total_correct": total_correct,
        "average_accuracy": round(average_accuracy, 2),
        "average_time_per_question": round(average_time_per_question, 2),
    }
