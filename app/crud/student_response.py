from typing import Optional, List
from sqlalchemy.orm import Session

from app.db.models import StudentResponse


def get_response(db: Session, response_id: int) -> Optional[StudentResponse]:
    return db.query(StudentResponse).filter(StudentResponse.id == response_id).first()


def get_responses(db: Session, student_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[StudentResponse]:
    query = db.query(StudentResponse)
    if student_id is not None:
        query = query.filter(StudentResponse.student_id == student_id)
    return query.offset(skip).limit(limit).all()


def create_response(
    db: Session,
    *,
    student_id: int,
    question_id: int,
    answer: str,
    is_correct: bool,
    response_time: Optional[float] = None,
    confidence_level: Optional[int] = None,
    feedback: Optional[str] = None,
) -> StudentResponse:
    db_response = StudentResponse(
        student_id=student_id,
        question_id=question_id,
        answer=answer,
        is_correct=is_correct,
        response_time=response_time,
        confidence_level=confidence_level,
        feedback=feedback,
    )
    db.add(db_response)
    db.commit()
    db.refresh(db_response)
    return db_response


def delete_response(db: Session, response_id: int) -> Optional[StudentResponse]:
    db_obj = db.query(StudentResponse).get(response_id)
    if db_obj:
        db.delete(db_obj)
        db.commit()
    return db_obj 