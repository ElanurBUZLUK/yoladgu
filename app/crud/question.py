from typing import Optional
from sqlalchemy.orm import Session
from app.db.models import Question, QuestionSkill
from app.schemas.question import QuestionCreate, QuestionUpdate

def get_question(db: Session, question_id: int) -> Optional[Question]:
    return db.query(Question).filter(Question.id == question_id, Question.is_active == True).first()

def get_questions(db: Session, skip: int = 0, limit: int = 100) -> list[Question]:
    return db.query(Question).filter(Question.is_active == True).offset(skip).limit(limit).all()

def create_question(db: Session, question_in: QuestionCreate, user_id: int) -> Question:
    db_question = Question(
        content=question_in.content,
        question_type=question_in.question_type,
        difficulty_level=question_in.difficulty_level,
        subject_id=question_in.subject_id,
        options=question_in.options,
        correct_answer=question_in.correct_answer,
        explanation=question_in.explanation,
        tags=question_in.tags,
        created_by=user_id
    )
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    if question_in.skill_ids:
        for skill_id, weight in question_in.skill_ids.items():
            db_question_skill = QuestionSkill(
                question_id=db_question.id,
                skill_id=skill_id,
                weight=weight
            )
            db.add(db_question_skill)
        db.commit()
    return db_question

def update_question(db: Session, db_obj: Question, question_in: QuestionUpdate) -> Question:
    update_data = question_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj

def delete_question(db: Session, question_id: int) -> Optional[Question]:
    db_obj = db.query(Question).get(question_id)
    if db_obj:
        db_obj.is_active = False
        db.commit()
        db.refresh(db_obj)
    return db_obj 