from typing import List, Optional

from app.db.models import Subject
from app.schemas.subject import SubjectCreate, SubjectUpdate
from sqlalchemy.orm import Session


def get_subject(db: Session, subject_id: int) -> Optional[Subject]:
    return db.query(Subject).filter(Subject.id == subject_id).first()


def get_subjects(db: Session, skip: int = 0, limit: int = 100) -> List[Subject]:
    return db.query(Subject).offset(skip).limit(limit).all()


def create_subject(db: Session, subject_in: SubjectCreate) -> Subject:
    db_subject = Subject(name=subject_in.name, description=subject_in.description)
    db.add(db_subject)
    db.commit()
    db.refresh(db_subject)
    return db_subject


def update_subject(db: Session, db_obj: Subject, subject_in: SubjectUpdate) -> Subject:
    update_data = subject_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def delete_subject(db: Session, subject_id: int) -> Optional[Subject]:
    db_obj = db.query(Subject).get(subject_id)
    if db_obj:
        db.delete(db_obj)
        db.commit()
    return db_obj
