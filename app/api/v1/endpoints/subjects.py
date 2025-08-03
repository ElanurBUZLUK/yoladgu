from typing import List

from app.core.security import get_current_user
from app.crud import subject as crud_subject
from app.db.database import get_db
from app.db.models import User as UserModel
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import schemas

router = APIRouter()


@router.get("/", response_model=List[schemas.SubjectResponse])
def read_subjects(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Tüm dersleri listeler.
    """
    return crud_subject.get_subjects(db, skip=skip, limit=limit)


@router.get("/{subject_id}", response_model=schemas.SubjectResponse)
def read_subject(
    subject_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Belirli bir dersi ID ile getirir.
    """
    db_subject = crud_subject.get_subject(db, subject_id=subject_id)
    if db_subject is None:
        raise HTTPException(status_code=404, detail="Subject not found")
    return db_subject


@router.post(
    "/", response_model=schemas.SubjectResponse, status_code=status.HTTP_201_CREATED
)
def create_subject(
    subject_in: schemas.SubjectCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Yeni bir ders oluşturur (sadece admin veya öğretmen).
    """
    if current_user.role not in ["admin", "teacher"]:
        raise HTTPException(status_code=403, detail="Not authorized to create subjects")
    return crud_subject.create_subject(db=db, subject_in=subject_in)


@router.put("/{subject_id}", response_model=schemas.SubjectResponse)
def update_subject(
    subject_id: int,
    subject_in: schemas.SubjectUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Bir dersi günceller (sadece admin veya öğretmen).
    """
    db_subject = crud_subject.get_subject(db, subject_id=subject_id)
    if not db_subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    if current_user.role not in ["admin", "teacher"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return crud_subject.update_subject(db=db, db_obj=db_subject, subject_in=subject_in)


@router.delete("/{subject_id}", response_model=schemas.SubjectResponse)
def delete_subject(
    subject_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Bir dersi siler (sadece admin veya öğretmen).
    """
    db_subject = crud_subject.get_subject(db, subject_id=subject_id)
    if not db_subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    if current_user.role not in ["admin", "teacher"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return crud_subject.delete_subject(db=db, subject_id=subject_id)
