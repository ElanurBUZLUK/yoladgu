from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app import schemas
from app.crud import topic as crud_topic
from app.db.database import get_db
from app.core.security import get_current_user
from app.db.models import User as UserModel

router = APIRouter()

@router.get("/", response_model=List[schemas.TopicResponse])
def read_topics(
    subject_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Tüm konuları listeler. subject_id verilirse o derse ait konular gelir.
    """
    return crud_topic.get_topics(db, subject_id=subject_id, skip=skip, limit=limit)

@router.get("/{topic_id}", response_model=schemas.TopicResponse)
def read_topic(
    topic_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Belirli bir konuyu ID ile getirir.
    """
    db_topic = crud_topic.get_topic(db, topic_id=topic_id)
    if db_topic is None:
        raise HTTPException(status_code=404, detail="Topic not found")
    return db_topic

@router.post("/", response_model=schemas.TopicResponse, status_code=status.HTTP_201_CREATED)
def create_topic(
    topic_in: schemas.TopicCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Yeni bir konu oluşturur (sadece admin veya öğretmen).
    """
    if current_user.role not in ["admin", "teacher"]:
        raise HTTPException(status_code=403, detail="Not authorized to create topics")
    return crud_topic.create_topic(db=db, topic_in=topic_in)

@router.put("/{topic_id}", response_model=schemas.TopicResponse)
def update_topic(
    topic_id: int,
    topic_in: schemas.TopicUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Bir konuyu günceller (sadece admin veya öğretmen).
    """
    db_topic = crud_topic.get_topic(db, topic_id=topic_id)
    if not db_topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    if current_user.role not in ["admin", "teacher"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return crud_topic.update_topic(db=db, db_obj=db_topic, topic_in=topic_in)

@router.delete("/{topic_id}", response_model=schemas.TopicResponse)
def delete_topic(
    topic_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Bir konuyu siler (sadece admin veya öğretmen).
    """
    db_topic = crud_topic.get_topic(db, topic_id=topic_id)
    if not db_topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    if current_user.role not in ["admin", "teacher"]:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return crud_topic.delete_topic(db=db, topic_id=topic_id) 