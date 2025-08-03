from typing import List, Optional

from app.db.models import Topic
from app.schemas.topic import TopicCreate, TopicUpdate
from sqlalchemy.orm import Session


def get_topic(db: Session, topic_id: int) -> Optional[Topic]:
    return db.query(Topic).filter(Topic.id == topic_id).first()


def get_topics(
    db: Session, subject_id: Optional[int] = None, skip: int = 0, limit: int = 100
) -> List[Topic]:
    query = db.query(Topic)
    if subject_id is not None:
        query = query.filter(Topic.subject_id == subject_id)
    return query.offset(skip).limit(limit).all()


def create_topic(db: Session, topic_in: TopicCreate) -> Topic:
    db_topic = Topic(
        subject_id=topic_in.subject_id,
        name=topic_in.name,
        description=topic_in.description,
    )
    db.add(db_topic)
    db.commit()
    db.refresh(db_topic)
    return db_topic


def update_topic(db: Session, db_obj: Topic, topic_in: TopicUpdate) -> Topic:
    update_data = topic_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def delete_topic(db: Session, topic_id: int) -> Optional[Topic]:
    db_obj = db.query(Topic).get(topic_id)
    if db_obj:
        db.delete(db_obj)
        db.commit()
    return db_obj
