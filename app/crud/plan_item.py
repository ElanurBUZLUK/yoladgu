from typing import List, Optional

from app.db.models import PlanItem
from app.schemas.plan_item import PlanItemCreate, PlanItemUpdate
from sqlalchemy.orm import Session


def get_plan_item(db: Session, item_id: int) -> Optional[PlanItem]:
    return db.query(PlanItem).filter(PlanItem.id == item_id).first()


def get_plan_items(
    db: Session, plan_id: Optional[int] = None, skip: int = 0, limit: int = 100
) -> List[PlanItem]:
    query = db.query(PlanItem)
    if plan_id is not None:
        query = query.filter(PlanItem.plan_id == plan_id)
    return query.offset(skip).limit(limit).all()


def create_plan_item(db: Session, item_in: PlanItemCreate, plan_id: int) -> PlanItem:
    db_item = PlanItem(
        plan_id=plan_id,
        subject_id=item_in.subject_id,
        topic_id=item_in.topic_id,
        question_count=item_in.question_count,
        difficulty=item_in.difficulty,
        estimated_time=item_in.estimated_time,
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


def update_plan_item(
    db: Session, db_obj: PlanItem, item_in: PlanItemUpdate
) -> PlanItem:
    update_data = item_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def delete_plan_item(db: Session, item_id: int) -> Optional[PlanItem]:
    db_obj = db.query(PlanItem).get(item_id)
    if db_obj:
        db.delete(db_obj)
        db.commit()
    return db_obj
