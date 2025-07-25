from typing import Optional, List
from sqlalchemy.orm import Session
from app.db.models import StudyPlan
from app.schemas.study_plan import StudyPlanCreate, StudyPlanUpdate

def get_study_plan(db: Session, plan_id: int) -> Optional[StudyPlan]:
    return db.query(StudyPlan).filter(StudyPlan.id == plan_id).first()

def get_study_plans(db: Session, user_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[StudyPlan]:
    query = db.query(StudyPlan)
    if user_id is not None:
        query = query.filter(StudyPlan.user_id == user_id)
    return query.offset(skip).limit(limit).all()

def create_study_plan(db: Session, plan_in: StudyPlanCreate, user_id: int) -> StudyPlan:
    db_plan = StudyPlan(
        user_id=user_id,
        name=plan_in.name,
        created_at=plan_in.created_at,
        target_success=plan_in.target_success
    )
    db.add(db_plan)
    db.commit()
    db.refresh(db_plan)
    return db_plan

def update_study_plan(db: Session, db_obj: StudyPlan, plan_in: StudyPlanUpdate) -> StudyPlan:
    update_data = plan_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj

def delete_study_plan(db: Session, plan_id: int) -> Optional[StudyPlan]:
    db_obj = db.query(StudyPlan).get(plan_id)
    if db_obj:
        db.delete(db_obj)
        db.commit()
    return db_obj 