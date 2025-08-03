from typing import List, Optional

from app.core.security import get_current_user
from app.crud import study_plan as crud_study_plan
from app.db.database import get_db
from app.db.models import User as UserModel
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import schemas

router = APIRouter()


@router.get("/", response_model=List[schemas.StudyPlanResponse])
def read_study_plans(
    user_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Tüm çalışma planlarını listeler. user_id verilirse o kullanıcıya ait planlar gelir.
    """
    return crud_study_plan.get_study_plans(db, user_id=user_id, skip=skip, limit=limit)


@router.get("/{plan_id}", response_model=schemas.StudyPlanResponse)
def read_study_plan(
    plan_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Belirli bir çalışma planını ID ile getirir.
    """
    db_plan = crud_study_plan.get_study_plan(db, plan_id=plan_id)
    if db_plan is None:
        raise HTTPException(status_code=404, detail="StudyPlan not found")
    return db_plan


@router.post(
    "/", response_model=schemas.StudyPlanResponse, status_code=status.HTTP_201_CREATED
)
def create_study_plan(
    plan_in: schemas.StudyPlanCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Yeni bir çalışma planı oluşturur (öğrenci, öğretmen veya admin).
    """
    return crud_study_plan.create_study_plan(
        db=db, plan_in=plan_in, user_id=current_user.id
    )


@router.put("/{plan_id}", response_model=schemas.StudyPlanResponse)
def update_study_plan(
    plan_id: int,
    plan_in: schemas.StudyPlanUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Bir çalışma planını günceller (sadece sahibi veya admin).
    """
    db_plan = crud_study_plan.get_study_plan(db, plan_id=plan_id)
    if not db_plan:
        raise HTTPException(status_code=404, detail="StudyPlan not found")
    if db_plan.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return crud_study_plan.update_study_plan(db=db, db_obj=db_plan, plan_in=plan_in)


@router.delete("/{plan_id}", response_model=schemas.StudyPlanResponse)
def delete_study_plan(
    plan_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Bir çalışma planını siler (sadece sahibi veya admin).
    """
    db_plan = crud_study_plan.get_study_plan(db, plan_id=plan_id)
    if not db_plan:
        raise HTTPException(status_code=404, detail="StudyPlan not found")
    if db_plan.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return crud_study_plan.delete_study_plan(db=db, plan_id=plan_id)
