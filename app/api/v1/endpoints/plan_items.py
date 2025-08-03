from typing import List, Optional

from app.core.security import get_current_user
from app.crud import plan_item as crud_plan_item
from app.db.database import get_db
from app.db.models import User as UserModel
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import schemas

router = APIRouter()


@router.get("/", response_model=List[schemas.PlanItemResponse])
def read_plan_items(
    plan_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Tüm plan adımlarını listeler. plan_id verilirse o plana ait adımlar gelir.
    """
    return crud_plan_item.get_plan_items(db, plan_id=plan_id, skip=skip, limit=limit)


@router.get("/{item_id}", response_model=schemas.PlanItemResponse)
def read_plan_item(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Belirli bir plan adımını ID ile getirir.
    """
    db_item = crud_plan_item.get_plan_item(db, item_id=item_id)
    if db_item is None:
        raise HTTPException(status_code=404, detail="PlanItem not found")
    return db_item


@router.post(
    "/", response_model=schemas.PlanItemResponse, status_code=status.HTTP_201_CREATED
)
def create_plan_item(
    item_in: schemas.PlanItemCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Yeni bir plan adımı oluşturur (plan sahibi, öğretmen veya admin).
    """
    return crud_plan_item.create_plan_item(
        db=db, item_in=item_in, plan_id=item_in.plan_id
    )


@router.put("/{item_id}", response_model=schemas.PlanItemResponse)
def update_plan_item(
    item_id: int,
    item_in: schemas.PlanItemUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Bir plan adımını günceller (plan sahibi, öğretmen veya admin).
    """
    db_item = crud_plan_item.get_plan_item(db, item_id=item_id)
    if not db_item:
        raise HTTPException(status_code=404, detail="PlanItem not found")
    return crud_plan_item.update_plan_item(db=db, db_obj=db_item, item_in=item_in)


@router.delete("/{item_id}", response_model=schemas.PlanItemResponse)
def delete_plan_item(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Bir plan adımını siler (plan sahibi, öğretmen veya admin).
    """
    db_item = crud_plan_item.get_plan_item(db, item_id=item_id)
    if not db_item:
        raise HTTPException(status_code=404, detail="PlanItem not found")
    return crud_plan_item.delete_plan_item(db=db, item_id=item_id)
