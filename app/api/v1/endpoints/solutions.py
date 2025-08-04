from typing import List, Optional

from app.core.security import get_current_user
from app.crud import solution as crud_solution
from app.db.database import get_db
from app.db.models import User as UserModel
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import schemas

router = APIRouter()


@router.get("/", response_model=List[schemas.SolutionResponse])
def read_solutions(
    user_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Çözümleri listeler. user_id verilirse o kullanıcıya ait çözümler gelir.
    """
    return crud_solution.get_solutions(db, user_id=user_id, skip=skip, limit=limit)


@router.get("/{solution_id}", response_model=schemas.SolutionResponse)
def read_solution(
    solution_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Belirli bir çözümü ID ile getirir.
    """
    db_solution = crud_solution.get_solution(db, solution_id=solution_id)
    if db_solution is None:
        raise HTTPException(status_code=404, detail="Solution not found")
    return db_solution


@router.post(
    "/", response_model=schemas.SolutionResponse, status_code=status.HTTP_201_CREATED
)
def create_solution(
    solution_in: schemas.SolutionCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Yeni bir çözüm kaydı oluşturur (öğrenci).
    """
    return crud_solution.create_solution(
        db=db, solution_in=solution_in, user_id=getattr(current_user, "id", 1)
    )


@router.put("/{solution_id}", response_model=schemas.SolutionResponse)
def update_solution(
    solution_id: int,
    solution_in: schemas.SolutionUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Bir çözümü günceller (sadece kendi çözümü veya admin).
    """
    db_solution = crud_solution.get_solution(db, solution_id=solution_id)
    if not db_solution:
        raise HTTPException(status_code=404, detail="Solution not found")
    if (
        getattr(db_solution, "user_id", 0) != getattr(current_user, "id", 0)
        and getattr(current_user, "role", "") != "admin"
    ):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return crud_solution.update_solution(
        db=db, db_obj=db_solution, solution_in=solution_in
    )


@router.delete("/{solution_id}", response_model=schemas.SolutionResponse)
def delete_solution(
    solution_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Bir çözümü siler (sadece kendi çözümü veya admin).
    """
    db_solution = crud_solution.get_solution(db, solution_id=solution_id)
    if not db_solution:
        raise HTTPException(status_code=404, detail="Solution not found")
    if (
        getattr(db_solution, "user_id", 0) != getattr(current_user, "id", 0)
        and getattr(current_user, "role", "") != "admin"
    ):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return crud_solution.delete_solution(db=db, solution_id=solution_id)
