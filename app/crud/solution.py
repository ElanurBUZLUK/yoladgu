from typing import List, Optional

from app.db.models import Solution
from app.schemas.solution import SolutionCreate, SolutionUpdate
from sqlalchemy.orm import Session


def get_solution(db: Session, solution_id: int) -> Optional[Solution]:
    return db.query(Solution).filter(Solution.id == solution_id).first()


def get_solutions(
    db: Session, user_id: Optional[int] = None, skip: int = 0, limit: int = 100
) -> List[Solution]:
    query = db.query(Solution)
    if user_id is not None:
        query = query.filter(Solution.user_id == user_id)
    return query.offset(skip).limit(limit).all()


def create_solution(db: Session, solution_in: SolutionCreate, user_id: int) -> Solution:
    db_solution = Solution(
        user_id=user_id,
        question_id=solution_in.question_id,
        solved_at=solution_in.solved_at,
        is_correct=solution_in.is_correct,
        duration=solution_in.duration,
    )
    db.add(db_solution)
    db.commit()
    db.refresh(db_solution)
    return db_solution


def update_solution(
    db: Session, db_obj: Solution, solution_in: SolutionUpdate
) -> Solution:
    update_data = solution_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def delete_solution(db: Session, solution_id: int) -> Optional[Solution]:
    db_obj = db.query(Solution).get(solution_id)
    if db_obj:
        db.delete(db_obj)
        db.commit()
    return db_obj
