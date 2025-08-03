from typing import Any, List

from app.crud.user import (
    create_user,
    delete_user,
    get_current_user,
    get_user,
    get_users,
    update_user,
    update_user_progress,
)
from app.db.database import get_db
from app.db.models import StudentProfile
from app.db.models import User as UserModel
from app.schemas.user import ProgressUpdate, User, UserCreate, UserUpdate
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/", response_model=List[User])
def read_users(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve users (requires authentication)
    """
    users = get_users(db, skip=skip, limit=limit)
    return users


@router.post("/", response_model=User)
def create_new_user(
    *,
    db: Session = Depends(get_db),
    user_in: UserCreate,
) -> Any:
    """
    Create new user (public endpoint for registration)
    """
    user = create_user(db, user=user_in)
    return user


@router.get("/me", response_model=User)
def read_user_me(
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Get current user information
    """
    return current_user


@router.get("/me/level")
def read_my_level(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    profile = db.query(StudentProfile).filter_by(student_id=current_user.id).first()
    if not profile:
        return {"level": 1.0, "min_level": 1.0, "max_level": 5.0}
    return {
        "level": profile.level,
        "min_level": profile.min_level,
        "max_level": profile.max_level,
    }


@router.put("/me/progress")
def update_my_progress(
    *,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
    progress_data: ProgressUpdate,
) -> Any:
    """
    Update current user's progress information
    """
    profile = update_user_progress(db, user_id=current_user.id, progress=progress_data)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update progress",
        )
    return {
        "message": "Progress updated successfully",
        "level": profile.level,
        "total_questions_answered": profile.total_questions_answered,
        "total_correct_answers": profile.total_correct_answers,
        "average_response_time": profile.average_response_time,
    }


@router.get("/{user_id}", response_model=User)
def read_user_by_id(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Get a specific user by id (requires authentication)
    """
    user = get_user(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return user


@router.put("/{user_id}", response_model=User)
def update_user_by_id(
    *,
    db: Session = Depends(get_db),
    user_id: int,
    user_in: UserUpdate,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Update a user (requires authentication and ownership or admin role)
    """
    # Check if user is updating their own profile or is admin
    if current_user.id != user_id and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    user = update_user(db, user_id=user_id, user=user_in)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return user


@router.delete("/{user_id}")
def delete_user_by_id(
    *,
    db: Session = Depends(get_db),
    user_id: int,
    current_user: UserModel = Depends(get_current_user),
) -> Any:
    """
    Delete a user (requires authentication and ownership or admin role)
    """
    # Check if user is deleting their own account or is admin
    if current_user.id != user_id and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )

    success = delete_user(db, user_id=user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return {"message": "User deleted successfully"}
