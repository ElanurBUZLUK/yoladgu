from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.crud.user import get_user, get_users, update_user, delete_user
from app.db.database import get_db
from app.schemas.user import UserProfile, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserProfile)
def get_current_user_profile(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Get current user profile"""
    return current_user


@router.put("/me", response_model=UserProfile)
def update_current_user(
    user_update: UserUpdate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Update current user profile"""
    updated_user = update_user(db, current_user.id, user_update)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return updated_user


@router.get("/me/level")
def get_current_user_level(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Get current user level"""
    try:
        # Get user profile for level info
        profile = get_user_profile(db, current_user.id)
        if profile:
            return {
                "level": profile.level,
                "max_level": profile.max_level,
                "experience": profile.total_questions_answered * 10,  # Mock experience
                "next_level_exp": 1500  # Mock next level requirement
            }
        else:
            return {
                "level": 1,
                "max_level": 20,
                "experience": 0,
                "next_level_exp": 100
            }
    except Exception as e:
        logger.error("get_user_level_error", user_id=current_user.id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user level"
        )


@router.put("/me/progress")
def update_current_user_progress(
    progress_data: dict,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Update current user progress"""
    try:
        updated_profile = update_user_profile(db, current_user.id, progress_data)
        return {"message": "Progress updated successfully", "profile": updated_profile}
    except Exception as e:
        logger.error("update_user_progress_error", user_id=current_user.id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update progress"
        )


@router.get("/{user_id}", response_model=UserProfile)
def get_user_by_id(
    user_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Get user by ID"""
    user = get_user(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.get("/", response_model=List[UserProfile])
def get_all_users(
    skip: int = 0,
    limit: int = 100,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """Get all users with pagination"""
    users = get_users(db, skip=skip, limit=limit)
    return users
