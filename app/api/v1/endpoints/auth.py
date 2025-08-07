from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import create_access_token, get_current_user
from app.crud.user import (
    authenticate_user,
    create_user,
    get_user_by_email,
    get_user_by_username,
)
from app.db.database import get_db
from app.schemas.token import Token, TokenResponse
from app.schemas.user import UserCreate, UserProfile

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserProfile)
def register_user(
    *,
    db: Session = Depends(get_db),
    user_in: UserCreate,
) -> Any:
    """
    Create new user registration
    """
    # Check if user with same email exists
    user = get_user_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists",
        )

    # Check if user with same username exists
    user = get_user_by_username(db, username=user_in.username)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this username already exists",
        )

    user = create_user(db, user=user_in)
    return user


@router.post("/login", response_model=TokenResponse)
def login_access_token(
    db: Session = Depends(get_db), 
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = authenticate_user(
        db, username=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    elif not getattr(user, "is_active", True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id}, 
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        username=user.username,
        email=user.email,
        role=user.role.value if user.role else "student"
    )


@router.get("/me", response_model=UserProfile)
def get_current_user_profile(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get current user profile
    """
    return current_user


@router.post("/logout")
def logout() -> Any:
    """
    Logout user (client should discard token)
    """
    return {"message": "Successfully logged out"}
