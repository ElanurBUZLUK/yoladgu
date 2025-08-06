from datetime import datetime
from typing import Optional

from app.core.config import settings
from app.core.security import get_password_hash, verify_password
from app.db.database import get_db
from app.db.models import StudentProfile, User
from app.schemas.user import UserCreate, UserUpdate
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(getattr(User, "id") == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(getattr(User, "email") == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(getattr(User, "username") == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100) -> list[User]:
    return db.query(User).offset(skip).limit(limit).all()


def create_user(db: Session, user: UserCreate) -> User:
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        username=user.username,
        full_name=f"{user.first_name} {user.last_name}",
        grade=user.grade_level,
        hashed_password=hashed_password,
        role=user.role,
        is_active=True,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user(db: Session, user_id: int, user: UserUpdate) -> Optional[User]:
    db_obj = db.query(User).filter(User.id == user_id).first()
    if db_obj:
        update_data = user.dict(exclude_unset=True)
        if "password" in update_data:
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            setattr(db_obj, "hashed_password", hashed_password)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
    return db_obj


def update_user_progress(
    db: Session, user_id: int, progress_data: dict
) -> Optional[StudentProfile]:
    # Önce profile'ı bul veya oluştur
    profile = (
        db.query(StudentProfile).filter(StudentProfile.student_id == user_id).first()
    )
    if not profile:
        profile = StudentProfile(student_id=user_id)
        db.add(profile)

    # Progress verilerini güncelle
    for field, value in progress_data.items():
        if hasattr(profile, field):
            setattr(profile, field, value)

    setattr(profile, "updated_at", datetime.utcnow())
    db.commit()
    db.refresh(profile)
    return profile


def delete_user(db: Session, user_id: int) -> Optional[User]:
    db_obj = db.query(User).get(user_id)
    if db_obj:
        db.delete(db_obj)
        db.commit()
    return db_obj


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, getattr(user, "hashed_password", "")):
        return None
    return user


def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub", "")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(db, username=username)
    if user is None:
        raise credentials_exception
    return user
