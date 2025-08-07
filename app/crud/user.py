"""
User CRUD Operations
Kullanıcı veritabanı işlemleri
"""

import structlog
from datetime import datetime
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.core.security import get_password_hash, verify_password
from app.db.models import User, StudentProfile
from app.schemas.user import UserCreate, UserUpdate

logger = structlog.get_logger()


def get_user(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID"""
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    """Get users with pagination"""
    return db.query(User).offset(skip).limit(limit).all()


def create_user(db: Session, user: UserCreate) -> User:
    """Create new user"""
    try:
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
        
        logger.info("user_created", user_id=db_user.id, email=db_user.email)
        return db_user
        
    except Exception as e:
        db.rollback()
        logger.error("user_creation_error", error=str(e))
        raise


def update_user(db: Session, user_id: int, user: UserUpdate) -> Optional[User]:
    """Update user"""
    try:
        db_obj = db.query(User).filter(User.id == user_id).first()
        if not db_obj:
            return None
            
        update_data = user.dict(exclude_unset=True)
        if "password" in update_data:
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            setattr(db_obj, "hashed_password", hashed_password)
            
        for field, value in update_data.items():
            setattr(db_obj, field, value)
            
        db_obj.updated_at = datetime.utcnow()
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        
        logger.info("user_updated", user_id=user_id)
        return db_obj
        
    except Exception as e:
        db.rollback()
        logger.error("user_update_error", user_id=user_id, error=str(e))
        raise


def delete_user(db: Session, user_id: int) -> bool:
    """Delete user"""
    try:
        db_obj = db.query(User).filter(User.id == user_id).first()
        if not db_obj:
            return False
            
        db.delete(db_obj)
        db.commit()
        
        logger.info("user_deleted", user_id=user_id)
        return True
        
    except Exception as e:
        db.rollback()
        logger.error("user_deletion_error", user_id=user_id, error=str(e))
        return False


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate user"""
    try:
        # Try username first
        user = get_user_by_username(db, username)
        if not user:
            # Try email
            user = get_user_by_email(db, username)
        
        if not user:
            return None
            
        if not verify_password(password, user.hashed_password):
            return None
            
        if not user.is_active:
            return None
            
        logger.info("user_authenticated", user_id=user.id, username=user.username)
        return user
        
    except Exception as e:
        logger.error("authentication_error", username=username, error=str(e))
        return None


def get_user_profile(db: Session, user_id: int) -> Optional[StudentProfile]:
    """Get user profile"""
    return db.query(StudentProfile).filter(StudentProfile.student_id == user_id).first()


def create_user_profile(db: Session, user_id: int) -> StudentProfile:
    """Create user profile"""
    try:
        profile = StudentProfile(student_id=user_id)
        db.add(profile)
        db.commit()
        db.refresh(profile)
        
        logger.info("user_profile_created", user_id=user_id)
        return profile
        
    except Exception as e:
        db.rollback()
        logger.error("profile_creation_error", user_id=user_id, error=str(e))
        raise


def update_user_profile(db: Session, user_id: int, profile_data: dict) -> Optional[StudentProfile]:
    """Update user profile"""
    try:
        profile = get_user_profile(db, user_id)
        if not profile:
            profile = create_user_profile(db, user_id)
        
        for field, value in profile_data.items():
            if hasattr(profile, field):
                setattr(profile, field, value)
        
        profile.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(profile)
        
        logger.info("user_profile_updated", user_id=user_id)
        return profile
        
    except Exception as e:
        db.rollback()
        logger.error("profile_update_error", user_id=user_id, error=str(e))
        raise 