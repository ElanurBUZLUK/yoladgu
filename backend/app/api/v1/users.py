from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from datetime import timedelta

from app.database import database_manager
from app.core.security import security_service
from app.services.user_service import user_service
from app.schemas.auth import (
    UserCreate, UserLogin, UserResponse, UserUpdate, 
    Token, ChangePassword, TokenData, PasswordReset, 
    PasswordResetConfirm, UserStats, BulkUserOperation, UserSearchQuery
)
from app.middleware.auth import (
    get_current_active_user, get_current_admin, 
    require_roles, get_optional_current_user
)
from app.models.user import User, UserRole

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Register a new user"""
    user = await user_service.create_user(db, user_data)
    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role,
        learning_style=user.learning_style,
        current_math_level=user.current_math_level,
        current_english_level=user.current_english_level,
        is_active=user.is_active,
        created_at=user.created_at.isoformat(),
        updated_at=user.updated_at.isoformat()
    )


@router.post("/login", response_model=Token)
async def login_user(
    user_credentials: UserLogin,
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Login user and return JWT tokens"""
    
    # Authenticate user
    user = await user_service.authenticate_user(
        db, user_credentials.username, user_credentials.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token_expires = timedelta(minutes=security_service.access_token_expire_minutes)
    access_token = security_service.create_access_token(
        data={"sub": str(user.id), "username": user.username, "role": user.role.value},
        expires_delta=access_token_expires
    )
    
    refresh_token = security_service.create_refresh_token(
        data={"sub": str(user.id), "username": user.username}
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=security_service.access_token_expire_minutes * 60
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Refresh access token using refresh token"""
    
    # Verify refresh token
    payload = security_service.verify_refresh_token(refresh_token)
    user_id = payload.get("sub")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Get user
    user = await user_service.get_user_by_id(db, user_id)
    if not user or user.is_active != "true":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new tokens
    access_token_expires = timedelta(minutes=security_service.access_token_expire_minutes)
    access_token = security_service.create_access_token(
        data={"sub": str(user.id), "username": user.username, "role": user.role.value},
        expires_delta=access_token_expires
    )
    
    new_refresh_token = security_service.create_refresh_token(
        data={"sub": str(user.id), "username": user.username}
    )
    
    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=security_service.access_token_expire_minutes * 60
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information"""
    return UserResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        learning_style=current_user.learning_style,
        current_math_level=current_user.current_math_level,
        current_english_level=current_user.current_english_level,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat(),
        updated_at=current_user.updated_at.isoformat()
    )


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Update current user information"""
    
    updated_user = await user_service.update_user(db, str(current_user.id), user_update)
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(
        id=str(updated_user.id),
        username=updated_user.username,
        email=updated_user.email,
        role=updated_user.role,
        learning_style=updated_user.learning_style,
        current_math_level=updated_user.current_math_level,
        current_english_level=updated_user.current_english_level,
        is_active=updated_user.is_active,
        created_at=updated_user.created_at.isoformat(),
        updated_at=updated_user.updated_at.isoformat()
    )


@router.post("/change-password")
async def change_password(
    password_data: ChangePassword,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Change user password"""
    
    success = await user_service.change_password(
        db, str(current_user.id), 
        password_data.current_password, 
        password_data.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to change password"
        )
    
    return {"message": "Password changed successfully"}


@router.put("/levels")
async def update_user_levels(
    math_level: int = None,
    english_level: int = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Update user subject levels"""
    
    updated_user = await user_service.update_user_levels(
        db, str(current_user.id), math_level, english_level
    )
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "message": "Levels updated successfully",
        "math_level": updated_user.current_math_level,
        "english_level": updated_user.current_english_level
    }


# Admin endpoints
@router.get("/", response_model=List[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    role: UserRole = None,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get list of users (Admin only)"""
    
    users = await user_service.get_users(db, skip, limit, role)
    
    return [
        UserResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            role=user.role,
            learning_style=user.learning_style,
            current_math_level=user.current_math_level,
            current_english_level=user.current_english_level,
            is_active=user.is_active,
            created_at=user.created_at.isoformat(),
            updated_at=user.updated_at.isoformat()
        )
        for user in users
    ]


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: str,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get user by ID (Admin only)"""
    
    user = await user_service.get_user_by_id(db, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role,
        learning_style=user.learning_style,
        current_math_level=user.current_math_level,
        current_english_level=user.current_english_level,
        is_active=user.is_active,
        created_at=user.created_at.isoformat(),
        updated_at=user.updated_at.isoformat()
    )


@router.put("/{user_id}/deactivate")
async def deactivate_user(
    user_id: str,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Deactivate user (Admin only)"""
    
    success = await user_service.deactivate_user(db, user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": "User deactivated successfully"}


@router.put("/{user_id}/activate")
async def activate_user(
    user_id: str,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Activate user (Admin only)"""
    
    success = await user_service.activate_user(db, user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": "User activated successfully"}


# Password reset endpoints
@router.post("/password-reset")
async def request_password_reset(
    reset_data: PasswordReset,
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Request password reset token"""
    
    token = await user_service.generate_password_reset_token(db, reset_data.email)
    
    if token:
        # In production, send email with token
        # For now, return token (remove in production!)
        return {
            "message": "Password reset token generated",
            "token": token  # Remove this in production!
        }
    else:
        # Don't reveal if email exists or not for security
        return {"message": "If the email exists, a reset token has been sent"}


@router.post("/password-reset/confirm")
async def confirm_password_reset(
    reset_confirm: PasswordResetConfirm,
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Confirm password reset with token"""
    
    success = await user_service.reset_password_with_token(
        db, reset_confirm.token, reset_confirm.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    return {"message": "Password reset successfully"}


# User statistics and analytics
@router.get("/me/stats", response_model=UserStats)
async def get_current_user_stats(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get current user statistics"""
    
    stats = await user_service.get_user_activity_summary(db, str(current_user.id))
    
    return UserStats(
        total_attempts=stats.get("total_attempts", 0),
        correct_attempts=stats.get("correct_attempts", 0),
        accuracy_rate=stats.get("accuracy_rate", 0.0),
        average_time_spent=stats.get("average_time_spent", 0.0),
        current_streak=stats.get("current_streak", 0),
        last_activity=stats.get("last_activity"),
        subjects_progress={
            "math_level": stats.get("math_level", 1),
            "english_level": stats.get("english_level", 1)
        }
    )


# Advanced admin endpoints
@router.post("/search", response_model=List[UserResponse])
async def search_users(
    search_query: UserSearchQuery,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Advanced user search (Admin only)"""
    
    users = await user_service.search_users(db, search_query, skip, limit)
    
    return [
        UserResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            role=user.role,
            learning_style=user.learning_style,
            current_math_level=user.current_math_level,
            current_english_level=user.current_english_level,
            is_active=user.is_active,
            created_at=user.created_at.isoformat(),
            updated_at=user.updated_at.isoformat()
        )
        for user in users
    ]


@router.get("/stats/summary")
async def get_user_stats_summary(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get user statistics summary (Admin only)"""
    
    stats = await user_service.get_user_stats_summary(db)
    return stats


@router.get("/{user_id}/stats", response_model=UserStats)
async def get_user_stats(
    user_id: str,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get user statistics by ID (Admin only)"""
    
    stats = await user_service.get_user_activity_summary(db, user_id)
    
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserStats(
        total_attempts=stats.get("total_attempts", 0),
        correct_attempts=stats.get("correct_attempts", 0),
        accuracy_rate=stats.get("accuracy_rate", 0.0),
        average_time_spent=stats.get("average_time_spent", 0.0),
        current_streak=stats.get("current_streak", 0),
        last_activity=stats.get("last_activity"),
        subjects_progress={
            "math_level": stats.get("math_level", 1),
            "english_level": stats.get("english_level", 1)
        }
    )


@router.post("/bulk-operation")
async def bulk_user_operation(
    operation_data: BulkUserOperation,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Perform bulk operations on users (Admin only)"""
    
    result = await user_service.bulk_user_operation(
        db, operation_data.user_ids, operation_data.operation
    )
    
    return result


@router.get("/health")
async def users_health():
    """Kullanıcı modülü health check"""
    return {"status": "ok", "module": "users"}