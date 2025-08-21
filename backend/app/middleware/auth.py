from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from app.core.security import security_service
from app.models.user import User, UserRole
from app.database import database_manager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(database_manager.get_session)
) -> User:
    """Get current authenticated user"""
    
    # Verify token
    payload = security_service.verify_token(credentials.credentials)
    
    # Extract user info from token
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    # Get user from database
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if user.is_active != "true":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user"""
    return current_user


async def get_current_student(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current user if they are a student"""
    if current_user.role != UserRole.STUDENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student access required"
        )
    return current_user


# Mock authentication for development
async def get_current_student_mock() -> User:
    """Mock student user for development"""
    from app.models.user import UserRole
    
    # Create a mock user
    mock_user = User(
        id="1",
        email="student@test.com",
        name="Test Student",
        role=UserRole.STUDENT,
        is_active="true"
    )
    return mock_user


async def get_current_teacher_mock() -> User:
    """Mock teacher user for development"""
    from app.models.user import UserRole
    
    # Create a mock user
    mock_user = User(
        id="2",
        email="teacher@test.com", 
        name="Test Teacher",
        role=UserRole.TEACHER,
        is_active="true"
    )
    return mock_user


async def get_current_teacher(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current user if they are a teacher"""
    if current_user.role != UserRole.TEACHER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Teacher access required"
        )
    return current_user


async def get_current_admin(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current user if they are an admin"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_roles(*allowed_roles: UserRole):
    """Decorator to require specific roles"""
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[role.value for role in allowed_roles]}"
            )
        return current_user
    return role_checker


# Optional authentication (for public endpoints that can benefit from user context)
async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: AsyncSession = Depends(database_manager.get_session)
) -> Optional[User]:
    """Get current user if authenticated, otherwise return None"""
    
    if credentials is None:
        return None
    
    try:
        payload = security_service.verify_token(credentials.credentials)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            return None
        
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if user and user.is_active == "true":
            return user
        
    except Exception:
        pass
    
    return None