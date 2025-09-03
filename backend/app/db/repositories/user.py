"""
User repository with profile management methods.
"""

from typing import Optional, Dict, Any, List
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_

from app.db.models import User
from app.db.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """User repository with profile-specific methods."""
    
    def __init__(self):
        super().__init__(User)
    
    async def get_by_email(self, session: AsyncSession, email: str) -> Optional[User]:
        """Get user by email."""
        statement = select(User).where(User.email == email)
        result = await session.exec(statement)
        return result.first()
    
    async def get_by_username(self, session: AsyncSession, username: str) -> Optional[User]:
        """Get user by username."""
        statement = select(User).where(User.username == username)
        result = await session.exec(statement)
        return result.first()
    
    async def get_by_username_or_email(
        self, 
        session: AsyncSession, 
        username_or_email: str
    ) -> Optional[User]:
        """Get user by username or email."""
        statement = select(User).where(
            (User.username == username_or_email) | (User.email == username_or_email)
        )
        result = await session.exec(statement)
        return result.first()
    
    async def update_theta(
        self, 
        session: AsyncSession, 
        user_id: str, 
        theta_math: Optional[float] = None,
        theta_en: Optional[float] = None
    ) -> Optional[User]:
        """Update user's theta values."""
        user = await self.get(session, user_id)
        if not user:
            return None
        
        updates = {}
        if theta_math is not None:
            updates["theta_math"] = theta_math
        if theta_en is not None:
            updates["theta_en"] = theta_en
        
        if updates:
            return await self.update(session, db_obj=user, obj_in=updates)
        return user
    
    async def update_error_profile(
        self,
        session: AsyncSession,
        user_id: str,
        subject: str,  # "math" or "en"
        error_updates: Dict[str, float]
    ) -> Optional[User]:
        """Update user's error profile for a specific subject."""
        user = await self.get(session, user_id)
        if not user:
            return None
        
        if subject == "math":
            current_profile = user.error_profile_math or {}
            current_profile.update(error_updates)
            updates = {"error_profile_math": current_profile}
        elif subject == "en":
            current_profile = user.error_profile_en or {}
            current_profile.update(error_updates)
            updates = {"error_profile_en": current_profile}
        else:
            raise ValueError(f"Invalid subject: {subject}")
        
        return await self.update(session, db_obj=user, obj_in=updates)
    
    async def get_users_by_tenant(
        self,
        session: AsyncSession,
        tenant_id: str,
        role: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """Get users by tenant with optional role filtering."""
        filters = {"tenant_id": tenant_id, "is_active": True}
        if role:
            filters["role"] = role
        
        return await self.get_multi(session, skip=skip, limit=limit, filters=filters)
    
    async def get_similar_users(
        self,
        session: AsyncSession,
        user_id: str,
        subject: str,
        limit: int = 10
    ) -> List[User]:
        """Get users with similar error profiles (for peer-aware recommendations)."""
        # TODO: Implement similarity calculation based on error profiles
        # This would involve calculating cosine similarity or other distance metrics
        # For now, return users with similar theta values
        
        user = await self.get(session, user_id)
        if not user:
            return []
        
        if subject == "math":
            theta_value = user.theta_math or 0.0
            theta_field = User.theta_math
        elif subject == "en":
            theta_value = user.theta_en or 0.0
            theta_field = User.theta_en
        else:
            return []
        
        # Find users with similar theta values (within 0.5 range)
        statement = select(User).where(
            and_(
                User.id != user_id,
                User.is_active == True,
                theta_field.between(theta_value - 0.5, theta_value + 0.5)
            )
        ).limit(limit)
        
        result = await session.exec(statement)
        return result.all()
    
    async def update_preferences(
        self,
        session: AsyncSession,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> Optional[User]:
        """Update user preferences."""
        user = await self.get(session, user_id)
        if not user:
            return None
        
        current_preferences = user.preferences or {}
        current_preferences.update(preferences)
        
        return await self.update(
            session, 
            db_obj=user, 
            obj_in={"preferences": current_preferences}
        )
    
    async def add_segment(
        self,
        session: AsyncSession,
        user_id: str,
        segment: str
    ) -> Optional[User]:
        """Add a segment to user."""
        user = await self.get(session, user_id)
        if not user:
            return None
        
        segments = user.segments or []
        if segment not in segments:
            segments.append(segment)
            return await self.update(
                session,
                db_obj=user,
                obj_in={"segments": segments}
            )
        return user
    
    async def remove_segment(
        self,
        session: AsyncSession,
        user_id: str,
        segment: str
    ) -> Optional[User]:
        """Remove a segment from user."""
        user = await self.get(session, user_id)
        if not user:
            return None
        
        segments = user.segments or []
        if segment in segments:
            segments.remove(segment)
            return await self.update(
                session,
                db_obj=user,
                obj_in={"segments": segments}
            )
        return user


# Create repository instance
user_repository = UserRepository()