"""
Database session management and dependency injection.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from app.db.base import get_async_session
from app.db.repositories.user import user_repository
from app.db.repositories.item import math_item_repository, english_item_repository
from app.db.repositories.attempt import attempt_repository


class DatabaseManager:
    """Database manager with repository access."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.users = user_repository
        self.math_items = math_item_repository
        self.english_items = english_item_repository
        self.attempts = attempt_repository


async def get_db_manager(
    session: AsyncSession = Depends(get_async_session)
) -> DatabaseManager:
    """Get database manager with repositories."""
    return DatabaseManager(session)


# Convenience function for getting just the session
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for direct use."""
    async for session in get_async_session():
        yield session


# Alias for FastAPI dependency injection
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for FastAPI dependency injection."""
    async for session in get_async_session():
        yield session