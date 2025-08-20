"""
Type stubs for SQLAlchemy ext.asyncio module
"""

from typing import Any, Optional, Callable
from .. import AsyncSession, AsyncEngine

def create_async_engine(*args: Any, **kwargs: Any) -> AsyncEngine: ...
def async_sessionmaker(*args: Any, **kwargs: Any) -> Callable: ...

# Re-export AsyncSession
__all__ = ["AsyncSession", "create_async_engine", "async_sessionmaker"]
