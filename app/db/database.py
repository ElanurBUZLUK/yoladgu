from typing import Optional

from app.core.config import settings, is_development
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import sessionmaker

# Import Base from models to avoid circular imports
from app.db.models import Base

# Synchronous engine (for backward compatibility)
# Remove echo=True for production - security risk!
if settings.DATABASE_URL is None:
    raise ValueError(
        "DATABASE_URL is not configured. Please check your environment variables."
    )

sync_engine = create_engine(
    settings.DATABASE_URL,
    echo=is_development(),  # Only echo in development
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

# Export engine for instrumentation
engine = sync_engine

# Async engine (recommended for FastAPI) - initialized lazily
async_engine: Optional[AsyncEngine] = None
AsyncSessionLocal: Optional[async_sessionmaker[AsyncSession]] = None


def init_async_db():
    """Initialize async database engine and session maker"""
    global async_engine, AsyncSessionLocal

    if async_engine is None:
        if settings.DATABASE_URL is None:
            raise ValueError(
                "DATABASE_URL is not configured. Please check your environment variables."
            )

        async_database_url = settings.DATABASE_URL.replace(
            "postgresql://", "postgresql+asyncpg://"
        )
        async_engine = create_async_engine(
            async_database_url,
            echo=is_development(),  # Only echo in development
            pool_size=settings.ASYNCPG_MIN_SIZE,
            max_overflow=settings.ASYNCPG_MAX_SIZE - settings.ASYNCPG_MIN_SIZE,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        AsyncSessionLocal = async_sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )


# Synchronous database session (for backward compatibility)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Async database session (recommended for FastAPI)
async def get_async_db():
    if AsyncSessionLocal is None:
        init_async_db()

    if AsyncSessionLocal is None:
        raise RuntimeError("Failed to initialize async database session")

    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
