import databases
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Database URL - asyncpg için async URL'e çevir
DATABASE_URL = settings.database_url
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# SQLAlchemy Async Engine
engine = create_async_engine(ASYNC_DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

# Databases (async)
database = databases.Database(DATABASE_URL)


async def get_database():
    """Async database connection"""
    return database


async def get_async_session():
    """Async session generator"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Database connection events
async def connect_to_db():
    """Connect to database on startup"""
    await database.connect()
    print("✅ Database connected")


async def close_db_connection():
    """Close database connection on shutdown"""
    await database.disconnect()
    print("❌ Database disconnected")