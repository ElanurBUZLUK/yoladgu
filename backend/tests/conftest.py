"""
Test configuration and fixtures.
"""

import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from app.core.config import settings


# Test database URL (use in-memory SQLite for tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    future=True
)

# Create test session factory
TestSessionLocal = sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def setup_database():
    """Set up test database."""
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest.fixture
async def db_session(setup_database) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async with TestSessionLocal() as session:
        yield session
        await session.rollback()


@pytest.fixture
def test_user_data():
    """Test user data fixture."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "hashed_password": "hashed_password_123",
        "tenant_id": "test_tenant",
        "lang": "tr",
        "role": "student",
        "grade": "9"
    }


@pytest.fixture
def test_math_item_data():
    """Test math item data fixture."""
    return {
        "tenant_id": "test_tenant",
        "stem": "Solve the equation: 2x + 5 = 13",
        "answer_key": "x=4",
        "skills": ["linear_equation", "algebra"],
        "bloom_level": "apply",
        "difficulty_a": 1.0,
        "difficulty_b": 0.2,
        "lang": "tr",
        "status": "active"
    }


@pytest.fixture
def test_english_item_data():
    """Test English item data fixture."""
    return {
        "tenant_id": "test_tenant",
        "passage": "I am going __ the library to study.",
        "blanks": [
            {
                "span": "__",
                "answer": "to",
                "distractors": ["at", "in", "on"],
                "skill_tag": "prepositions"
            }
        ],
        "level_cefr": "A2",
        "error_tags": ["prepositions"],
        "lang": "en",
        "status": "active"
    }


@pytest.fixture
def test_attempt_data():
    """Test attempt data fixture."""
    return {
        "answer": "x=4",
        "correct": True,
        "time_ms": 15000,
        "hints_used": 1,
        "item_type": "math",
        "context": {
            "device": "mobile",
            "session_id": "test_session_123"
        }
    }