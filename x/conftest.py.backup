import pytest
import asyncio
import os
import tempfile
import shutil
from typing import AsyncGenerator, Generator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from alembic.config import Config
from alembic import command
import structlog
from app.core.config import settings
from app.core.database import get_db, Base
from app.models.user import User
from app.models.question import Question
from app.models.student_attempt import StudentAttempt
from app.models.error_pattern import ErrorPattern
from app.models.math_profile import MathProfile
from app.models.spaced_repetition import SpacedRepetition
from app.models.selection_history import SelectionHistory
from app.models.pdf_upload import PDFUpload
from app.models.math_error_detail import MathErrorDetail
from app.services.user_service import UserService
from app.services.question_service import QuestionService
from app.services.sample_data_service import SampleDataService

logger = structlog.get_logger()

# Test configuration
TEST_DATABASE_URL = "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_adaptive_learning"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    # Create test database URL
    test_db_url = os.getenv("TEST_DATABASE_URL", TEST_DATABASE_URL)
    
    # Create engine with test configuration
    engine = create_async_engine(
        test_db_url,
        poolclass=StaticPool,
        echo=False,
        future=True
    )
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest.fixture(scope="session")
async def test_session_factory(test_engine):
    """Create test session factory."""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    return async_session


@pytest.fixture(scope="session")
async def setup_test_database(test_engine):
    """Setup test database with migrations."""
    try:
        # Create all tables
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Run Alembic migrations
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL.replace("+asyncpg", ""))
        
        # Upgrade to head
        command.upgrade(alembic_cfg, "head")
        
        logger.info("Test database setup completed")
        yield
        
    except Exception as e:
        logger.error(f"Error setting up test database: {e}")
        raise
    finally:
        # Cleanup tables
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db_session(test_session_factory, setup_test_database) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session with transaction rollback."""
    async with test_session_factory() as session:
        # Start transaction
        transaction = await session.begin_nested()
        
        try:
            yield session
        finally:
            # Rollback transaction
            await transaction.rollback()
            await session.close()


@pytest.fixture
async def db_session_commit(test_session_factory, setup_test_database) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session with commit (for tests that need persistence)."""
    async with test_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


@pytest.fixture
def override_get_db(db_session):
    """Override the get_db dependency for testing."""
    async def _override_get_db():
        yield db_session
    
    return _override_get_db


@pytest.fixture
async def test_user(db_session) -> User:
    """Create a test user."""
    user_service = UserService()
    
    user_data = {
        "email": "testuser@example.com",
        "password": "testpass123",
        "full_name": "Test User",
        "role": "student",
        "is_active": True
    }
    
    user = await user_service.create_user(db_session, user_data)
    return user


@pytest.fixture
async def test_admin_user(db_session) -> User:
    """Create a test admin user."""
    user_service = UserService()
    
    user_data = {
        "email": "admin@example.com",
        "password": "adminpass123",
        "full_name": "Admin User",
        "role": "admin",
        "is_active": True
    }
    
    user = await user_service.create_user(db_session, user_data)
    return user


@pytest.fixture
async def test_teacher_user(db_session) -> User:
    """Create a test teacher user."""
    user_service = UserService()
    
    user_data = {
        "email": "teacher@example.com",
        "password": "teacherpass123",
        "full_name": "Teacher User",
        "role": "teacher",
        "is_active": True
    }
    
    user = await user_service.create_user(db_session, user_data)
    return user


@pytest.fixture
async def sample_questions(db_session) -> list[Question]:
    """Create sample questions for testing."""
    question_service = QuestionService()
    
    questions_data = [
        {
            "content": "What is 2 + 2?",
            "answer": "4",
            "subject": "math",
            "topic": "addition",
            "difficulty_level": 1,
            "estimated_difficulty": 1.0,
            "is_active": True
        },
        {
            "content": "Solve: 3x + 5 = 14",
            "answer": "x = 3",
            "subject": "math",
            "topic": "algebra",
            "difficulty_level": 2,
            "estimated_difficulty": 2.0,
            "is_active": True
        },
        {
            "content": "What is the capital of France?",
            "answer": "Paris",
            "subject": "geography",
            "topic": "capitals",
            "difficulty_level": 1,
            "estimated_difficulty": 1.0,
            "is_active": True
        }
    ]
    
    questions = []
    for q_data in questions_data:
        question = await question_service.create_question(db_session, q_data)
        questions.append(question)
    
    return questions


@pytest.fixture
async def sample_student_attempts(db_session, test_user, sample_questions) -> list[StudentAttempt]:
    """Create sample student attempts for testing."""
    attempts = []
    
    for i, question in enumerate(sample_questions):
        attempt_data = {
            "user_id": test_user.id,
            "question_id": question.id,
            "user_answer": f"Answer {i}",
            "is_correct": i % 2 == 0,  # Alternate correct/incorrect
            "time_spent": 30.0 + i * 10,
            "confidence_level": 0.7 + i * 0.1
        }
        
        attempt = StudentAttempt(**attempt_data)
        db_session.add(attempt)
        attempts.append(attempt)
    
    await db_session.commit()
    return attempts


@pytest.fixture
async def sample_math_profile(db_session, test_user) -> MathProfile:
    """Create a sample math profile for testing."""
    profile_data = {
        "user_id": test_user.id,
        "current_level": 2,
        "strength_areas": ["algebra", "geometry"],
        "weakness_areas": ["calculus"],
        "learning_style": "visual",
        "preferred_difficulty": 2.0,
        "total_questions_attempted": 50,
        "correct_answers": 35,
        "average_time_per_question": 45.0
    }
    
    profile = MathProfile(**profile_data)
    db_session.add(profile)
    await db_session.commit()
    return profile


@pytest.fixture
async def sample_error_patterns(db_session) -> list[ErrorPattern]:
    """Create sample error patterns for testing."""
    patterns_data = [
        {
            "pattern_name": "Addition Error",
            "pattern_description": "Common addition mistakes",
            "subject": "math",
            "topic": "addition",
            "difficulty_level": 1,
            "is_active": True
        },
        {
            "pattern_name": "Algebra Error",
            "pattern_description": "Common algebra mistakes",
            "subject": "math",
            "topic": "algebra",
            "difficulty_level": 2,
            "is_active": True
        }
    ]
    
    patterns = []
    for p_data in patterns_data:
        pattern = ErrorPattern(**p_data)
        db_session.add(pattern)
        patterns.append(pattern)
    
    await db_session.commit()
    return patterns


@pytest.fixture
async def temp_upload_dir():
    """Create a temporary upload directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
async def sample_pdf_file(temp_upload_dir):
    """Create a sample PDF file for testing."""
    pdf_path = os.path.join(temp_upload_dir, "test.pdf")
    
    # Create a minimal PDF file for testing
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test PDF) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF\n")
    
    return pdf_path


@pytest.fixture
async def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "content": "This is a mock LLM response for testing purposes.",
        "model": "gpt-3.5-turbo",
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "total_tokens": 75
        },
        "cost": 0.001
    }


@pytest.fixture
async def mock_embedding_response():
    """Mock embedding response for testing."""
    return {
        "embedding": [0.1] * 1536,  # 1536-dimensional embedding
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 10,
            "total_tokens": 10
        }
    }


@pytest.fixture
def test_app_settings():
    """Test application settings."""
    return {
        "environment": "testing",
        "debug": True,
        "database_url": TEST_DATABASE_URL,
        "secret_key": "test-secret-key",
        "jwt_secret": "test-jwt-secret",
        "encryption_key": "test-encryption-key",
        "rate_limit_enabled": False,  # Disable rate limiting for tests
        "prometheus_enabled": False,  # Disable metrics for tests
        "content_moderation_enabled": False,  # Disable moderation for tests
        "cost_monitoring_enabled": False  # Disable cost monitoring for tests
    }


@pytest.fixture
async def clean_test_data(db_session):
    """Clean test data after each test."""
    yield
    
    # Clean up all test data
    tables = [
        StudentAttempt,
        SelectionHistory,
        SpacedRepetition,
        MathErrorDetail,
        PDFUpload,
        MathProfile,
        ErrorPattern,
        Question,
        User
    ]
    
    for table in tables:
        await db_session.execute(table.__table__.delete())
    
    await db_session.commit()


# Helper functions for testing
async def create_test_user_with_token(db_session, email: str = "test@example.com", role: str = "student"):
    """Create a test user and return with access token."""
    from app.core.security import create_access_token
    
    user_service = UserService()
    user_data = {
        "email": email,
        "password": "testpass123",
        "full_name": "Test User",
        "role": role,
        "is_active": True
    }
    
    user = await user_service.create_user(db_session, user_data)
    token = create_access_token(data={"sub": user.email})
    
    return user, token


async def create_test_question(db_session, **kwargs):
    """Create a test question with default values."""
    default_data = {
        "content": "Test question content",
        "answer": "Test answer",
        "subject": "math",
        "topic": "test",
        "difficulty_level": 1,
        "estimated_difficulty": 1.0,
        "is_active": True
    }
    default_data.update(kwargs)
    
    question = Question(**default_data)
    db_session.add(question)
    await db_session.commit()
    await db_session.refresh(question)
    return question


async def create_test_attempt(db_session, user_id: int, question_id: int, **kwargs):
    """Create a test student attempt with default values."""
    default_data = {
        "user_id": user_id,
        "question_id": question_id,
        "user_answer": "Test answer",
        "is_correct": True,
        "time_spent": 30.0,
        "confidence_level": 0.7
    }
    default_data.update(kwargs)
    
    attempt = StudentAttempt(**default_data)
    db_session.add(attempt)
    await db_session.commit()
    await db_session.refresh(attempt)
    return attempt


# Test configuration
def pytest_configure(config):
    """Configure pytest."""
    # Set test environment variables
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["TEST_DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["RATE_LIMIT_ENABLED"] = "false"
    os.environ["PROMETHEUS_ENABLED"] = "false"
    os.environ["CONTENT_MODERATION_ENABLED"] = "false"
    os.environ["COST_MONITORING_ENABLED"] = "false"


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Mark tests that need database
    for item in items:
        if "db_session" in item.fixturenames:
            item.add_marker(pytest.mark.asyncio)
            item.add_marker(pytest.mark.database)
        
        if "test_engine" in item.fixturenames:
            item.add_marker(pytest.mark.asyncio)
            item.add_marker(pytest.mark.integration)