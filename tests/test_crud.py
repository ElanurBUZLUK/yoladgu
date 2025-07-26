import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.database import get_db
from app.db.models import Base, User, Question, StudentResponse
from app.crud import user as user_crud
from app.crud import question as question_crud

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

def test_create_user(db_session):
    user_data = {
        "email": "test@example.com",
        "username": "testuser",
        "full_name": "Test User"
    }
    user = user_crud.create_user(db_session, user_data)
    assert user.email == "test@example.com"
    assert user.username == "testuser"

def test_get_user(db_session):
    user_data = {
        "email": "test@example.com",
        "username": "testuser",
        "full_name": "Test User"
    }
    created_user = user_crud.create_user(db_session, user_data)
    user = user_crud.get_user(db_session, user_id=created_user.id)
    assert user is not None
    assert user.email == "test@example.com"

def test_create_question(db_session):
    question_data = {
        "content": "What is 2+2?",
        "question_type": "multiple_choice",
        "difficulty_level": 1,
        "subject_id": 1,
        "options": ["3", "4", "5", "6"],
        "correct_answer": "4"
    }
    question = question_crud.create_question(db_session, question_data)
    assert question.content == "What is 2+2?"
    assert question.correct_answer == "4"

def test_get_questions(db_session):
    # Create test questions
    questions_data = [
        {
            "content": "Question 1",
            "question_type": "multiple_choice",
            "difficulty_level": 1,
            "subject_id": 1,
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A"
        },
        {
            "content": "Question 2",
            "question_type": "multiple_choice",
            "difficulty_level": 2,
            "subject_id": 1,
            "options": ["A", "B", "C", "D"],
            "correct_answer": "B"
        }
    ]
    
    for q_data in questions_data:
        question_crud.create_question(db_session, q_data)
    
    questions = question_crud.get_questions(db_session, skip=0, limit=10)
    assert len(questions) == 2 