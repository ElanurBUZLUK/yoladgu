import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app
from app.models.user import User
from app.models.question import QuestionType

# Mock user data
mock_user = User(id=1, username="testuser", email="test@example.com", user_level=1, xp=0, streaks=0)

# Create a TestClient instance
client = TestClient(app)

# Mock the get_current_teacher dependency
def override_get_current_teacher():
    return mock_user

# Mock the get_current_student dependency
def override_get_current_student():
    return mock_user

# Apply the dependency overrides
app.dependency_overrides[app.middleware.auth.get_current_teacher] = override_get_current_teacher
app.dependency_overrides[app.middleware.auth.get_current_student] = override_get_current_student

@pytest.fixture
def mock_llm_gateway():
    with patch("app.services.llm_gateway.generate_json", new_callable=AsyncMock) as mock_generate_json:
        yield mock_generate_json

def test_health():
    response = client.get("/api/v1/math/rag/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "module": "math-rag"}

def test_generate_questions(mock_llm_gateway):
    mock_llm_gateway.return_value.success = True
    mock_llm_gateway.return_value.parsed_json = {
        "items": [
            {
                "kind": "MULTIPLE_CHOICE",
                "stem": "What is 2+2?",
                "options": ["3", "4", "5"],
                "answer": {"value": "4"},
                "meta": {}
            }
        ],
        "usage": {"total_tokens": 100}
    }

    response = client.post(
        "/api/v1/math/rag/generate",
        json={"topic": "addition", "difficulty_level": 1, "question_type": "MULTIPLE_CHOICE", "n": 1}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["stem"] == "What is 2+2?"
    assert data["usage"] == {"total_tokens": 100}

def test_solve_problem(mock_llm_gateway):
    mock_llm_gateway.return_value.success = True
    mock_llm_gateway.return_value.parsed_json = {
        "solution": "The answer is 4.",
        "steps": "Step 1: 2+2=4",
        "usage": {"total_tokens": 50}
    }

    response = client.post(
        "/api/v1/math/rag/solve",
        json={"problem": "What is 2+2?", "show_steps": True}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["solution"] == "The answer is 4."
    assert data["steps"] == "Step 1: 2+2=4"
    assert data["usage"] == {"total_tokens": 50}

def test_check_answer(mock_llm_gateway):
    mock_llm_gateway.return_value.success = True
    mock_llm_gateway.return_value.parsed_json = {
        "correct": True,
        "explanation": "The answer is indeed 4.",
        "usage": {"total_tokens": 75}
    }

    response = client.post(
        "/api/v1/math/rag/check",
        json={"question": "What is 2+2?", "user_answer": "4", "answer_key": "4", "require_explanation": True}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["correct"] is True
    assert data["explanation"] == "The answer is indeed 4."
    assert data["usage"] == {"total_tokens": 75}

def test_get_hint(mock_llm_gateway):
    mock_llm_gateway.return_value.success = True
    mock_llm_gateway.return_value.parsed_json = {
        "hint": "Try adding the two numbers together.",
        "usage": {"total_tokens": 25}
    }

    response = client.post(
        "/api/v1/math/rag/hint",
        json={"problem": "What is 2+2?"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["hint"] == "Try adding the two numbers together."
    assert data["usage"] == {"total_tokens": 25}