import pytest
from fastapi.testclient import TestClient
from app.main import app
import redis
import time
from unittest.mock import patch

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Question Recommendation System" in data["message"]

def test_auth_endpoints():
    # Test login endpoint
    login_data = {
        "username": "testuser",
        "password": "testpass"
    }
    response = client.post("/api/v1/auth/login", data=login_data)
    # Should return 401 for invalid credentials
    assert response.status_code in [401, 422]

def test_users_endpoints():
    # Test get users endpoint
    response = client.get("/api/v1/users/")
    assert response.status_code in [200, 401]  # 401 if authentication required

def test_questions_endpoints():
    # Test get questions endpoint
    response = client.get("/api/v1/questions/")
    assert response.status_code in [200, 401]  # 401 if authentication required

def test_solutions_endpoints():
    # Test get solutions endpoint
    response = client.get("/api/v1/solutions/")
    assert response.status_code in [200, 401]  # 401 if authentication required

def test_study_plans_endpoints():
    # Test get study plans endpoint
    response = client.get("/api/v1/study-plans/")
    assert response.status_code in [200, 401]  # 401 if authentication required

def test_topics_endpoints():
    # Test get topics endpoint
    response = client.get("/api/v1/topics/")
    assert response.status_code in [200, 401]  # 401 if authentication required

def test_subjects_endpoints():
    # Test get subjects endpoint
    response = client.get("/api/v1/subjects/")
    assert response.status_code in [200, 401]  # 401 if authentication required

def test_plan_items_endpoints():
    # Test get plan items endpoint
    response = client.get("/api/v1/plan-items/")
    assert response.status_code in [200, 401]  # 401 if authentication required 

def test_submit_answer_stream(monkeypatch):
    # Redis XADD mock
    events = []
    class DummyRedis:
        def xadd(self, stream, event):
            events.append((stream, event))
    monkeypatch.setattr(redis, "Redis", lambda *a, **kw: DummyRedis())
    # Login ve örnek kullanıcı oluşturma burada atlanıyor (örnek için varsayalım)
    # Cevap gönder
    response = client.post(
        "/api/v1/questions/1/answer",
        data={"answer": "A", "confidence_level": 5, "feedback": "Zor bir soruydu"},
        headers={"Authorization": "Bearer testtoken"}
    )
    assert response.status_code in (200, 403, 404)  # Kullanıcı/oturum yoksa 403/404 olabilir
    # Event Redis'e yazıldı mı?
    assert any(stream == "student_responses_stream" for stream, _ in events) 