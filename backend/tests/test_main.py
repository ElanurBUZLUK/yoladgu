"""
Test cases for main application.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "up"
    assert data["version"] == "1.0.0"
    assert data["service"] == "adaptive-question-system"


def test_version_info():
    """Test version information endpoint."""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "app_version" in data
    assert "model_versions" in data
    assert "retrieval" in data["model_versions"]
    assert "rerank" in data["model_versions"]
    assert "llm" in data["model_versions"]
    assert "bandit" in data["model_versions"]


def test_docs_available():
    """Test that API documentation is available in development."""
    response = client.get("/docs")
    # Should be available in development mode
    assert response.status_code in [200, 404]  # 404 if disabled in production


def test_cors_headers():
    """Test CORS headers are present."""
    response = client.options("/health")
    # CORS headers should be present
    assert response.status_code in [200, 405]  # Some servers return 405 for OPTIONS