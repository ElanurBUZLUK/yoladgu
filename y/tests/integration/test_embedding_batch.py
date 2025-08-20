import pytest
import asyncio
from httpx import AsyncClient
from app.main import app
from app.core.config import settings

# Adjust settings for testing if necessary
settings.environment = "testing"
settings.redis_url = "redis://localhost:6379/1" # Use a separate Redis DB for tests

@pytest.fixture(scope="module")
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_embedding_batch_integration(client):
    # This test assumes you have a way to generate dummy items for upsert
    # and that the vector_index_manager.batch_upsert_embeddings works.
    
    batch_upsert_request = {
        "items": [
            {"id": "test_id_1", "text": "embedding text one", "meta": {"source": "test"}},
            {"id": "test_id_2", "text": "embedding text two", "meta": {"source": "test"}}
        ],
        "table_name": "questions", # Or 'error_patterns'
        "namespace": "test_namespace"
    }
    
    response = await client.post("/api/v1/vector/batch-upsert", json=batch_upsert_request)
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["processed"] == 2
    assert data["namespace"] == "test_namespace"
    assert "slot" in data
    assert isinstance(data["errors"], list)
    assert len(data["errors"]) == 0

    # You might want to add a cleanup step here to remove the test embeddings
    # or query them to ensure they were inserted correctly.
