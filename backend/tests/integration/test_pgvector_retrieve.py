import pytest
import asyncio
from httpx import AsyncClient
from app.main import app # Assuming your FastAPI app instance is named 'app'
from app.core.config import settings

# Adjust settings for testing if necessary
settings.environment = "testing"
settings.redis_url = "redis://localhost:6379/1" # Use a separate Redis DB for tests

@pytest.fixture(scope="module")
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.mark.asyncio
async def test_pgvector_retrieve_integration(client):
    # This test assumes you have some data with embeddings in your test DB
    # and that the vector_index_manager is correctly configured.
    # You might need to populate test data and embeddings before running this.
    
    # Example: Retrieve from 'questions' table with a dummy query
    retrieve_request = {
        "query": "What is the capital of France?",
        "namespace": "default",
        "slot": "active",
        "k": 1
    }
    
    response = await client.post("/api/v1/english/rag/retrieve", json=retrieve_request)
    
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)
    # Add more assertions based on expected retrieval results
    
    # Example: Check if items have expected fields
    if data["items"]:
        item = data["items"][0]
        assert "obj_ref" in item
        assert "meta" in item
        assert "dist" in item
