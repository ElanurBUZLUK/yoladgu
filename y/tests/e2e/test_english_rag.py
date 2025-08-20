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
async def test_english_rag_answer_smoke(client):
    # This is a smoke test. It assumes the system is set up with some data
    # and that LLM calls will succeed.
    
    # You might need to mock LLM calls or ensure a live LLM is available for true E2E
    
    answer_request = {
        "query": "What is the main idea of the text about climate change?",
        "question": "Summarize the key points about climate change.",
        "namespace": "default",
        "slot": "active",
        "k": 3,
        "compress_context": True,
        "max_context": 1600
    }
    
    response = await client.post("/api/v1/english/rag/answer", json=answer_request)
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0
    assert "sources" in data
    assert isinstance(data["sources"], list)

@pytest.mark.asyncio
async def test_english_rag_retrieve_smoke(client):
    retrieve_request = {
        "query": "Tell me about renewable energy sources.",
        "namespace": "default",
        "slot": "active",
        "k": 2
    }
    
    response = await client.post("/api/v1/english/rag/retrieve", json=retrieve_request)
    
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)
    if data["items"]:
        item = data["items"][0]
        assert "obj_ref" in item
        assert "meta" in item
        assert "dist" in item
