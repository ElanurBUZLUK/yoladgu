import pytest
import asyncio
from app.services.cache_service import SemanticCache
from redis.asyncio import Redis

@pytest.fixture
async def test_cache_service():
    # Use a separate DB for testing to avoid conflicts
    cache = SemanticCache(redis_url="redis://localhost:6379/1", namespace="test_semcache")
    await cache.connect()
    yield cache
    await cache.client.flushdb() # Clean up after test
    await cache.close()

@pytest.mark.asyncio
async def test_cache_set_get(test_cache_service):
    key = "test_prompt"
    value = {"answer": "test_answer", "source": "test_source"}
    await test_cache_service.set(key, value, ttl=1)
    retrieved_value = await test_cache_service.get(key)
    assert retrieved_value == value

@pytest.mark.asyncio
async def test_cache_get_miss(test_cache_service):
    key = "non_existent_prompt"
    retrieved_value = await test_cache_service.get(key)
    assert retrieved_value is None

@pytest.mark.asyncio
async def test_cache_ttl(test_cache_service):
    key = "ttl_prompt"
    value = {"data": "short_lived"}
    await test_cache_service.set(key, value, ttl=1) # 1 second TTL
    await asyncio.sleep(1.1) # Wait for TTL to expire
    retrieved_value = await test_cache_service.get(key)
    assert retrieved_value is None

@pytest.mark.asyncio
async def test_cache_ping(test_cache_service):
    assert await test_cache_service.ping()
    # Simulate connection loss (this is hard to do reliably without mocking)
    # For now, just check successful ping
