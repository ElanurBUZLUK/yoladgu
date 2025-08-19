import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from redis.asyncio import Redis
from app.utils.distlock_idem import idempotent_singleflight, IdempotencyConfig

@pytest.fixture
def mock_redis_client():
    client = AsyncMock(spec=Redis)
    client.get.return_value = None
    client.set.return_value = True # Simulate successful lock acquisition
    client.delete.return_value = 1
    return client

@pytest.mark.asyncio
async def test_idempotent_singleflight_first_call_success(mock_redis_client):
    config = IdempotencyConfig()
    worker_mock = AsyncMock(return_value={"status": "processed", "data": "result"})
    
    result = await idempotent_singleflight(mock_redis_client, "test_key", config, worker_mock)
    
    worker_mock.assert_called_once()
    mock_redis_client.set.assert_called_once()
    assert result == {"status": "processed", "data": "result"}
    mock_redis_client.delete.assert_not_called()

@pytest.mark.asyncio
async def test_idempotent_singleflight_second_call_cached(mock_redis_client):
    config = IdempotencyConfig()
    worker_mock = AsyncMock(return_value={"status": "processed", "data": "result"})
    
    # Simulate first call completing and caching result
    mock_redis_client.get.return_value = b'{"status": "done", "result": {"status": "processed", "data": "cached_result"}}'
    
    result = await idempotent_singleflight(mock_redis_client, "test_key", config, worker_mock)
    
    worker_mock.assert_not_called()
    mock_redis_client.set.assert_not_called()
    assert result == {"status": "processed", "data": "cached_result"}

@pytest.mark.asyncio
async def test_idempotent_singleflight_in_progress_wait_and_succeed(mock_redis_client):
    config = IdempotencyConfig(poll_interval_ms=10, wait_timeout_ms=100)
    worker_mock = AsyncMock(return_value={"status": "processed", "data": "result"})
    
    # Simulate another process holding the lock initially
    mock_redis_client.set.return_value = False # First set fails
    
    # Simulate the other process completing after some time
    async def get_side_effect(key):
        await asyncio.sleep(0.05) # Simulate delay
        return b'{"status": "done", "result": {"status": "processed", "data": "polled_result"}}'
    
    mock_redis_client.get.side_effect = get_side_effect
    
    result = await idempotent_singleflight(mock_redis_client, "test_key", config, worker_mock)
    
    worker_mock.assert_not_called()
    assert result == {"status": "processed", "data": "polled_result"}
    mock_redis_client.set.assert_called_once() # Initial failed set

@pytest.mark.asyncio
async def test_idempotent_singleflight_worker_failure(mock_redis_client):
    config = IdempotencyConfig()
    worker_mock = AsyncMock(side_effect=ValueError("Worker failed"))
    
    with pytest.raises(ValueError, match="Worker failed"):
        await idempotent_singleflight(mock_redis_client, "test_key", config, worker_mock)
    
    worker_mock.assert_called_once()
    mock_redis_client.delete.assert_called_once() # Should delete the in_progress key

@pytest.mark.asyncio
async def test_idempotent_singleflight_timeout(mock_redis_client):
    config = IdempotencyConfig(poll_interval_ms=10, wait_timeout_ms=50)
    worker_mock = AsyncMock()
    
    # Simulate lock always held and never done
    mock_redis_client.set.return_value = False # Initial set fails
    mock_redis_client.get.return_value = b'{"status": "in_progress", "token": "some_token"}'
    
    with pytest.raises(asyncio.TimeoutError, match="Idempotent operation still in progress"):
        await idempotent_singleflight(mock_redis_client, "test_key", config, worker_mock)
    
    worker_mock.assert_not_called()
    mock_redis_client.set.assert_called_once()
