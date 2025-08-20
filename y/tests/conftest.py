import pytest
import asyncio
from app.services.cache_service import cache_service

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session", autouse=True)
async def setup_test_environment(event_loop):
    """Fixture to set up the test environment for the session."""
    # Connect to Redis before tests run
    await cache_service.connect()
    print("\n✅ Redis client connected for test session.")
    
    yield
    
    # Disconnect from Redis after tests are done
    await cache_service.close()
    print("\n❌ Redis client disconnected for test session.")

