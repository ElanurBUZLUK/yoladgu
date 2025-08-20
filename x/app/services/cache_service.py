from __future__ import annotations
import hashlib
import json
from typing import Any, Optional
import structlog
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError

from app.core.config import settings

logger = structlog.get_logger(__name__)

class SemanticCache:
    """
    Asynchronous prompt-to-response semantic cache using Redis.
    Manages its own connection lifecycle.
    """
    _client: Optional[Redis] = None
    _pool: Optional[ConnectionPool] = None

    def __init__(self, redis_url: str = settings.redis_url, namespace: str = "semcache"):
        """
        Initializes the cache service.

        Args:
            redis_url: The URL for the Redis instance.
            namespace: The namespace to use for cache keys in Redis.
        """
        self.redis_url = redis_url
        self.ns = namespace
        logger.info("SemanticCache initialized", redis_url=self.redis_url, namespace=self.ns)

    async def connect(self):
        """
        Establishes the connection to Redis and creates a connection pool.
        This method should be called during application startup.
        """
        if self._pool is None:
            try:
                logger.info("Creating Redis connection pool...")
                self._pool = ConnectionPool.from_url(self.redis_url, max_connections=20)
                self._client = Redis(connection_pool=self._pool)
                await self._client.ping()
                logger.info("Redis connection pool established successfully.")
            except RedisError as e:
                logger.error("Failed to connect to Redis", error=str(e))
                self._pool = None
                self._client = None
                raise

    async def close(self):
        """
        Closes the Redis connection and disconnects the pool.
        This method should be called during application shutdown.
        """
        if self._client:
            logger.info("Closing Redis connection...")
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
            logger.info("Redis connection pool disconnected.")

    async def ping(self) -> bool:
        """
        Checks the connection to Redis by sending a PING command.

        Returns:
            True if the connection is alive, False otherwise.
        """
        if not self.client:
            return False
        try:
            return await self.client.ping()
        except RedisError as e:
            logger.error("Redis ping failed", error=str(e))
            return False

    @property
    def client(self) -> Redis:
        """
        Provides access to the Redis client.

        Raises:
            RuntimeError: If the client is not connected.

        Returns:
            The asynchronous Redis client instance.
        """
        if self._client is None:
            raise RuntimeError("Redis client is not connected. Call connect() during application startup.")
        return self._client

    @staticmethod
    def _key_for(prompt: str) -> str:
        """Generates a SHA256 hash for a given prompt to use as a cache key."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _redis_key(self, key: str) -> str:
        """Constructs the full Redis key with the namespace."""
        return f"{self.ns}:{key}"

    async def get(self, prompt: str) -> Optional[dict[str, Any]]:
        """
        Retrieves a cached response for a given prompt.

        Args:
            prompt: The prompt to look up in the cache.

        Returns:
            The cached dictionary object if found, otherwise None.
        """
        cache_key = self._redis_key(self._key_for(prompt))
        try:
            cached_data = await self.client.get(cache_key)
            if not cached_data:
                return None
            
            return json.loads(cached_data)
        except json.JSONDecodeError:
            logger.warning("Failed to decode cached JSON data", key=cache_key)
            return None
        except RedisError as e:
            logger.error("Redis GET command failed", key=cache_key, error=str(e))
            return None

    async def set(self, prompt: str, value: dict, ttl: int = 3600) -> None:
        """
        Caches a response for a given prompt with a specified TTL.

        Args:
            prompt: The prompt to cache the response for.
            value: The dictionary object to cache.
            ttl: The time-to-live for the cache entry in seconds.
        """
        cache_key = self._redis_key(self._key_for(prompt))
        try:
            await self.client.set(cache_key, json.dumps(value, ensure_ascii=False), ex=ttl)
        except RedisError as e:
            logger.error("Redis SET command failed", key=cache_key, error=str(e))

# A single, shared instance of the cache service
cache_service = SemanticCache()
