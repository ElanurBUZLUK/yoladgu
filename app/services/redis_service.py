"""
Redis Service for Centralized Redis Client Management
Handles caching, streams, and session management
"""

import json
from typing import Any, Dict, List, Optional

import redis
import structlog
from app.core.config import settings

logger = structlog.get_logger()


class RedisService:
    """Singleton service for Redis operations"""

    _instance = None
    _client = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(RedisService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Redis connection (only once)"""
        if self._initialized:
            return

        try:
            # Create Redis client with connection pooling
            self._client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Test connection
            self._client.ping()
            logger.info(
                "redis_connection_established",
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
            )
        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
            self._client = None

        self._initialized = True

    @property
    def client(self) -> Optional[redis.Redis]:
        """Get the Redis client instance"""
        return self._client

    def is_available(self) -> bool:
        """Check if Redis is available"""
        if not self._client:
            return False
        try:
            self._client.ping()
            return True
        except:
            return False

    def close(self):
        """Close Redis connection"""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("redis_connection_closed")

    # Cache Operations
    def cache_set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set cache value with TTL"""
        if not self._client:
            logger.warning("redis_not_available", operation="cache_set")
            return False

        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            self._client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error("cache_set_error", key=key, error=str(e))
            return False

    def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        if not self._client:
            logger.warning("redis_not_available", operation="cache_get")
            return None

        try:
            value = self._client.get(key)
            if value is None:
                return None

            # Try to parse as JSON
            try:
                # Ensure value is a string for json.loads
                str_value = str(value) if value is not None else None
                if str_value:
                    return json.loads(str_value)
                return value
            except json.JSONDecodeError:
                return value
        except Exception as e:
            logger.error("cache_get_error", key=key, error=str(e))
            return None

    def cache_delete(self, key: str) -> bool:
        """Delete cache key"""
        if not self._client:
            logger.warning("redis_not_available", operation="cache_delete")
            return False

        try:
            self._client.delete(key)
            return True
        except Exception as e:
            logger.error("cache_delete_error", key=key, error=str(e))
            return False

    def cache_exists(self, key: str) -> bool:
        """Check if cache key exists"""
        if not self._client:
            return False

        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.error("cache_exists_error", key=key, error=str(e))
            return False

    # Stream Operations
    def stream_add(self, stream_name: str, data: Dict[str, Any]) -> Optional[str]:
        """Add message to Redis stream"""
        if not self._client:
            logger.warning("redis_not_available", operation="stream_add")
            return None

        try:
            # Convert complex objects to JSON strings
            stream_data = {}
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    stream_data[k] = json.dumps(v)
                else:
                    stream_data[k] = str(v)

            message_id = self._client.xadd(stream_name, stream_data)
            logger.debug(
                "stream_message_added", stream=stream_name, message_id=message_id
            )
            return str(message_id) if message_id else None
        except Exception as e:
            logger.error("stream_add_error", stream=stream_name, error=str(e))
            return None

    def stream_read(
        self, stream_name: str, count: int = 10, block: int = 1000
    ) -> List[Dict]:
        """Read messages from Redis stream"""
        if not self._client:
            logger.warning("redis_not_available", operation="stream_read")
            return []

        try:
            messages = self._client.xread({stream_name: "$"}, count=count, block=block)
            result = []

            # Ensure messages is iterable and handle different response types
            if not messages or not hasattr(messages, "__iter__"):
                return result

            for stream, msgs in messages:  # type: ignore
                for msg_id, fields in msgs:
                    # Parse JSON fields back to objects
                    parsed_fields = {}
                    for k, v in fields.items():
                        try:
                            parsed_fields[k] = json.loads(v)
                        except json.JSONDecodeError:
                            parsed_fields[k] = v

                    result.append(
                        {"id": msg_id, "stream": stream, "data": parsed_fields}
                    )

            return result
        except Exception as e:
            logger.error("stream_read_error", stream=stream_name, error=str(e))
            return []

    def stream_length(self, stream_name: str) -> int:
        """Get stream length"""
        if not self._client:
            return 0

        try:
            length = self._client.xlen(stream_name)
            # Handle different response types
            if hasattr(length, "__int__"):
                return int(length)  # type: ignore
            elif isinstance(length, (str, bytes)):
                return int(length)  # type: ignore
            elif length is not None:
                return length  # type: ignore
            else:
                return 0
        except Exception as e:
            logger.error("stream_length_error", stream=stream_name, error=str(e))
            return 0

    # Session Operations (for user sessions)
    def session_set(
        self, session_id: str, data: Dict[str, Any], ttl: int = 86400
    ) -> bool:
        """Set session data"""
        return self.cache_set(f"session:{session_id}", data, ttl)

    def session_get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        return self.cache_get(f"session:{session_id}")

    def session_delete(self, session_id: str) -> bool:
        """Delete session"""
        return self.cache_delete(f"session:{session_id}")

    # Health Check
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive Redis health check"""
        if not self._client:
            return {"status": "unhealthy", "error": "Redis client not initialized"}

        try:
            # Basic ping
            ping_result = self._client.ping()

            # Memory info
            memory_info = self._client.info("memory")
            if not isinstance(memory_info, dict):
                memory_info = {}

            # Connection info
            client_info = self._client.info("clients")
            if not isinstance(client_info, dict):
                client_info = {}

            return {
                "status": "healthy",
                "ping": ping_result,
                "memory_used": memory_info.get("used_memory_human", "unknown"),
                "connected_clients": client_info.get("connected_clients", 0),
                "max_memory": memory_info.get("maxmemory_human", "unlimited"),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Global Redis service instance
redis_service = RedisService()
