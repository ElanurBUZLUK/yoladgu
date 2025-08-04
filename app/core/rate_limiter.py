"""
Advanced Rate Limiting & Cache Management
Embedding API'ları için özelleştirilmiş rate limiting ve cache yönetimi
"""

import hashlib
import json
import time
from typing import Any, Dict, Optional

import redis.asyncio as redis
import structlog
from app.core.config import settings
from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = structlog.get_logger()


class AdvancedRateLimiter:
    """
    Gelişmiş rate limiter
    - Endpoint bazlı farklı limitler
    - User-based quota
    - Cache-aware rate limiting
    - Burst allowance
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.prefix = "rate_limit"

        # Endpoint konfigürasyonları
        self.endpoint_configs = {
            # Embedding endpoints - computationally expensive
            "POST:/api/v1/embeddings/compute": {
                "per_minute": 20,
                "per_hour": 200,
                "per_day": 1000,
                "burst_allowance": 5,
                "priority": "high",
            },
            "POST:/api/v1/embeddings/similarity": {
                "per_minute": 30,
                "per_hour": 300,
                "per_day": 1500,
                "burst_allowance": 10,
                "priority": "medium",
            },
            "POST:/api/v1/embeddings/vector/search": {
                "per_minute": 50,
                "per_hour": 500,
                "per_day": 2000,
                "burst_allowance": 15,
                "priority": "low",  # Çünkü cache'li ve hızlı
            },
            "POST:/api/v1/embeddings/batch/update": {
                "per_minute": 2,
                "per_hour": 10,
                "per_day": 50,
                "burst_allowance": 1,
                "priority": "critical",
            },
            # Scheduler endpoints - admin only
            "POST:/api/v1/scheduler/trigger/*": {
                "per_minute": 5,
                "per_hour": 20,
                "per_day": 100,
                "burst_allowance": 2,
                "priority": "admin",
            },
            # Default limits
            "default": {
                "per_minute": 100,
                "per_hour": 1000,
                "per_day": 10000,
                "burst_allowance": 20,
                "priority": "normal",
            },
        }

        # User type limits
        self.user_type_multipliers = {
            "admin": 10.0,
            "premium": 3.0,
            "teacher": 2.0,
            "student": 1.0,
            "anonymous": 0.5,
        }

    def get_endpoint_config(self, method: str, path: str) -> Dict[str, Any]:
        """Endpoint konfigürasyonunu al"""
        endpoint_key = f"{method}:{path}"

        # Exact match
        if endpoint_key in self.endpoint_configs:
            return self.endpoint_configs[endpoint_key]

        # Pattern match
        for pattern, config in self.endpoint_configs.items():
            if pattern.endswith("*") and endpoint_key.startswith(pattern[:-1]):
                return config

        # Default
        return self.endpoint_configs["default"]

    def get_user_type(self, request: Request) -> str:
        """User type'ını request'ten çıkar"""
        # JWT token'dan user bilgisi
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Token decode et ve user type al
            # Şimdilik basit bir implementasyon
            return "student"  # Default

        return "anonymous"

    def get_rate_limit_key(self, identifier: str, endpoint: str, window: str) -> str:
        """Rate limit key oluştur"""
        return f"{self.prefix}:{identifier}:{endpoint}:{window}"

    def get_cache_key(self, request: Request) -> str:
        """Request için cache key oluştur"""
        # Query params ve body'den key oluştur
        cache_parts = [
            request.method,
            str(request.url.path),
            str(sorted(request.query_params.items())),
        ]

        cache_string = "|".join(cache_parts)
        return f"cache:request:{hashlib.md5(cache_string.encode()).hexdigest()}"

    async def check_rate_limit(self, request: Request) -> Optional[JSONResponse]:
        """Rate limit kontrolü"""
        try:
            # Endpoint config al
            config = self.get_endpoint_config(request.method, request.url.path)

            # User identifier
            user_type = self.get_user_type(request)
            client_ip = request.client.host if request.client else "unknown"
            user_identifier = f"{user_type}:{client_ip}"

            # User type multiplier uygula
            multiplier = self.user_type_multipliers.get(user_type, 1.0)
            adjusted_limits = {
                key: int(value * multiplier) if isinstance(value, int) else value
                for key, value in config.items()
            }

            endpoint_key = f"{request.method}:{request.url.path}"
            current_time = int(time.time())

            # Check different time windows
            windows = [
                ("minute", 60, adjusted_limits["per_minute"]),
                ("hour", 3600, adjusted_limits["per_hour"]),
                ("day", 86400, adjusted_limits["per_day"]),
            ]

            for window_name, window_seconds, limit in windows:
                window_start = current_time - (current_time % window_seconds)
                key = self.get_rate_limit_key(
                    user_identifier, endpoint_key, f"{window_name}:{window_start}"
                )

                # Current count al
                if not self.redis:
                    logger.warning("redis_not_available_for_rate_limit")
                    return None  # Fail open if Redis unavailable

                current_count = await self.redis.get(key)
                current_count = int(current_count) if current_count else 0

                # Burst allowance check
                burst_limit = limit + adjusted_limits["burst_allowance"]

                if current_count >= burst_limit:
                    # Rate limit exceeded
                    retry_after = window_seconds - (current_time % window_seconds)

                    logger.warning(
                        "rate_limit_exceeded",
                        user_identifier=user_identifier,
                        endpoint=endpoint_key,
                        window=window_name,
                        current_count=current_count,
                        limit=limit,
                        burst_limit=burst_limit,
                    )

                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "message": f"Too many requests. Limit: {limit}/{window_name}",
                            "retry_after": retry_after,
                            "current_count": current_count,
                            "limit": limit,
                            "window": window_name,
                        },
                        headers={
                            "Retry-After": str(retry_after),
                            "X-RateLimit-Limit": str(limit),
                            "X-RateLimit-Remaining": str(
                                max(0, burst_limit - current_count - 1)
                            ),
                            "X-RateLimit-Reset": str(window_start + window_seconds),
                        },
                    )

            # Increment counters
            for window_name, window_seconds, limit in windows:
                window_start = current_time - (current_time % window_seconds)
                key = self.get_rate_limit_key(
                    user_identifier, endpoint_key, f"{window_name}:{window_start}"
                )

                # Increment with expiry
                if self.redis:
                    pipe = self.redis.pipeline()
                    pipe.incr(key)
                    pipe.expire(key, window_seconds + 60)  # Buffer for cleanup
                    await pipe.execute()

            # Add rate limit headers
            _remaining = max(0, adjusted_limits["per_minute"] - (current_count + 1))

            return None  # No rate limit hit

        except Exception as e:
            logger.error("rate_limit_check_error", error=str(e))
            return None  # Fail open

    async def get_cached_response(self, request: Request) -> Optional[Response]:
        """Cache'den response al"""
        try:
            # Only cache GET requests and specific POST endpoints
            cacheable_endpoints = [
                "GET:/api/v1/embeddings/stats",
                "POST:/api/v1/embeddings/similarity",
                "POST:/api/v1/embeddings/vector/search",
            ]

            endpoint_key = f"{request.method}:{request.url.path}"
            if endpoint_key not in cacheable_endpoints:
                return None

            cache_key = self.get_cache_key(request)
            if not self.redis:
                return None
            cached_data = await self.redis.get(cache_key)

            if cached_data:
                try:
                    response_data = json.loads(cached_data)

                    # Cache hit headers ekle
                    headers = {
                        "X-Cache": "HIT",
                        "X-Cache-Key": cache_key[-12:],  # Son 12 karakter
                        "Content-Type": "application/json",
                    }

                    logger.debug(
                        "cache_hit", cache_key=cache_key[-12:], endpoint=endpoint_key
                    )

                    return JSONResponse(content=response_data, headers=headers)

                except json.JSONDecodeError:
                    # Invalid cache data, delete it
                    await self.redis.delete(cache_key)

            return None

        except Exception as e:
            logger.error("cache_get_error", error=str(e))
            return None

    async def cache_response(
        self, request: Request, response: Response, ttl: int = 300
    ):
        """Response'u cache'e kaydet"""
        try:
            # Only cache successful responses
            if response.status_code != 200:
                return

            # Cacheable endpoints
            cacheable_endpoints = [
                "GET:/api/v1/embeddings/stats",
                "POST:/api/v1/embeddings/similarity",
                "POST:/api/v1/embeddings/vector/search",
            ]

            endpoint_key = f"{request.method}:{request.url.path}"
            if endpoint_key not in cacheable_endpoints:
                return

            cache_key = self.get_cache_key(request)

            # Response body'yi al
            if hasattr(response, "body"):
                response_content = response.body
                # Convert memoryview to bytes if needed
                if isinstance(response_content, memoryview):
                    response_content = bytes(response_content)
            else:
                return

            try:
                # JSON response'u cache'e kaydet
                response_data = json.loads(response_content)

                # Cache metadata ekle
                cache_data = {
                    "content": response_data,
                    "cached_at": time.time(),
                    "endpoint": endpoint_key,
                }

                if self.redis:
                    await self.redis.setex(cache_key, ttl, json.dumps(cache_data))

                logger.debug(
                    "response_cached",
                    cache_key=cache_key[-12:],
                    endpoint=endpoint_key,
                    ttl=ttl,
                )

            except json.JSONDecodeError:
                pass  # Non-JSON response, skip caching

        except Exception as e:
            logger.error("cache_set_error", error=str(e))

    async def get_stats(self) -> Dict[str, Any]:
        """Rate limiter istatistikleri"""
        try:
            # Rate limit keys'leri say
            rate_limit_keys = await self.redis.keys(f"{self.prefix}:*")

            # Cache keys'leri say
            cache_keys = await self.redis.keys("cache:request:*")

            # Memory usage estimate (sample first 100 keys for performance)
            memory_usage = 0
            sample_keys = rate_limit_keys[:100]
            for key in sample_keys:
                try:
                    usage = await self.redis.memory_usage(key)
                    memory_usage += int(usage) if usage else 0
                except:
                    continue

            return {
                "rate_limit_keys_count": len(rate_limit_keys),
                "cache_keys_count": len(cache_keys),
                "estimated_memory_bytes": memory_usage,
                "endpoint_configs": list(self.endpoint_configs.keys()),
                "user_type_multipliers": self.user_type_multipliers,
            }

        except Exception as e:
            logger.error("rate_limiter_stats_error", error=str(e))
            return {"error": str(e)}


# Global instance
async def get_rate_limiter() -> AdvancedRateLimiter:
    redis_client = redis.from_url(settings.redis_url)
    return AdvancedRateLimiter(redis_client)
