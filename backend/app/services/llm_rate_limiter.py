"""
Advanced Rate Limiting for LLM APIs with token-based and request-based limits
"""
import asyncio
import logging
import time
from typing import Dict, Optional, Tuple, Any, List
from enum import Enum
from dataclasses import dataclass, field
import json
from collections import defaultdict, deque

import redis.asyncio as aioredis
from app.core.config import settings

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Rate limit types"""
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    TOKENS_PER_MINUTE = "tokens_per_minute"
    TOKENS_PER_HOUR = "tokens_per_hour"
    COST_PER_HOUR = "cost_per_hour"
    COST_PER_DAY = "cost_per_day"


@dataclass
class RateLimit:
    """Rate limit configuration"""
    limit_type: RateLimitType
    limit: int
    window_seconds: int
    burst_allowance: int = 0  # Allow bursts up to this amount


@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    limit_type: str
    current_usage: int
    limit: int
    reset_time: float
    retry_after: Optional[float] = None
    message: Optional[str] = None


class LLMRateLimiter:
    """Advanced rate limiter for LLM services"""
    
    def __init__(self):
        self.redis_client = None
        self.local_cache = defaultdict(lambda: defaultdict(deque))
        self.provider_limits = {}
        
        # Initialize provider-specific limits
        self._initialize_provider_limits()
        
        # Connect to Redis - will be called during initialization
        # asyncio.create_task(self._connect_redis())
    
    async def initialize(self):
        """Initialize the rate limiter with Redis connection"""
        await self._connect_redis()
    
    def _initialize_provider_limits(self):
        """Initialize rate limits for different providers"""
        
        # OpenAI GPT-4 limits
        self.provider_limits["openai_gpt4"] = [
            RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 500, 60, burst_allowance=50),
            RateLimit(RateLimitType.TOKENS_PER_MINUTE, 150000, 60, burst_allowance=15000),
            RateLimit(RateLimitType.COST_PER_HOUR, 100, 3600),  # $100/hour
            RateLimit(RateLimitType.COST_PER_DAY, 1000, 86400)  # $1000/day
        ]
        
        # OpenAI GPT-3.5 limits
        self.provider_limits["openai_gpt35"] = [
            RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 3500, 60, burst_allowance=350),
            RateLimit(RateLimitType.TOKENS_PER_MINUTE, 1000000, 60, burst_allowance=100000),
            RateLimit(RateLimitType.COST_PER_HOUR, 50, 3600),  # $50/hour
            RateLimit(RateLimitType.COST_PER_DAY, 500, 86400)  # $500/day
        ]
        
        # Anthropic Claude limits
        self.provider_limits["anthropic_claude_opus"] = [
            RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 100, 60, burst_allowance=20),
            RateLimit(RateLimitType.TOKENS_PER_MINUTE, 80000, 60, burst_allowance=8000),
            RateLimit(RateLimitType.COST_PER_HOUR, 75, 3600),  # $75/hour
            RateLimit(RateLimitType.COST_PER_DAY, 750, 86400)  # $750/day
        ]
        
        self.provider_limits["anthropic_claude_sonnet"] = [
            RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 200, 60, burst_allowance=40),
            RateLimit(RateLimitType.TOKENS_PER_MINUTE, 200000, 60, burst_allowance=20000),
            RateLimit(RateLimitType.COST_PER_HOUR, 30, 3600),  # $30/hour
            RateLimit(RateLimitType.COST_PER_DAY, 300, 86400)  # $300/day
        ]
        
        self.provider_limits["anthropic_claude_haiku"] = [
            RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 1000, 60, burst_allowance=200),
            RateLimit(RateLimitType.TOKENS_PER_MINUTE, 500000, 60, burst_allowance=50000),
            RateLimit(RateLimitType.COST_PER_HOUR, 10, 3600),  # $10/hour
            RateLimit(RateLimitType.COST_PER_DAY, 100, 86400)  # $100/day
        ]
        
        # Local model (no external limits)
        self.provider_limits["local_model"] = [
            RateLimit(RateLimitType.REQUESTS_PER_MINUTE, 100, 60),  # Hardware dependent
            RateLimit(RateLimitType.COST_PER_HOUR, 0, 3600),  # No cost
            RateLimit(RateLimitType.COST_PER_DAY, 0, 86400)  # No cost
        ]
    
    async def _connect_redis(self):
        """Connect to Redis for distributed rate limiting"""
        try:
            if settings.redis_url:
                self.redis_client = aioredis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                logger.info("Connected to Redis for distributed rate limiting")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using local cache only.")
    
    async def check_rate_limit(self,
                              provider: str,
                              user_id: Optional[str] = None,
                              tokens: int = 0,
                              cost: float = 0.0) -> RateLimitResult:
        """
        Check if request is within rate limits
        """
        
        # Get provider limits
        if provider not in self.provider_limits:
            logger.warning(f"No rate limits defined for provider: {provider}")
            return RateLimitResult(
                allowed=True,
                limit_type="unknown",
                current_usage=0,
                limit=0,
                reset_time=time.time()
            )
        
        limits = self.provider_limits[provider]
        
        # Check each rate limit
        for rate_limit in limits:
            result = await self._check_individual_limit(
                provider=provider,
                user_id=user_id,
                rate_limit=rate_limit,
                tokens=tokens,
                cost=cost
            )
            
            if not result.allowed:
                return result
        
        # All limits passed
        return RateLimitResult(
            allowed=True,
            limit_type="all",
            current_usage=0,
            limit=0,
            reset_time=time.time()
        )
    
    async def _check_individual_limit(self,
                                    provider: str,
                                    user_id: Optional[str],
                                    rate_limit: RateLimit,
                                    tokens: int,
                                    cost: float) -> RateLimitResult:
        """Check individual rate limit"""
        
        # Determine what to count
        if rate_limit.limit_type in [RateLimitType.REQUESTS_PER_MINUTE, RateLimitType.REQUESTS_PER_HOUR]:
            increment = 1
        elif rate_limit.limit_type in [RateLimitType.TOKENS_PER_MINUTE, RateLimitType.TOKENS_PER_HOUR]:
            increment = tokens
        elif rate_limit.limit_type in [RateLimitType.COST_PER_HOUR, RateLimitType.COST_PER_DAY]:
            increment = cost
        else:
            increment = 1
        
        # Create cache key
        key_parts = [provider, rate_limit.limit_type.value]
        if user_id:
            key_parts.append(user_id)
        
        cache_key = ":".join(key_parts)
        
        # Check limit using sliding window
        current_time = time.time()
        window_start = current_time - rate_limit.window_seconds
        
        if self.redis_client:
            current_usage = await self._redis_sliding_window_check(
                cache_key, current_time, window_start, increment, rate_limit
            )
        else:
            current_usage = await self._local_sliding_window_check(
                cache_key, current_time, window_start, increment, rate_limit
            )
        
        # Check if limit exceeded
        effective_limit = rate_limit.limit + rate_limit.burst_allowance
        allowed = current_usage <= effective_limit
        
        return RateLimitResult(
            allowed=allowed,
            limit_type=rate_limit.limit_type.value,
            current_usage=int(current_usage),
            limit=rate_limit.limit,
            reset_time=current_time + rate_limit.window_seconds,
            retry_after=None if allowed else rate_limit.window_seconds,
            message=None if allowed else f"Rate limit exceeded for {rate_limit.limit_type.value}"
        )
    
    async def _redis_sliding_window_check(self,
                                        key: str,
                                        current_time: float,
                                        window_start: float,
                                        increment: float,
                                        rate_limit: RateLimit) -> float:
        """Redis-based sliding window rate limit check"""
        
        try:
            # Use Redis sorted set for sliding window
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current entries
            pipe.zcard(key)
            
            # Add current request
            if increment > 0:
                pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiry
            pipe.expire(key, rate_limit.window_seconds + 1)
            
            results = await pipe.execute()
            
            # Calculate current usage
            if increment > 0:
                # For requests, count number of entries
                if rate_limit.limit_type in [RateLimitType.REQUESTS_PER_MINUTE, RateLimitType.REQUESTS_PER_HOUR]:
                    current_usage = results[1] + 1
                else:
                    # For tokens/cost, sum the scores
                    entries = await self.redis_client.zrangebyscore(
                        key, window_start, current_time, withscores=True
                    )
                    current_usage = sum(score for _, score in entries) + increment
            else:
                current_usage = results[1]
            
            return current_usage
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to local cache
            return await self._local_sliding_window_check(
                key, current_time, window_start, increment, rate_limit
            )
    
    async def _local_sliding_window_check(self,
                                        key: str,
                                        current_time: float,
                                        window_start: float,
                                        increment: float,
                                        rate_limit: RateLimit) -> float:
        """Local cache-based sliding window rate limit check"""
        
        # Get or create deque for this key
        entries = self.local_cache[key][rate_limit.limit_type.value]
        
        # Remove old entries
        while entries and entries[0][0] < window_start:
            entries.popleft()
        
        # Add current request
        if increment > 0:
            entries.append((current_time, increment))
        
        # Calculate current usage
        if rate_limit.limit_type in [RateLimitType.REQUESTS_PER_MINUTE, RateLimitType.REQUESTS_PER_HOUR]:
            current_usage = len(entries)
        else:
            current_usage = sum(entry[1] for entry in entries)
        
        return current_usage
    
    async def record_request(self,
                           provider: str,
                           user_id: Optional[str] = None,
                           tokens: int = 0,
                           cost: float = 0.0):
        """Record a successful request for rate limiting"""
        
        # This is called after a successful request to update counters
        await self.check_rate_limit(provider, user_id, tokens, cost)
    
    async def get_rate_limit_status(self,
                                   provider: str,
                                   user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current rate limit status for provider/user"""
        
        if provider not in self.provider_limits:
            return {"error": f"Unknown provider: {provider}"}
        
        status = {
            "provider": provider,
            "user_id": user_id,
            "limits": []
        }
        
        limits = self.provider_limits[provider]
        current_time = time.time()
        
        for rate_limit in limits:
            window_start = current_time - rate_limit.window_seconds
            
            # Create cache key
            key_parts = [provider, rate_limit.limit_type.value]
            if user_id:
                key_parts.append(user_id)
            cache_key = ":".join(key_parts)
            
            # Get current usage
            if self.redis_client:
                try:
                    current_usage = await self._redis_sliding_window_check(
                        cache_key, current_time, window_start, 0, rate_limit
                    )
                except:
                    current_usage = await self._local_sliding_window_check(
                        cache_key, current_time, window_start, 0, rate_limit
                    )
            else:
                current_usage = await self._local_sliding_window_check(
                    cache_key, current_time, window_start, 0, rate_limit
                )
            
            # Calculate remaining
            remaining = max(0, rate_limit.limit - current_usage)
            reset_time = current_time + rate_limit.window_seconds
            
            limit_status = {
                "type": rate_limit.limit_type.value,
                "limit": rate_limit.limit,
                "current_usage": int(current_usage),
                "remaining": int(remaining),
                "reset_time": reset_time,
                "window_seconds": rate_limit.window_seconds,
                "burst_allowance": rate_limit.burst_allowance
            }
            
            status["limits"].append(limit_status)
        
        return status
    
    async def reset_rate_limits(self,
                               provider: str,
                               user_id: Optional[str] = None,
                               limit_type: Optional[RateLimitType] = None):
        """Reset rate limits for provider/user"""
        
        # Create base key pattern
        key_parts = [provider]
        if limit_type:
            key_parts.append(limit_type.value)
        else:
            key_parts.append("*")
        
        if user_id:
            key_parts.append(user_id)
        
        pattern = ":".join(key_parts)
        
        try:
            if self.redis_client:
                # Delete matching keys from Redis
                if "*" in pattern:
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        await self.redis_client.delete(*keys)
                else:
                    await self.redis_client.delete(pattern)
            
            # Clear local cache
            keys_to_remove = []
            for key in self.local_cache:
                if pattern.replace("*", "") in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.local_cache[key]
            
            logger.info(f"Reset rate limits for pattern: {pattern}")
            
        except Exception as e:
            logger.error(f"Failed to reset rate limits: {e}")
    
    def update_provider_limits(self,
                              provider: str,
                              limits: List[RateLimit]):
        """Update rate limits for a provider"""
        
        self.provider_limits[provider] = limits
        logger.info(f"Updated rate limits for provider: {provider}")
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics"""
        
        stats = {
            "providers": list(self.provider_limits.keys()),
            "redis_connected": self.redis_client is not None,
            "local_cache_keys": len(self.local_cache),
            "provider_limits": {}
        }
        
        # Get status for each provider
        for provider in self.provider_limits:
            provider_status = await self.get_rate_limit_status(provider)
            stats["provider_limits"][provider] = provider_status
        
        return stats


# Global instance
llm_rate_limiter = LLMRateLimiter()
