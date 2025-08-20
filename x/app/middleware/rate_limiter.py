import time
import asyncio
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import structlog
from app.services.cache_service import cache_service
from app.core.config import settings
from app.schemas.error import RateLimitErrorResponse, ErrorType, ErrorSeverity
from app.services.monitoring_service import monitoring_service

logger = structlog.get_logger()


class RateLimiterMiddleware:
    """Enhanced rate limiting middleware with config-based limits"""
    
    def __init__(self):
        # Config-based rate limit configurations
        self.limits = {
            "default": {
                "requests": settings.rate_limit_requests_per_minute,
                "window": 60
            },
            "auth": {
                "requests": 10,
                "window": 60
            },
            "upload": {
                "requests": 5,
                "window": 60
            },
            "admin": {
                "requests": 1000,
                "window": 60
            },
            "student": {
                "requests": 200,
                "window": 60
            },
            "teacher": {
                "requests": 500,
                "window": 60
            },
            "llm": {
                "requests": 50,
                "window": 60
            },
            "vector_search": {
                "requests": 100,
                "window": 60
            },
            "monitoring": {
                "requests": 30,
                "window": 60
            }
        }
        
        # IP-based rate limiting
        self.ip_limits = {
            "requests": settings.rate_limit_requests_per_hour,
            "window": 3600  # 1 hour
        }
        
        # Burst limits
        self.burst_limits = {
            "requests": settings.rate_limit_burst_size,
            "window": 10  # 10 seconds
        }
        
        logger.info("Rate limiter initialized", 
                   enabled=settings.rate_limit_enabled,
                   storage_backend=settings.rate_limit_storage_backend)
    
    async def __call__(self, request: Request, call_next):
        """Middleware call method"""
        
        if not settings.rate_limit_enabled:
            return await call_next(request)
        
        start_time = time.time()
        
        try:
            # Rate limit kontrolü
            rate_limit_result = await self.check_rate_limit(request)
            
            if not rate_limit_result["allowed"]:
                # Record rate limit hit in monitoring
                client_id = await self.get_client_identifier(request)
                monitoring_service.record_rate_limit_hit(
                    endpoint=str(request.url.path),
                    user_id=client_id
                )
                return await self.create_rate_limit_response(request, rate_limit_result)
            
            # Request'i işle
            response = await call_next(request)
            
            # Rate limit header'larını ekle
            await self.add_rate_limit_headers(response, rate_limit_result)
            
            return response
            
        except Exception as e:
            logger.error("Error in rate limiter middleware", error=str(e))
            # Continue without rate limiting on error
            return await call_next(request)
    
    async def check_rate_limit(self, request: Request) -> Dict[str, Any]:
        """Enhanced rate limit kontrolü"""
        
        # Client identifier'ı belirle
        client_id = await self.get_client_identifier(request)
        
        # Multiple rate limit checks
        checks = [
            await self._check_user_rate_limit(request, client_id),
            await self._check_ip_rate_limit(request, client_id),
            await self._check_burst_limit(request, client_id)
        ]
        
        # If any check fails, return the first failure
        for check in checks:
            if not check["allowed"]:
                return check
        
        # All checks passed, return the user rate limit result
        return checks[0]
    
    async def _check_user_rate_limit(self, request: Request, client_id: str) -> Dict[str, Any]:
        """Check user-based rate limits"""
        limit_type = await self.get_limit_type(request)
        limit_config = self.limits.get(limit_type, self.limits["default"])
        
        cache_key = f"rate_limit:user:{limit_type}:{client_id}"
        return await self._check_limit(cache_key, limit_config)
    
    async def _check_ip_rate_limit(self, request: Request, client_id: str) -> Dict[str, Any]:
        """Check IP-based rate limits"""
        if not request.client:
            return {"allowed": True}
        
        ip = request.client.host
        cache_key = f"rate_limit:ip:{ip}"
        return await self._check_limit(cache_key, self.ip_limits)
    
    async def _check_burst_limit(self, request: Request, client_id: str) -> Dict[str, Any]:
        """Check burst rate limits"""
        cache_key = f"rate_limit:burst:{client_id}"
        return await self._check_limit(cache_key, self.burst_limits)
    
    async def _check_limit(self, cache_key: str, limit_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic limit check"""
        current_requests = await cache_service.get(cache_key)
        
        if current_requests is None:
            # First request
            await cache_service.set(cache_key, 1, limit_config["window"])
            return {
                "allowed": True,
                "limit": limit_config["requests"],
                "remaining": limit_config["requests"] - 1,
                "reset_time": int(time.time()) + limit_config["window"],
                "window": limit_config["window"]
            }
        
        current_requests = int(current_requests)
        
        if current_requests >= limit_config["requests"]:
            # Limit exceeded
            return {
                "allowed": False,
                "limit": limit_config["requests"],
                "remaining": 0,
                "reset_time": int(time.time()) + limit_config["window"],
                "window": limit_config["window"]
            }
        
        # Increment request count
        await cache_service.incr(cache_key)
        
        return {
            "allowed": True,
            "limit": limit_config["requests"],
            "remaining": limit_config["requests"] - current_requests - 1,
            "reset_time": int(time.time()) + limit_config["window"],
            "window": limit_config["window"]
        }
    
    async def get_client_identifier(self, request: Request) -> str:
        """Enhanced client identifier"""
        
        # User ID if available
        if hasattr(request.state, 'user_id') and request.state.user_id:
            return f"user:{request.state.user_id}"
        
        # IP address
        if request.client:
            return f"ip:{request.client.host}"
        
        # User agent as fallback
        user_agent = request.headers.get("user-agent", "unknown")
        return f"ua:{hash(user_agent) % 10000}"
    
    async def get_limit_type(self, request: Request) -> str:
        """Enhanced limit type determination"""
        
        path = request.url.path.lower()
        
        # Auth endpoints
        if "/auth" in path:
            return "auth"
        
        # Upload endpoints
        if "/upload" in path or "/pdf" in path:
            return "upload"
        
        # LLM endpoints
        if "/llm" in path or "/generate" in path or "/chat" in path:
            return "llm"
        
        # Vector search endpoints
        if "/vector" in path or "/search" in path or "/embedding" in path:
            return "vector_search"
        
        # Monitoring endpoints
        if "/monitoring" in path or "/metrics" in path or "/health" in path:
            return "monitoring"
        
        # User role-based limits
        if hasattr(request.state, 'user_role'):
            if request.state.user_role == "admin":
                return "admin"
            elif request.state.user_role == "teacher":
                return "teacher"
            elif request.state.user_role == "student":
                return "student"
        
        return "default"
    
    async def create_rate_limit_response(
        self, 
        request: Request, 
        rate_limit_result: Dict[str, Any]
    ) -> JSONResponse:
        """Enhanced rate limit response"""
        
        error_response = RateLimitErrorResponse(
            message="Rate limit exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
            severity=ErrorSeverity.MEDIUM,
            request_id=getattr(request.state, 'request_id', None),
            path=str(request.url.path),
            method=request.method,
            retry_after=rate_limit_result["window"],
            rate_limit_info={
                "limit": rate_limit_result["limit"],
                "remaining": rate_limit_result["remaining"],
                "reset_time": rate_limit_result["reset_time"],
                "window": rate_limit_result["window"]
            },
            suggestions=[
                f"Wait {rate_limit_result['window']} seconds before making another request",
                "Reduce request frequency",
                "Check rate limit documentation for your user type",
                "Consider upgrading your plan for higher limits"
            ]
        )
        
        return JSONResponse(
            status_code=429,
            content=error_response.dict(),
            headers={
                "X-RateLimit-Limit": str(rate_limit_result["limit"]),
                "X-RateLimit-Remaining": str(rate_limit_result["remaining"]),
                "X-RateLimit-Reset": str(rate_limit_result["reset_time"]),
                "Retry-After": str(rate_limit_result["window"]),
                "X-RateLimit-Window": str(rate_limit_result["window"])
            }
        )
    
    async def add_rate_limit_headers(
        self, 
        response, 
        rate_limit_result: Dict[str, Any]
    ):
        """Add rate limit headers to response"""
        
        response.headers["X-RateLimit-Limit"] = str(rate_limit_result["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_result["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_limit_result["reset_time"])
        response.headers["X-RateLimit-Window"] = str(rate_limit_result["window"])
    
    async def get_rate_limit_status(self, client_id: str) -> Dict[str, Any]:
        """Get comprehensive rate limit status"""
        
        status = {}
        
        for limit_type, config in self.limits.items():
            cache_key = f"rate_limit:user:{limit_type}:{client_id}"
            current_requests = await cache_service.get(cache_key)
            
            if current_requests is None:
                current_requests = 0
            
            status[limit_type] = {
                "current": int(current_requests),
                "limit": config["requests"],
                "remaining": max(0, config["requests"] - int(current_requests)),
                "window": config["window"],
                "percentage_used": (int(current_requests) / config["requests"]) * 100
            }
        
        return status
    
    async def reset_rate_limit(self, client_id: str, limit_type: Optional[str] = None):
        """Reset rate limits for client"""
        
        if limit_type:
            cache_key = f"rate_limit:user:{limit_type}:{client_id}"
            await cache_service.delete(cache_key)
            logger.info("Rate limit reset", client_id=client_id, limit_type=limit_type)
        else:
            # Reset all limit types
            for lt in self.limits.keys():
                cache_key = f"rate_limit:user:{lt}:{client_id}"
                await cache_service.delete(cache_key)
            logger.info("All rate limits reset", client_id=client_id)
    
    def update_rate_limit_config(self, limit_type: str, requests: int, window: int):
        """Update rate limit configuration"""
        
        self.limits[limit_type] = {
            "requests": requests,
            "window": window
        }
        logger.info("Rate limit config updated", limit_type=limit_type, requests=requests, window=window)
    
    async def get_rate_limit_analytics(self) -> Dict[str, Any]:
        """Get rate limit analytics"""
        
        try:
            # This would typically query Redis for analytics data
            # For now, return a basic structure
            
            return {
                "total_limited_requests": 0,
                "rate_limits_by_type": {},
                "top_limited_clients": [],
                "rate_limit_effectiveness": 0.0,
                "config": {
                    "enabled": settings.rate_limit_enabled,
                    "storage_backend": settings.rate_limit_storage_backend,
                    "limits": self.limits
                }
            }
        except Exception as e:
            logger.error("Error getting rate limit analytics", error=str(e))
            return {"error": str(e)}


# Global rate limiter instance
rate_limiter = RateLimiterMiddleware()
