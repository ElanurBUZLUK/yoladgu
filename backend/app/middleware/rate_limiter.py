import time
import asyncio
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from app.core.cache import cache_service
from app.schemas.error import RateLimitErrorResponse, ErrorType, ErrorSeverity


class RateLimiterMiddleware:
    """Rate limiting middleware - API isteklerini sınırlandırır"""
    
    def __init__(self):
        # Rate limit konfigürasyonları
        self.limits = {
            "default": {"requests": 100, "window": 60},  # 100 requests per minute
            "auth": {"requests": 10, "window": 60},      # 10 auth requests per minute
            "upload": {"requests": 5, "window": 60},     # 5 upload requests per minute
            "admin": {"requests": 1000, "window": 60},   # 1000 requests per minute for admins
            "student": {"requests": 200, "window": 60},  # 200 requests per minute for students
            "teacher": {"requests": 500, "window": 60},  # 500 requests per minute for teachers
        }
        
        # IP-based rate limiting
        self.ip_limits = {
            "requests": 1000,
            "window": 60
        }
    
    async def __call__(self, request: Request, call_next):
        """Middleware call method"""
        
        # Rate limit kontrolü
        rate_limit_result = await self.check_rate_limit(request)
        
        if not rate_limit_result["allowed"]:
            return await self.create_rate_limit_response(request, rate_limit_result)
        
        # Request'i işle
        response = await call_next(request)
        
        # Rate limit header'larını ekle
        await self.add_rate_limit_headers(response, rate_limit_result)
        
        return response
    
    async def check_rate_limit(self, request: Request) -> Dict[str, Any]:
        """Rate limit kontrolü yap"""
        
        # Client identifier'ı belirle
        client_id = await self.get_client_identifier(request)
        
        # Rate limit tipini belirle
        limit_type = await self.get_limit_type(request)
        
        # Limit konfigürasyonunu al
        limit_config = self.limits.get(limit_type, self.limits["default"])
        
        # Cache key oluştur
        cache_key = f"rate_limit:{limit_type}:{client_id}"
        
        # Mevcut istek sayısını al
        current_requests = await cache_service.get(cache_key)
        
        if current_requests is None:
            # İlk istek
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
            # Limit aşıldı
            return {
                "allowed": False,
                "limit": limit_config["requests"],
                "remaining": 0,
                "reset_time": int(time.time()) + limit_config["window"],
                "window": limit_config["window"]
            }
        
        # İstek sayısını artır
        await cache_service.incr(cache_key)
        
        return {
            "allowed": True,
            "limit": limit_config["requests"],
            "remaining": limit_config["requests"] - current_requests - 1,
            "reset_time": int(time.time()) + limit_config["window"],
            "window": limit_config["window"]
        }
    
    async def get_client_identifier(self, request: Request) -> str:
        """Client identifier'ı belirle"""
        
        # User ID varsa onu kullan
        if hasattr(request.state, 'user_id') and request.state.user_id:
            return f"user:{request.state.user_id}"
        
        # IP adresini kullan
        if request.client:
            return f"ip:{request.client.host}"
        
        # Fallback
        return "unknown"
    
    async def get_limit_type(self, request: Request) -> str:
        """Rate limit tipini belirle"""
        
        path = request.url.path
        
        # Auth endpoints
        if path.startswith("/api/v1/auth"):
            return "auth"
        
        # Upload endpoints
        if path.startswith("/api/v1/pdf/upload") or "upload" in path:
            return "upload"
        
        # User role'a göre limit
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
        """Rate limit response oluştur"""
        
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
                "Check rate limit documentation for your user type"
            ]
        )
        
        return JSONResponse(
            status_code=429,
            content=error_response.dict(),
            headers={
                "X-RateLimit-Limit": str(rate_limit_result["limit"]),
                "X-RateLimit-Remaining": str(rate_limit_result["remaining"]),
                "X-RateLimit-Reset": str(rate_limit_result["reset_time"]),
                "Retry-After": str(rate_limit_result["window"])
            }
        )
    
    async def add_rate_limit_headers(
        self, 
        response, 
        rate_limit_result: Dict[str, Any]
    ):
        """Rate limit header'larını ekle"""
        
        response.headers["X-RateLimit-Limit"] = str(rate_limit_result["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_result["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_limit_result["reset_time"])
    
    async def get_rate_limit_status(self, client_id: str) -> Dict[str, Any]:
        """Rate limit durumunu döndür"""
        
        status = {}
        
        for limit_type, config in self.limits.items():
            cache_key = f"rate_limit:{limit_type}:{client_id}"
            current_requests = await cache_service.get(cache_key)
            
            if current_requests is None:
                current_requests = 0
            
            status[limit_type] = {
                "current": int(current_requests),
                "limit": config["requests"],
                "remaining": max(0, config["requests"] - int(current_requests)),
                "window": config["window"]
            }
        
        return status
    
    async def reset_rate_limit(self, client_id: str, limit_type: Optional[str] = None):
        """Rate limit'i sıfırla"""
        
        if limit_type:
            cache_key = f"rate_limit:{limit_type}:{client_id}"
            await cache_service.delete(cache_key)
        else:
            # Tüm limit tiplerini sıfırla
            for lt in self.limits.keys():
                cache_key = f"rate_limit:{lt}:{client_id}"
                await cache_service.delete(cache_key)
    
    def update_rate_limit_config(self, limit_type: str, requests: int, window: int):
        """Rate limit konfigürasyonunu güncelle"""
        
        self.limits[limit_type] = {
            "requests": requests,
            "window": window
        }
    
    async def get_rate_limit_analytics(self) -> Dict[str, Any]:
        """Rate limit analytics döndür"""
        
        # Bu kısım production'da Redis'ten analytics verilerini çekebilir
        # Şimdilik basit bir yapı döndürüyoruz
        
        return {
            "total_limited_requests": 0,
            "rate_limits_by_type": {},
            "top_limited_clients": [],
            "rate_limit_effectiveness": 0.0
        }


# Global rate limiter instance
rate_limiter = RateLimiterMiddleware()
