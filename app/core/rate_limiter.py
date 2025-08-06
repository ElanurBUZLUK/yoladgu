"""
Rate Limiter
API istek sınırlama
"""

import structlog
from typing import Dict, Optional
from datetime import datetime, timedelta
from fastapi import HTTPException, Request
from app.services.redis_service import redis_service

logger = structlog.get_logger()


class RateLimiter:
    """Rate limiter sınıfı"""
    
    def __init__(self):
        self.default_limit = 100  # requests per window
        self.default_window = 3600  # seconds (1 hour)
        self.limits = {
            "auth": {"limit": 10, "window": 300},  # 10 requests per 5 minutes
            "api": {"limit": 1000, "window": 3600},  # 1000 requests per hour
            "admin": {"limit": 10000, "window": 3600},  # 10000 requests per hour
            "ml": {"limit": 50, "window": 3600},  # 50 ML requests per hour
            "stream": {"limit": 500, "window": 3600},  # 500 stream requests per hour
        }
    
    async def check_rate_limit(self, request: Request, limit_type: str = "api") -> bool:
        """Rate limit kontrolü"""
        try:
            # Client IP'sini al
            client_ip = self._get_client_ip(request)
            
            # Limit ayarlarını al
            limit_config = self.limits.get(limit_type, {
                "limit": self.default_limit,
                "window": self.default_window
            })
            
            # Rate limit key'i oluştur
            key = f"rate_limit:{limit_type}:{client_ip}"
            
            # Rate limit kontrolü
            is_allowed = await redis_service.set_rate_limit(
                key=key,
                limit=limit_config["limit"],
                window=limit_config["window"]
            )
            
            if not is_allowed:
                logger.warning("rate_limit_exceeded", 
                              client_ip=client_ip,
                              limit_type=limit_type,
                              limit=limit_config["limit"])
            
            return is_allowed
            
        except Exception as e:
            logger.error("rate_limit_check_error", error=str(e))
            # Hata durumunda izin ver
            return True
    
    async def get_rate_limit_info(self, request: Request, limit_type: str = "api") -> Dict:
        """Rate limit bilgilerini getir"""
        try:
            client_ip = self._get_client_ip(request)
            key = f"rate_limit:{limit_type}:{client_ip}"
            
            limit_info = await redis_service.get_rate_limit(key)
            limit_config = self.limits.get(limit_type, {
                "limit": self.default_limit,
                "window": self.default_window
            })
            
            return {
                "current_count": limit_info["current_count"],
                "limit": limit_config["limit"],
                "remaining": max(0, limit_config["limit"] - limit_info["current_count"]),
                "window_seconds": limit_config["window"],
                "reset_time": datetime.now() + timedelta(seconds=limit_info["remaining_ttl"])
            }
            
        except Exception as e:
            logger.error("get_rate_limit_info_error", error=str(e))
            return {
                "current_count": 0,
                "limit": self.default_limit,
                "remaining": self.default_limit,
                "window_seconds": self.default_window,
                "reset_time": datetime.now() + timedelta(hours=1)
            }
    
    def _get_client_ip(self, request: Request) -> str:
        """Client IP'sini al"""
        # X-Forwarded-For header'ını kontrol et
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # X-Real-IP header'ını kontrol et
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Client IP'sini al
        return request.client.host if request.client else "unknown"


# Global rate limiter instance
rate_limiter = RateLimiter()


async def get_rate_limiter() -> RateLimiter:
    """Rate limiter instance'ını getir"""
    return rate_limiter


async def check_rate_limit_middleware(request: Request, limit_type: str = "api"):
    """Rate limit middleware"""
    try:
        is_allowed = await rate_limiter.check_rate_limit(request, limit_type)
        
        if not is_allowed:
            limit_info = await rate_limiter.get_rate_limit_info(request, limit_type)
            
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": limit_info["limit"],
                    "remaining": limit_info["remaining"],
                    "reset_time": limit_info["reset_time"].isoformat()
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("rate_limit_middleware_error", error=str(e))
        # Hata durumunda izin ver
        pass


def rate_limit(limit_type: str = "api"):
    """Rate limit decorator"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Request objesini bul
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request:
                await check_rate_limit_middleware(request, limit_type)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator 