import logging
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from app.core.config import settings


class LoggingMiddleware:
    """Logging middleware - API isteklerini ve yanıtlarını loglar"""
    
    def __init__(self):
        self.logger = logging.getLogger("api")
        self.sensitive_paths = ["/api/v1/auth/login", "/api/v1/auth/register"]
        self.sensitive_headers = ["authorization", "cookie", "x-api-key"]
    
    async def __call__(self, request: Request, call_next):
        """Middleware call method"""
        
        # Request başlangıç zamanı
        start_time = time.time()
        
        # Request logla
        await self.log_request(request)
        
        # Response'u yakala
        response = await call_next(request)
        
        # Response süresini hesapla
        process_time = time.time() - start_time
        
        # Response logla
        await self.log_response(request, response, process_time)
        
        return response
    
    async def log_request(self, request: Request):
        """Request'i logla"""
        
        # Sensitive bilgileri filtrele
        filtered_headers = self.filter_sensitive_headers(request.headers)
        filtered_query_params = self.filter_sensitive_params(request.query_params)
        
        # Request body'sini al (sadece belirli endpoint'ler için)
        request_body = None
        if self.should_log_request_body(request):
            try:
                body = await request.body()
                if body:
                    request_body = body.decode('utf-8')
            except Exception:
                request_body = "[Unable to read request body]"
        
        # Log mesajını oluştur
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "request",
            "method": request.method,
            "path": str(request.url.path),
            "query_params": dict(filtered_query_params),
            "headers": dict(filtered_headers),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "request_id": getattr(request.state, 'request_id', None),
            "user_id": getattr(request.state, 'user_id', None),
            "user_role": getattr(request.state, 'user_role', None)
        }
        
        if request_body:
            log_data["body"] = request_body
        
        # Log seviyesini belirle
        log_level = self.get_request_log_level(request)
        
        # Logla
        log_message = f"Request: {request.method} {request.url.path}"
        
        if log_level == "DEBUG":
            self.logger.debug(log_message, extra=log_data)
        elif log_level == "INFO":
            self.logger.info(log_message, extra=log_data)
        else:
            self.logger.warning(log_message, extra=log_data)
    
    async def log_response(self, request: Request, response: Response, process_time: float):
        """Response'u logla"""
        
        # Response body'sini al (sadece belirli durumlar için)
        response_body = None
        if self.should_log_response_body(request, response):
            try:
                if hasattr(response, 'body'):
                    response_body = response.body.decode('utf-8') if response.body else None
                elif isinstance(response, StreamingResponse):
                    response_body = "[Streaming Response]"
            except Exception:
                response_body = "[Unable to read response body]"
        
        # Log mesajını oluştur
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "response",
            "method": request.method,
            "path": str(request.url.path),
            "status_code": response.status_code,
            "process_time": round(process_time, 4),
            "response_size": len(response_body) if response_body else 0,
            "headers": dict(response.headers),
            "request_id": getattr(request.state, 'request_id', None),
            "user_id": getattr(request.state, 'user_id', None),
            "user_role": getattr(request.state, 'user_role', None)
        }
        
        if response_body:
            log_data["body"] = response_body
        
        # Log seviyesini belirle
        log_level = self.get_response_log_level(response.status_code)
        
        # Logla
        log_message = f"Response: {request.method} {request.url.path} - {response.status_code} ({process_time:.4f}s)"
        
        if log_level == "DEBUG":
            self.logger.debug(log_message, extra=log_data)
        elif log_level == "INFO":
            self.logger.info(log_message, extra=log_data)
        elif log_level == "WARNING":
            self.logger.warning(log_message, extra=log_data)
        else:
            self.logger.error(log_message, extra=log_data)
    
    def filter_sensitive_headers(self, headers) -> Dict[str, str]:
        """Sensitive header'ları filtrele"""
        
        filtered = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value
        
        return filtered
    
    def filter_sensitive_params(self, query_params) -> Dict[str, str]:
        """Sensitive query parametrelerini filtrele"""
        
        sensitive_params = ["token", "password", "secret", "key"]
        filtered = {}
        
        for key, value in query_params.items():
            if key.lower() in sensitive_params:
                filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value
        
        return filtered
    
    def should_log_request_body(self, request: Request) -> bool:
        """Request body'sinin loglanıp loglanmayacağını belirle"""
        
        # Sensitive endpoint'lerde body loglama
        if str(request.url.path) in self.sensitive_paths:
            return False
        
        # Sadece POST, PUT, PATCH isteklerinde body logla
        if request.method in ["POST", "PUT", "PATCH"]:
            return True
        
        return False
    
    def should_log_response_body(self, request: Request, response: Response) -> bool:
        """Response body'sinin loglanıp loglanmayacağını belirle"""
        
        # Sadece hata durumlarında veya debug modda body logla
        if response.status_code >= 400:
            return True
        
        if settings.debug:
            return True
        
        return False
    
    def get_request_log_level(self, request: Request) -> str:
        """Request log seviyesini belirle"""
        
        if request is None:
            return "INFO"
        
        # Health check endpoint'leri için DEBUG
        if request.url.path == "/health":
            return "DEBUG"
        
        # Auth endpoint'leri için WARNING (güvenlik)
        if request.url.path.startswith("/api/v1/auth"):
            return "WARNING"
        
        return "INFO"
    
    def get_response_log_level(self, status_code: int) -> str:
        """Response log seviyesini belirle"""
        
        if status_code < 400:
            return "INFO"
        elif status_code < 500:
            return "WARNING"
        else:
            return "ERROR"
    
    async def log_error(self, request: Request, error: Exception, process_time: float):
        """Error'ları logla"""
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "error",
            "method": request.method,
            "path": str(request.url.path),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "process_time": round(process_time, 4),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "request_id": getattr(request.state, 'request_id', None),
            "user_id": getattr(request.state, 'user_id', None),
            "user_role": getattr(request.state, 'user_role', None)
        }
        
        log_message = f"Error: {request.method} {request.url.path} - {type(error).__name__}: {str(error)}"
        
        self.logger.error(log_message, extra=log_data)
    
    async def log_performance(self, request: Request, process_time: float):
        """Performance logla"""
        
        # Sadece yavaş istekleri logla (1 saniyeden fazla)
        if process_time > 1.0:
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "performance",
                "method": request.method,
                "path": str(request.url.path),
                "process_time": round(process_time, 4),
                "threshold": 1.0,
                "request_id": getattr(request.state, 'request_id', None),
                "user_id": getattr(request.state, 'user_id', None)
            }
            
            log_message = f"Slow Request: {request.method} {request.url.path} - {process_time:.4f}s"
            
            self.logger.warning(log_message, extra=log_data)
    
    async def get_log_summary(self, time_period: str = "1h") -> Dict[str, Any]:
        """Log summary döndür"""
        
        # Bu kısım production'da log aggregation servisinden veri çekebilir
        # Şimdilik basit bir yapı döndürüyoruz
        
        return {
            "total_requests": 0,
            "total_errors": 0,
            "average_response_time": 0.0,
            "requests_by_method": {},
            "requests_by_status": {},
            "top_endpoints": [],
            "error_rate": 0.0,
            "time_period": time_period
        }
    
    async def export_logs(self, start_time: datetime, end_time: datetime) -> str:
        """Log'ları export et"""
        
        # Bu kısım production'da log dosyalarından veri çekebilir
        # Şimdilik basit bir yapı döndürüyoruz
        
        return f"Logs exported from {start_time} to {end_time}"


# Global logging middleware instance
logging_middleware = LoggingMiddleware()
