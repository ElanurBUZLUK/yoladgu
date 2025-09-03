"""
Custom middleware for the application.
"""

import time
import uuid
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.config import settings


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log response (structured logging)
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "process_time": process_time,
            "user_agent": request.headers.get("user-agent"),
            "client_ip": request.client.host if request.client else None
        }
        
        print(json.dumps(log_data))  # In production, use proper logger
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = None):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute or settings.RATE_LIMIT_REQUESTS
        self.request_counts = {}
        self.window_start = {}
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/version"]:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Reset window if needed
        if (client_ip not in self.window_start or 
            current_time - self.window_start[client_ip] >= 60):
            self.window_start[client_ip] = current_time
            self.request_counts[client_ip] = 0
        
        # Check rate limit
        self.request_counts[client_ip] += 1
        if self.request_counts[client_ip] > self.requests_per_minute:
            return Response(
                content=json.dumps({
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded: {self.requests_per_minute} requests per minute"
                }),
                status_code=429,
                media_type="application/json"
            )
        
        return await call_next(request)


class PIIRedactionMiddleware(BaseHTTPMiddleware):
    """Middleware for redacting PII from logs."""
    
    PII_PATTERNS = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP Address
        r'\b\d{11}\b',  # Phone numbers (11 digits)
    ]
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # In production, implement PII redaction logic
        # This would scan request/response bodies and logs for PII patterns
        # and replace them with masked values
        
        response = await call_next(request)
        return response