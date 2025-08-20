import logging
import traceback
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from pydantic import ValidationError

from app.schemas.error import (
    ErrorResponse, ValidationErrorResponse, AuthenticationErrorResponse,
    AuthorizationErrorResponse, NotFoundErrorResponse, ConflictErrorResponse,
    RateLimitErrorResponse, InternalErrorResponse, ExternalServiceErrorResponse,
    DatabaseErrorResponse, FileUploadErrorResponse, LLMErrorResponse, MCPErrorResponse,
    ErrorType, ErrorSeverity
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware:
    """Error handling middleware - API hatalarını yakalar ve standart formatta yanıt verir"""
    
    def __init__(self):
        self.error_logs = []
    
    async def __call__(self, request: Request, call_next):
        """Middleware call method"""
        
        # Request ID oluştur
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Request başlangıç zamanı
        start_time = datetime.utcnow()
        
        try:
            # Request'i işle
            response = await call_next(request)
            
            # Response süresini hesapla
            process_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Response header'larına bilgi ekle
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as exc:
            # Hata durumunda
            return await self.handle_exception(request, exc, request_id, start_time)
    
    async def handle_exception(
        self, 
        request: Request, 
        exc: Exception, 
        request_id: str,
        start_time: datetime
    ) -> JSONResponse:
        """Exception'ları işle"""
        
        # Hata tipini belirle
        error_response = await self.create_error_response(request, exc, request_id)
        
        # Hata logla
        await self.log_error(request, exc, error_response, request_id, start_time)
        
        # HTTP status code belirle
        status_code = self.get_http_status_code(error_response.error_type)
        
        # Response oluştur
        return JSONResponse(
            status_code=status_code,
            content=error_response.dict(),
            headers={
                "X-Request-ID": request_id,
                "X-Error-Type": error_response.error_type.value
            }
        )
    
    async def create_error_response(
        self, 
        request: Request, 
        exc: Exception, 
        request_id: str
    ) -> ErrorResponse:
        """Error response oluştur"""
        
        # Hata tipini belirle
        if isinstance(exc, RequestValidationError):
            return await self.handle_validation_error(request, exc, request_id)
        elif isinstance(exc, HTTPException):
            return await self.handle_http_exception(request, exc, request_id)
        elif isinstance(exc, SQLAlchemyError):
            return await self.handle_database_error(request, exc, request_id)
        elif isinstance(exc, ValidationError):
            return await self.handle_validation_error(request, exc, request_id)
        else:
            return await self.handle_generic_error(request, exc, request_id)
    
    async def handle_validation_error(
        self, 
        request: Request, 
        exc: Exception, 
        request_id: str
    ) -> ValidationErrorResponse:
        """Validation error'ları işle"""
        
        error_details = []
        
        if hasattr(exc, 'errors'):
            for error in exc.errors():
                error_details.append({
                    "field": " -> ".join(str(loc) for loc in error["loc"]),
                    "message": error["msg"],
                    "code": error["type"],
                    "value": error.get("input")
                })
        
        return ValidationErrorResponse(
            message="Validation error occurred",
            details=error_details,
            error_code="VALIDATION_ERROR",
            request_id=request_id,
            path=str(request.url.path),
            method=request.method,
            suggestions=[
                "Check the request format and required fields",
                "Ensure all required parameters are provided",
                "Verify data types and constraints"
            ]
        )
    
    async def handle_http_exception(
        self, 
        request: Request, 
        exc: HTTPException, 
        request_id: str
    ) -> ErrorResponse:
        """HTTP exception'ları işle"""
        
        if exc.status_code == 401:
            return AuthenticationErrorResponse(
                message="Authentication required",
                error_code="AUTHENTICATION_REQUIRED",
                request_id=request_id,
                path=str(request.url.path),
                method=request.method,
                suggestions=[
                    "Provide valid authentication credentials",
                    "Check if your token is expired",
                    "Ensure you're logged in"
                ]
            )
        elif exc.status_code == 403:
            return AuthorizationErrorResponse(
                message="Access denied",
                error_code="ACCESS_DENIED",
                request_id=request_id,
                path=str(request.url.path),
                method=request.method,
                suggestions=[
                    "Check your user permissions",
                    "Contact administrator for access",
                    "Verify your role has required privileges"
                ]
            )
        elif exc.status_code == 404:
            return NotFoundErrorResponse(
                message="Resource not found",
                error_code="RESOURCE_NOT_FOUND",
                request_id=request_id,
                path=str(request.url.path),
                method=request.method,
                suggestions=[
                    "Check the resource ID or path",
                    "Verify the resource exists",
                    "Ensure you have access to the resource"
                ]
            )
        elif exc.status_code == 409:
            return ConflictErrorResponse(
                message="Resource conflict",
                error_code="RESOURCE_CONFLICT",
                request_id=request_id,
                path=str(request.url.path),
                method=request.method,
                suggestions=[
                    "Check for duplicate resources",
                    "Verify unique constraints",
                    "Resolve conflicts before retrying"
                ]
            )
        elif exc.status_code == 429:
            return RateLimitErrorResponse(
                message="Rate limit exceeded",
                error_code="RATE_LIMIT_EXCEEDED",
                request_id=request_id,
                path=str(request.url.path),
                method=request.method,
                retry_after=60,
                suggestions=[
                    "Wait before making another request",
                    "Reduce request frequency",
                    "Check rate limit documentation"
                ]
            )
        else:
            return InternalErrorResponse(
                message=str(exc.detail) if hasattr(exc, 'detail') else "HTTP error occurred",
                error_code=f"HTTP_{exc.status_code}",
                request_id=request_id,
                path=str(request.url.path),
                method=request.method
            )
    
    async def handle_database_error(
        self, 
        request: Request, 
        exc: SQLAlchemyError, 
        request_id: str
    ) -> DatabaseErrorResponse:
        """Database error'ları işle"""
        
        error_message = "Database operation failed"
        operation = "unknown"
        table = "unknown"
        constraint = None
        
        if isinstance(exc, IntegrityError):
            error_message = "Data integrity constraint violation"
            if hasattr(exc, 'orig'):
                constraint = str(exc.orig)
        
        return DatabaseErrorResponse(
            message=error_message,
            error_code="DATABASE_ERROR",
            request_id=request_id,
            path=str(request.url.path),
            method=request.method,
            operation=operation,
            table=table,
            constraint=constraint,
            suggestions=[
                "Check data constraints and relationships",
                "Verify input data format",
                "Contact support if issue persists"
            ]
        )
    
    async def handle_generic_error(
        self, 
        request: Request, 
        exc: Exception, 
        request_id: str
    ) -> InternalErrorResponse:
        """Generic error'ları işle"""
        
        error_id = str(uuid.uuid4())
        
        return InternalErrorResponse(
            message="Internal server error",
            error_code="INTERNAL_ERROR",
            error_id=error_id,
            request_id=request_id,
            path=str(request.url.path),
            method=request.method,
            suggestions=[
                "Try again later",
                "Contact support if issue persists",
                "Check system status"
            ]
        )
    
    def get_http_status_code(self, error_type: ErrorType) -> int:
        """Error tipine göre HTTP status code döndür"""
        
        status_codes = {
            ErrorType.VALIDATION_ERROR: 400,
            ErrorType.AUTHENTICATION_ERROR: 401,
            ErrorType.AUTHORIZATION_ERROR: 403,
            ErrorType.NOT_FOUND_ERROR: 404,
            ErrorType.CONFLICT_ERROR: 409,
            ErrorType.RATE_LIMIT_ERROR: 429,
            ErrorType.INTERNAL_ERROR: 500,
            ErrorType.EXTERNAL_SERVICE_ERROR: 502,
            ErrorType.DATABASE_ERROR: 500,
            ErrorType.FILE_UPLOAD_ERROR: 400,
            ErrorType.LLM_ERROR: 503,
            ErrorType.MCP_ERROR: 503
        }
        
        return status_codes.get(error_type, 500)
    
    async def log_error(
        self, 
        request: Request, 
        exc: Exception, 
        error_response: ErrorResponse,
        request_id: str,
        start_time: datetime
    ):
        """Hata logla"""
        
        # Hata detaylarını topla
        error_log = {
            "error_id": error_response.error_id if hasattr(error_response, 'error_id') else None,
            "request_id": request_id,
            "error_type": error_response.error_type.value,
            "error_message": error_response.message, # Renamed from "message"
            "severity": error_response.severity.value,
            "timestamp": error_response.timestamp,
            "path": str(request.url.path),
            "method": request.method,
            "user_id": getattr(request.state, 'user_id', None),
            "ip_address": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "process_time": (datetime.utcnow() - start_time).total_seconds(),
            "stack_trace": traceback.format_exc() if settings.debug else None
        }
        
        # Log seviyesini belirle
        log_level = self.get_log_level(error_response.severity)
        
        # Log mesajını oluştur
        log_message = f"Error {error_response.error_type.value}: {error_response.message}"
        
        # Logla
        if log_level == "ERROR":
            logger.error(log_message, extra=error_log)
        elif log_level == "WARNING":
            logger.warning(log_message, extra=error_log)
        else:
            logger.info(log_message, extra=error_log)
        
        # Error log'u sakla (production'da bu kısım database'e kaydedilebilir)
        self.error_logs.append(error_log)
        
        # Log sayısını sınırla
        if len(self.error_logs) > 1000:
            self.error_logs = self.error_logs[-500:]
    
    def get_log_level(self, severity: ErrorSeverity) -> str:
        """Severity'ye göre log seviyesi döndür"""
        
        log_levels = {
            ErrorSeverity.LOW: "INFO",
            ErrorSeverity.MEDIUM: "WARNING",
            ErrorSeverity.HIGH: "ERROR",
            ErrorSeverity.CRITICAL: "ERROR"
        }
        
        return log_levels.get(severity, "INFO")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Error summary döndür"""
        
        if not self.error_logs:
            return {"total_errors": 0}
        
        # Error tiplerine göre sayım
        error_types = {}
        error_severities = {}
        
        for log in self.error_logs:
            error_type = log["error_type"]
            severity = log["severity"]
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            error_severities[severity] = error_severities.get(severity, 0) + 1
        
        return {
            "total_errors": len(self.error_logs),
            "errors_by_type": error_types,
            "errors_by_severity": error_severities,
            "recent_errors": self.error_logs[-10:] if self.error_logs else []
        }


# Global error handler instance
error_handler = ErrorHandlerMiddleware()
