import logging
import traceback
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class ErrorCode:
    """Standard error codes for the application"""
    
    # General errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # Service-specific errors
    PDF_PROCESSING_ERROR = "PDF_PROCESSING_ERROR"
    EMBEDDING_GENERATION_ERROR = "EMBEDDING_GENERATION_ERROR"
    VECTOR_DB_ERROR = "VECTOR_DB_ERROR"
    LLM_SERVICE_ERROR = "LLM_SERVICE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    
    # Business logic errors
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    INVALID_OPERATION = "INVALID_OPERATION"
    CONCURRENT_MODIFICATION = "CONCURRENT_MODIFICATION"
    
    # External service errors
    EXTERNAL_SERVICE_UNAVAILABLE = "EXTERNAL_SERVICE_UNAVAILABLE"
    EXTERNAL_SERVICE_TIMEOUT = "EXTERNAL_SERVICE_TIMEOUT"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"


class ErrorSeverity:
    """Error severity levels"""
    
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class StandardizedError:
    """Standardized error response structure"""
    
    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = 500,
        severity: str = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.severity = severity
        self.details = details or {}
        self.request_id = request_id
        self.timestamp = timestamp or datetime.utcnow()
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format"""
        return {
            "error": {
                "message": self.message,
                "code": self.code,
                "severity": self.severity,
                "timestamp": self.timestamp.isoformat(),
                "request_id": self.request_id,
                "details": self.details,
                "context": self.context
            }
        }
    
    def to_response(self) -> JSONResponse:
        """Convert error to FastAPI JSONResponse"""
        return JSONResponse(
            status_code=self.status_code,
            content=self.to_dict()
        )


class ErrorHandler:
    """Centralized error handling service"""
    
    def __init__(self):
        self.error_counts = {}
        self.severity_thresholds = {
            ErrorSeverity.LOW: 100,
            ErrorSeverity.MEDIUM: 50,
            ErrorSeverity.HIGH: 10,
            ErrorSeverity.CRITICAL: 1
        }
    
    def create_error_response(
        self,
        message: str,
        code: str = ErrorCode.INTERNAL_ERROR,
        status_code: int = 500,
        severity: str = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> StandardizedError:
        """Create a standardized error response"""
        
        # Track error count
        self._track_error(code, severity)
        
        # Log error based on severity
        self._log_error(message, code, severity, details, context)
        
        return StandardizedError(
            message=message,
            code=code,
            status_code=status_code,
            severity=severity,
            details=details,
            request_id=request_id,
            context=context
        )
    
    def handle_validation_error(
        self,
        validation_error: ValidationError,
        request_id: Optional[str] = None
    ) -> StandardizedError:
        """Handle Pydantic validation errors"""
        
        error_details = []
        for error in validation_error.errors():
            error_details.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        return self.create_error_response(
            message="Validation failed",
            code=ErrorCode.VALIDATION_ERROR,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            severity=ErrorSeverity.LOW,
            details={"validation_errors": error_details},
            request_id=request_id,
            context={"error_type": "validation_error"}
        )
    
    def handle_database_error(
        self,
        error: Exception,
        operation: str,
        request_id: Optional[str] = None
    ) -> StandardizedError:
        """Handle database-related errors"""
        
        # Determine severity based on error type
        severity = ErrorSeverity.MEDIUM
        if "connection" in str(error).lower():
            severity = ErrorSeverity.HIGH
        elif "timeout" in str(error).lower():
            severity = ErrorSeverity.MEDIUM
        elif "constraint" in str(error).lower():
            severity = ErrorSeverity.LOW
        
        return self.create_error_response(
            message=f"Database operation failed: {operation}",
            code=ErrorCode.DATABASE_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            severity=severity,
            details={
                "operation": operation,
                "error_type": type(error).__name__,
                "error_message": str(error)
            },
            request_id=request_id,
            context={"error_type": "database_error", "operation": operation}
        )
    
    def handle_external_service_error(
        self,
        error: Exception,
        service_name: str,
        request_id: Optional[str] = None
    ) -> StandardizedError:
        """Handle external service errors"""
        
        # Determine error type and status code
        error_message = str(error).lower()
        
        if "timeout" in error_message or "timed out" in error_message:
            code = ErrorCode.EXTERNAL_SERVICE_TIMEOUT
            status_code = status.HTTP_504_GATEWAY_TIMEOUT
            severity = ErrorSeverity.MEDIUM
        elif "unavailable" in error_message or "connection refused" in error_message:
            code = ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            severity = ErrorSeverity.HIGH
        else:
            code = ErrorCode.EXTERNAL_SERVICE_ERROR
            status_code = status.HTTP_502_BAD_GATEWAY
            severity = ErrorSeverity.MEDIUM
        
        return self.create_error_response(
            message=f"External service error: {service_name}",
            code=code,
            status_code=status_code,
            severity=severity,
            details={
                "service": service_name,
                "error_type": type(error).__name__,
                "error_message": str(error)
            },
            request_id=request_id,
            context={"error_type": "external_service_error", "service": service_name}
        )
    
    def handle_concurrent_modification_error(
        self,
        resource_type: str,
        resource_id: str,
        request_id: Optional[str] = None
    ) -> StandardizedError:
        """Handle concurrent modification errors"""
        
        return self.create_error_response(
            message=f"Resource {resource_type} was modified by another request",
            code=ErrorCode.CONCURRENT_MODIFICATION,
            status_code=status.HTTP_409_CONFLICT,
            severity=ErrorSeverity.MEDIUM,
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "suggestion": "Please retry your request"
            },
            request_id=request_id,
            context={"error_type": "concurrent_modification", "resource": f"{resource_type}:{resource_id}"}
        )
    
    def handle_resource_limit_error(
        self,
        resource_type: str,
        current_usage: int,
        limit: int,
        request_id: Optional[str] = None
    ) -> StandardizedError:
        """Handle resource limit exceeded errors"""
        
        return self.create_error_response(
            message=f"Resource limit exceeded for {resource_type}",
            code=ErrorCode.RESOURCE_LIMIT_EXCEEDED,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            severity=ErrorSeverity.MEDIUM,
            details={
                "resource_type": resource_type,
                "current_usage": current_usage,
                "limit": limit,
                "suggestion": "Please wait before making additional requests"
            },
            request_id=request_id,
            context={"error_type": "resource_limit", "resource": resource_type}
        )
    
    def handle_pdf_processing_error(
        self,
        error: Exception,
        file_path: str,
        request_id: Optional[str] = None
    ) -> StandardizedError:
        """Handle PDF processing errors"""
        
        error_message = str(error).lower()
        
        if "memory" in error_message or "out of memory" in error_message:
            severity = ErrorSeverity.HIGH
            details = {
                "file_path": file_path,
                "error_type": "memory_error",
                "suggestion": "File may be too large or corrupted"
            }
        elif "corrupted" in error_message or "invalid" in error_message:
            severity = ErrorSeverity.MEDIUM
            details = {
                "file_path": file_path,
                "error_type": "file_corruption",
                "suggestion": "Please check if the PDF file is valid"
            }
        else:
            severity = ErrorSeverity.MEDIUM
            details = {
                "file_path": file_path,
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
        
        return self.create_error_response(
            message="PDF processing failed",
            code=ErrorCode.PDF_PROCESSING_ERROR,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            severity=severity,
            details=details,
            request_id=request_id,
            context={"error_type": "pdf_processing", "file_path": file_path}
        )
    
    def handle_embedding_error(
        self,
        error: Exception,
        text_length: int,
        request_id: Optional[str] = None
    ) -> StandardizedError:
        """Handle embedding generation errors"""
        
        error_message = str(error).lower()
        
        if "rate limit" in error_message or "quota" in error_message:
            severity = ErrorSeverity.HIGH
            details = {
                "text_length": text_length,
                "error_type": "rate_limit",
                "suggestion": "Please wait before making additional requests"
            }
        elif "timeout" in error_message:
            severity = ErrorSeverity.MEDIUM
            details = {
                "text_length": text_length,
                "error_type": "timeout",
                "suggestion": "Text may be too long, try with shorter content"
            }
        else:
            severity = ErrorSeverity.MEDIUM
            details = {
                "text_length": text_length,
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
        
        return self.create_error_response(
            message="Embedding generation failed",
            code=ErrorCode.EMBEDDING_GENERATION_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            severity=severity,
            details=details,
            request_id=request_id,
            context={"error_type": "embedding_generation", "text_length": text_length}
        )
    
    def _track_error(self, code: str, severity: str):
        """Track error count for monitoring"""
        
        if code not in self.error_counts:
            self.error_counts[code] = {
                "total": 0,
                "by_severity": {s: 0 for s in ErrorSeverity.__dict__.values() if not s.startswith('_')}
            }
        
        self.error_counts[code]["total"] += 1
        self.error_counts[code]["by_severity"][severity] += 1
        
        # Check if we need to alert for high severity errors
        if severity in self.severity_thresholds:
            threshold = self.severity_thresholds[severity]
            current_count = self.error_counts[code]["by_severity"][severity]
            
            if current_count >= threshold:
                logger.warning(f"🚨 Error threshold reached for {code}: {current_count} {severity} errors")
    
    def _log_error(
        self,
        message: str,
        code: str,
        severity: str,
        details: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ):
        """Log error with appropriate level based on severity"""
        
        log_context = {
            "error_code": code,
            "severity": severity,
            "details": details,
            "context": context
        }
        
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"🚨 CRITICAL ERROR: {message}", extra=log_context)
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"❌ HIGH ERROR: {message}", extra=log_context)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"⚠️ MEDIUM ERROR: {message}", extra=log_context)
        else:
            logger.info(f"ℹ️ LOW ERROR: {message}", extra=log_context)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        
        return {
            "total_errors": sum(counts["total"] for counts in self.error_counts.values()),
            "errors_by_code": self.error_counts,
            "severity_distribution": self._get_severity_distribution(),
            "most_common_errors": self._get_most_common_errors()
        }
    
    def _get_severity_distribution(self) -> Dict[str, int]:
        """Get distribution of errors by severity"""
        
        distribution = {severity: 0 for severity in ErrorSeverity.__dict__.values() if not severity.startswith('_')}
        
        for code_counts in self.error_counts.values():
            for severity, count in code_counts["by_severity"].items():
                distribution[severity] += count
        
        return distribution
    
    def _get_most_common_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most common error codes"""
        
        error_list = []
        for code, counts in self.error_counts.items():
            error_list.append({
                "code": code,
                "total_count": counts["total"],
                "severity_breakdown": counts["by_severity"]
            })
        
        # Sort by total count
        error_list.sort(key=lambda x: x["total_count"], reverse=True)
        return error_list[:limit]


# Global error handler instance
error_handler = ErrorHandler()


# FastAPI exception handlers
def register_exception_handlers(app):
    """Register exception handlers with FastAPI app"""
    
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request, exc: ValidationError):
        """Handle Pydantic validation errors"""
        error_response = error_handler.handle_validation_error(exc)
        return error_response.to_response()
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        """Handle HTTP exceptions"""
        error_response = error_handler.create_error_response(
            message=exc.detail,
            code=ErrorCode.INTERNAL_ERROR,
            status_code=exc.status_code,
            severity=ErrorSeverity.MEDIUM
        )
        return error_response.to_response()
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        """Handle general exceptions"""
        error_response = error_handler.create_error_response(
            message="An unexpected error occurred",
            code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            severity=ErrorSeverity.HIGH,
            details={
                "error_type": type(exc).__name__,
                "error_message": str(exc)
            }
        )
        return error_response.to_response()


# Utility functions for consistent error handling
def create_error_response(
    message: str,
    code: str = ErrorCode.INTERNAL_ERROR,
    status_code: int = 500,
    severity: str = ErrorSeverity.MEDIUM,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> StandardizedError:
    """Create a standardized error response using the global error handler"""
    return error_handler.create_error_response(
        message=message,
        code=code,
        status_code=status_code,
        severity=severity,
        details=details,
        request_id=request_id,
        context=context
    )


def handle_validation_error(
    validation_error: ValidationError,
    request_id: Optional[str] = None
) -> StandardizedError:
    """Handle Pydantic validation errors using the global error handler"""
    return error_handler.handle_validation_error(validation_error, request_id)


def handle_database_error(
    error: Exception,
    operation: str,
    request_id: Optional[str] = None
) -> StandardizedError:
    """Handle database errors using the global error handler"""
    return error_handler.handle_database_error(error, operation, request_id)


def handle_external_service_error(
    error: Exception,
    service_name: str,
    request_id: Optional[str] = None
) -> StandardizedError:
    """Handle external service errors using the global error handler"""
    return error_handler.handle_external_service_error(error, service_name, request_id)
