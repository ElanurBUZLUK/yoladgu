"""
Custom Exception Classes for Better Error Handling
"""

from typing import Any, Dict, Optional

import structlog
from fastapi import HTTPException, status

logger = structlog.get_logger()


class BaseServiceException(Exception):
    """Base exception for all service-level errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DatabaseError(BaseServiceException):
    """Database operation errors"""

    pass


class Neo4jError(BaseServiceException):
    """Neo4j graph database errors"""

    pass


class RedisError(BaseServiceException):
    """Redis cache/stream errors"""

    pass


class MLModelError(BaseServiceException):
    """Machine learning model errors"""

    pass


class EmbeddingError(BaseServiceException):
    """Embedding service errors"""

    pass


class LLMServiceError(BaseServiceException):
    """LLM service errors"""

    pass


class ValidationError(BaseServiceException):
    """Data validation errors"""

    pass


class AuthenticationError(BaseServiceException):
    """Authentication errors"""

    pass


class AuthorizationError(BaseServiceException):
    """Authorization errors"""

    pass


# HTTP Exception Mappers
def handle_database_error(e: DatabaseError) -> HTTPException:
    """Convert database error to HTTP exception"""
    logger.error("database_error", error=e.message, details=e.details)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={"error": "Database operation failed", "message": e.message},
    )


def handle_neo4j_error(e: Neo4jError) -> HTTPException:
    """Convert Neo4j error to HTTP exception"""
    logger.error("neo4j_error", error=e.message, details=e.details)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": "Graph database unavailable", "message": e.message},
    )


def handle_redis_error(e: RedisError) -> HTTPException:
    """Convert Redis error to HTTP exception"""
    logger.error("redis_error", error=e.message, details=e.details)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": "Cache service unavailable", "message": e.message},
    )


def handle_ml_model_error(e: MLModelError) -> HTTPException:
    """Convert ML model error to HTTP exception"""
    logger.error("ml_model_error", error=e.message, details=e.details)
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"error": "ML model processing failed", "message": e.message},
    )


def handle_embedding_error(e: EmbeddingError) -> HTTPException:
    """Convert embedding error to HTTP exception"""
    logger.error("embedding_error", error=e.message, details=e.details)
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"error": "Embedding generation failed", "message": e.message},
    )


def handle_llm_service_error(e: LLMServiceError) -> HTTPException:
    """Convert LLM service error to HTTP exception"""
    logger.error("llm_service_error", error=e.message, details=e.details)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": "AI service unavailable", "message": e.message},
    )


def handle_validation_error(e: ValidationError) -> HTTPException:
    """Convert validation error to HTTP exception"""
    logger.error("validation_error", error=e.message, details=e.details)
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": "Validation failed", "message": e.message},
    )


def handle_authentication_error(e: AuthenticationError) -> HTTPException:
    """Convert authentication error to HTTP exception"""
    logger.error("authentication_error", error=e.message, details=e.details)
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "Authentication failed", "message": e.message},
    )


def handle_authorization_error(e: AuthorizationError) -> HTTPException:
    """Convert authorization error to HTTP exception"""
    logger.error("authorization_error", error=e.message, details=e.details)
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"error": "Authorization failed", "message": e.message},
    )


# Generic handler
def handle_generic_error(e: Exception) -> HTTPException:
    """Handle unexpected errors with proper logging"""
    logger.error("unexpected_error", error=str(e), error_type=type(e).__name__)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
        },
    )


# Error mapper dictionary
ERROR_HANDLERS = {
    DatabaseError: handle_database_error,
    Neo4jError: handle_neo4j_error,
    RedisError: handle_redis_error,
    MLModelError: handle_ml_model_error,
    EmbeddingError: handle_embedding_error,
    LLMServiceError: handle_llm_service_error,
    ValidationError: handle_validation_error,
    AuthenticationError: handle_authentication_error,
    AuthorizationError: handle_authorization_error,
}


def convert_to_http_exception(e: Exception) -> HTTPException:
    """Convert any exception to appropriate HTTP exception"""
    handler = ERROR_HANDLERS.get(type(e))
    if handler:
        return handler(e)
    else:
        return handle_generic_error(e)
