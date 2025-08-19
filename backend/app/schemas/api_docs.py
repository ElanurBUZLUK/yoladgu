from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI
from ..core.config import settings
from ..services.security_service import Permission, Role


class ApiEndpointMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class ApiEndpointCategory(str, Enum):
    AUTHENTICATION = "authentication"
    USERS = "users"
    QUESTIONS = "questions"
    ANSWERS = "answers"
    ANALYTICS = "analytics"
    DASHBOARD = "dashboard"
    PDF = "pdf"
    SCHEDULER = "scheduler"
    MCP = "mcp"


class ApiEndpointSecurity(str, Enum):
    NONE = "none"
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"
    PUBLIC = "public"


class ApiParameter(BaseModel):
    name: str
    type: str
    required: bool = False
    description: Optional[str] = None
    example: Optional[Any] = None
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None


class ApiResponse(BaseModel):
    status_code: int
    description: str
    schema: Optional[str] = None
    example: Optional[Dict[str, Any]] = None


class ApiEndpoint(BaseModel):
    path: str
    method: ApiEndpointMethod
    summary: str
    description: Optional[str] = None
    category: ApiEndpointCategory
    security: ApiEndpointSecurity
    parameters: List[ApiParameter] = []
    request_body: Optional[Dict[str, Any]] = None
    responses: List[ApiResponse] = []
    tags: List[str] = []
    deprecated: bool = False
    rate_limited: bool = False
    cacheable: bool = False


class ApiGroup(BaseModel):
    name: str
    description: str
    endpoints: List[ApiEndpoint]
    version: str = "v1"


class ApiVersion(BaseModel):
    version: str
    status: str  # "stable", "beta", "deprecated"
    release_date: str
    deprecation_date: Optional[str] = None
    groups: List[ApiGroup]


class ApiDocumentation(BaseModel):
    title: str = "Adaptive Learning System API"
    version: str = "1.0.0"
    description: str = "AI-powered adaptive learning system for mathematics and English"
    contact: Dict[str, str] = {
        "name": "API Support",
        "email": "support@adaptivelearning.com"
    }
    license: Dict[str, str] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    servers: List[Dict[str, str]] = [
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.adaptivelearning.com", "description": "Production server"}
    ]
    versions: List[ApiVersion]
    security_schemes: Dict[str, Dict[str, Any]] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }


class ApiExample(BaseModel):
    name: str
    summary: str
    description: str
    endpoint: str
    method: ApiEndpointMethod
    request: Dict[str, Any]
    response: Dict[str, Any]
    curl_command: Optional[str] = None


class ApiTutorial(BaseModel):
    title: str
    description: str
    difficulty: str  # "beginner", "intermediate", "advanced"
    steps: List[Dict[str, Any]]
    examples: List[ApiExample]
    prerequisites: List[str] = []


class ApiChangelog(BaseModel):
    version: str
    date: str
    changes: List[Dict[str, Any]]
    breaking_changes: List[str] = []
    deprecations: List[str] = []
    new_features: List[str] = []
    bug_fixes: List[str] = []


class ApiMetric(BaseModel):
    endpoint: str
    method: ApiEndpointMethod
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float
    time_period: str


class ApiHealthCheck(BaseModel):
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    version: str
    uptime: float
    response_time: float
    database_status: str
    cache_status: str
    external_services: Dict[str, str]
    errors: List[str] = []


class ApiRateLimit(BaseModel):
    endpoint: str
    method: ApiEndpointMethod
    limit: int
    remaining: int
    reset_time: str
    window: str  # "1m", "1h", "1d"


class ApiUsage(BaseModel):
    user_id: str
    endpoint: str
    method: ApiEndpointMethod
    timestamp: str
    response_time: float
    status_code: int
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class ApiSchema(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    properties: Dict[str, Any]
    required: List[str] = []
    examples: List[Dict[str, Any]] = []


class ApiErrorCode(BaseModel):
    code: str
    message: str
    description: str
    http_status: int
    category: str
    suggestions: List[str] = []
    documentation_url: Optional[str] = None


class ApiSDK(BaseModel):
    name: str
    language: str
    version: str
    download_url: str
    documentation_url: str
    examples: List[Dict[str, Any]]
    installation_instructions: str
    dependencies: List[str] = []


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Custom OpenAPI schema with enhanced security and documentation"""
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.api_docs_title,
        version=settings.api_docs_version,
        description=settings.api_docs_description,
        routes=app.routes,
    )
    
    # Enhanced security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from /api/v1/auth/login endpoint"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for service-to-service communication"
        }
    }
    
    # Global security requirement
    openapi_schema["security"] = [
        {"BearerAuth": []}
    ]
    
    # Enhanced info section
    openapi_schema["info"]["contact"] = {
        "name": settings.api_docs_contact_name,
        "email": settings.api_docs_contact_email,
        "url": "https://github.com/adaptive-learning/backend"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    # Add servers
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.adaptive-learning.com",
            "description": "Production server"
        }
    ]
    
    # Add tags with descriptions
    openapi_schema["tags"] = [
        {
            "name": "authentication",
            "description": "User authentication and authorization endpoints"
        },
        {
            "name": "math",
            "description": "Mathematics question and RAG endpoints"
        },
        {
            "name": "english",
            "description": "English question and RAG endpoints"
        },
        {
            "name": "users",
            "description": "User management endpoints"
        },
        {
            "name": "dashboard",
            "description": "Dashboard and analytics endpoints"
        },
        {
            "name": "llm",
            "description": "LLM provider management and policy endpoints"
        },
        {
            "name": "vector",
            "description": "Vector database management endpoints"
        },
        {
            "name": "monitoring",
            "description": "System monitoring and metrics endpoints"
        },
        {
            "name": "pdf",
            "description": "PDF upload and processing endpoints"
        },
        {
            "name": "analytics",
            "description": "Advanced analytics and reporting endpoints"
        },
        {
            "name": "system",
            "description": "System initialization and management endpoints"
        }
    ]
    
    # Add examples for common responses
    openapi_schema["components"]["examples"] = {
        "LoginRequest": {
            "summary": "User login request",
            "value": {
                "email": "student@example.com",
                "password": "password123"
            }
        },
        "LoginResponse": {
            "summary": "Successful login response",
            "value": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "expires_in": 1800,
                "user": {
                    "id": 1,
                    "email": "student@example.com",
                    "full_name": "John Doe",
                    "role": "student",
                    "is_active": True
                }
            }
        },
        "ErrorResponse": {
            "summary": "Error response",
            "value": {
                "detail": "Error message",
                "error_code": "ERROR_CODE",
                "status_code": 400
            }
        },
        "RateLimitResponse": {
            "summary": "Rate limit exceeded response",
            "value": {
                "detail": "Rate limit exceeded",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "retry_after": 60,
                "rate_limit_info": {
                    "limit": 100,
                    "remaining": 0,
                    "reset_time": 1640995200
                }
            }
        }
    }
    
    # Add security requirements to paths
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            operation = openapi_schema["paths"][path][method]
            
            # Add security requirements based on endpoint
            if _requires_auth(path, method):
                operation["security"] = [{"BearerAuth": []}]
            
            # Add role-based security descriptions
            if _is_admin_only(path, method):
                operation["summary"] = f"{operation.get('summary', '')} [Admin Only]"
                operation["description"] = f"{operation.get('description', '')}\n\n**Required Role:** Admin"
            elif _is_teacher_or_admin(path, method):
                operation["summary"] = f"{operation.get('summary', '')} [Teacher/Admin]"
                operation["description"] = f"{operation.get('description', '')}\n\n**Required Role:** Teacher or Admin"
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def _requires_auth(path: str, method: str) -> bool:
    """Check if endpoint requires authentication"""
    # Public endpoints that don't require auth
    public_endpoints = [
        "/health",
        "/health/simple",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/refresh",
    ]
    
    return not any(path.startswith(endpoint) for endpoint in public_endpoints)


def _is_admin_only(path: str, method: str) -> bool:
    """Check if endpoint is admin-only"""
    admin_endpoints = [
        "/api/v1/users/",
        "/api/v1/system/",
        "/api/v1/monitoring/",
        "/api/v1/llm/policies",
        "/api/v1/vector/rebuild-index",
        "/api/v1/cost/limits",
    ]
    
    return any(path.startswith(endpoint) for endpoint in admin_endpoints)


def _is_teacher_or_admin(path: str, method: str) -> bool:
    """Check if endpoint requires teacher or admin role"""
    teacher_admin_endpoints = [
        "/api/v1/questions/",
        "/api/v1/analytics/",
        "/api/v1/dashboard/",
        "/api/v1/moderation/",
    ]
    
    return any(path.startswith(endpoint) for endpoint in teacher_admin_endpoints)


def get_api_version_info() -> Dict[str, Any]:
    """Get API version information"""
    return {
        "version": settings.api_docs_version,
        "environment": settings.environment.value,
        "features": {
            "authentication": True,
            "role_based_access": True,
            "rate_limiting": settings.rate_limit_enabled,
            "monitoring": settings.prometheus_enabled,
            "content_moderation": settings.content_moderation_enabled,
            "cost_monitoring": settings.cost_monitoring_enabled,
            "vector_search": True,
            "llm_integration": True,
            "pdf_processing": True,
            "analytics": True,
        },
        "endpoints": {
            "total": _count_endpoints(),
            "authenticated": _count_authenticated_endpoints(),
            "public": _count_public_endpoints(),
        },
        "security": {
            "jwt_enabled": True,
            "cors_enabled": True,
            "rate_limiting": settings.rate_limit_enabled,
            "content_moderation": settings.content_moderation_enabled,
        }
    }


def _count_endpoints() -> int:
    """Count total endpoints (placeholder)"""
    return 50  # Approximate count


def _count_authenticated_endpoints() -> int:
    """Count authenticated endpoints (placeholder)"""
    return 45  # Approximate count


def _count_public_endpoints() -> int:
    """Count public endpoints (placeholder)"""
    return 5  # Approximate count


def get_permission_matrix() -> Dict[str, Dict[str, List[str]]]:
    """Get permission matrix for different roles"""
    return {
        "student": {
            "permissions": [
                Permission.QUESTION_READ.value,
                Permission.ATTEMPT_READ.value,
                Permission.ATTEMPT_CREATE.value,
                Permission.ATTEMPT_UPDATE.value,
                Permission.ANALYTICS_READ.value,
            ],
            "endpoints": [
                "/api/v1/math/rag/next-question",
                "/api/v1/math/rag/submit-answer",
                "/api/v1/english/rag/next-question",
                "/api/v1/dashboard/overview",
                "/api/v1/users/me",
            ]
        },
        "teacher": {
            "permissions": [
                Permission.USER_READ.value,
                Permission.QUESTION_READ.value,
                Permission.QUESTION_CREATE.value,
                Permission.QUESTION_UPDATE.value,
                Permission.ATTEMPT_READ.value,
                Permission.ATTEMPT_UPDATE.value,
                Permission.ANALYTICS_READ.value,
                Permission.ANALYTICS_CREATE.value,
                Permission.ANALYTICS_UPDATE.value,
                Permission.MODERATION_READ.value,
                Permission.MODERATION_UPDATE.value,
            ],
            "endpoints": [
                "/api/v1/questions/",
                "/api/v1/analytics/",
                "/api/v1/dashboard/",
                "/api/v1/moderation/",
                "/api/v1/users/",
            ]
        },
        "admin": {
            "permissions": [perm.value for perm in Permission],
            "endpoints": [
                "/api/v1/users/",
                "/api/v1/system/",
                "/api/v1/monitoring/",
                "/api/v1/llm/",
                "/api/v1/vector/",
                "/api/v1/cost/",
                "/api/v1/moderation/",
            ]
        }
    }


def get_rate_limit_info() -> Dict[str, Any]:
    """Get rate limiting information"""
    return {
        "enabled": settings.rate_limit_enabled,
        "limits": {
            "default": {
                "requests_per_minute": settings.rate_limit_requests_per_minute,
                "requests_per_hour": settings.rate_limit_requests_per_hour,
                "requests_per_day": settings.rate_limit_requests_per_day,
                "burst_size": settings.rate_limit_burst_size,
            },
            "auth": {
                "requests_per_minute": 10,
                "window": 60,
            },
            "upload": {
                "requests_per_minute": 5,
                "window": 60,
            },
            "admin": {
                "requests_per_minute": 1000,
                "window": 60,
            },
            "student": {
                "requests_per_minute": 200,
                "window": 60,
            },
            "teacher": {
                "requests_per_minute": 500,
                "window": 60,
            },
        },
        "headers": {
            "X-RateLimit-Limit": "Request limit for the time window",
            "X-RateLimit-Remaining": "Remaining requests in the current window",
            "X-RateLimit-Reset": "Time when the rate limit resets (Unix timestamp)",
            "Retry-After": "Seconds to wait before retrying (when rate limited)",
        }
    }
