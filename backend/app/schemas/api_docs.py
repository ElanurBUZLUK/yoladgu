from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


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
