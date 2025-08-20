from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ErrorType(str, Enum):
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND_ERROR = "not_found_error"
    CONFLICT_ERROR = "conflict_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    INTERNAL_ERROR = "internal_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    DATABASE_ERROR = "database_error"
    FILE_UPLOAD_ERROR = "file_upload_error"
    LLM_ERROR = "llm_error"
    MCP_ERROR = "mcp_error"


class ErrorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str
    code: Optional[str] = None
    value: Optional[Any] = None


class ErrorResponse(BaseModel):
    error_type: ErrorType
    message: str
    details: Optional[List[ErrorDetail]] = None
    error_code: Optional[str] = None
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = None
    path: Optional[str] = None
    method: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = None
    suggestions: Optional[List[str]] = None
    documentation_url: Optional[str] = None


class ValidationErrorResponse(ErrorResponse):
    error_type: ErrorType = ErrorType.VALIDATION_ERROR
    severity: ErrorSeverity = ErrorSeverity.LOW


class AuthenticationErrorResponse(ErrorResponse):
    error_type: ErrorType = ErrorType.AUTHENTICATION_ERROR
    severity: ErrorSeverity = ErrorSeverity.HIGH


class AuthorizationErrorResponse(ErrorResponse):
    error_type: ErrorType = ErrorType.AUTHORIZATION_ERROR
    severity: ErrorSeverity = ErrorSeverity.HIGH


class NotFoundErrorResponse(ErrorResponse):
    error_type: ErrorType = ErrorType.NOT_FOUND_ERROR
    severity: ErrorSeverity = ErrorSeverity.LOW


class ConflictErrorResponse(ErrorResponse):
    error_type: ErrorType = ErrorType.CONFLICT_ERROR
    severity: ErrorSeverity = ErrorSeverity.MEDIUM


class RateLimitErrorResponse(ErrorResponse):
    error_type: ErrorType = ErrorType.RATE_LIMIT_ERROR
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    retry_after: Optional[int] = None
    rate_limit_info: Optional[Dict[str, Any]] = None


class InternalErrorResponse(ErrorResponse):
    error_type: ErrorType = ErrorType.INTERNAL_ERROR
    severity: ErrorSeverity = ErrorSeverity.CRITICAL
    error_id: Optional[str] = None


class ExternalServiceErrorResponse(ErrorResponse):
    error_type: ErrorType = ErrorType.EXTERNAL_SERVICE_ERROR
    severity: ErrorSeverity = ErrorSeverity.HIGH
    service_name: Optional[str] = None
    service_endpoint: Optional[str] = None
    retry_available: bool = True
    retry_after: Optional[int] = None


class DatabaseErrorResponse(ErrorResponse):
    error_type: ErrorType = ErrorType.DATABASE_ERROR
    severity: ErrorSeverity = ErrorSeverity.HIGH
    operation: Optional[str] = None
    table: Optional[str] = None
    constraint: Optional[str] = None


class FileUploadErrorResponse(ErrorResponse):
    error_type: ErrorType = ErrorType.FILE_UPLOAD_ERROR
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    allowed_types: Optional[List[str]] = None
    max_size: Optional[int] = None


class LLMErrorResponse(ErrorResponse):
    error_type: ErrorType = ErrorType.LLM_ERROR
    severity: ErrorSeverity = ErrorSeverity.HIGH
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_length: Optional[int] = None
    response_length: Optional[int] = None
    cost_estimate: Optional[float] = None


class MCPErrorResponse(ErrorResponse):
    error_type: ErrorType = ErrorType.MCP_ERROR
    severity: ErrorSeverity = ErrorSeverity.HIGH
    tool_name: Optional[str] = None
    server_name: Optional[str] = None
    connection_status: Optional[str] = None


class ErrorLog(BaseModel):
    error_id: str
    error_type: ErrorType
    message: str
    severity: ErrorSeverity
    timestamp: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_path: Optional[str] = None
    request_method: Optional[str] = None
    request_headers: Optional[Dict[str, str]] = None
    request_body: Optional[Dict[str, Any]] = None
    response_status: Optional[int] = None
    response_body: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    environment: Optional[str] = None
    version: Optional[str] = None


class ErrorSummary(BaseModel):
    total_errors: int
    errors_by_type: Dict[str, int]
    errors_by_severity: Dict[str, int]
    recent_errors: List[ErrorLog]
    most_common_errors: List[Dict[str, Any]]


class ErrorMetrics(BaseModel):
    error_rate: float  # percentage
    error_count: int
    success_rate: float  # percentage
    average_response_time: float
    p95_response_time: float
    p99_response_time: float
    time_period: str  # "1h", "24h", "7d", "30d"


class ErrorAlert(BaseModel):
    alert_id: str
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    threshold: int
    current_count: int
    time_window: str
    triggered_at: str
    resolved_at: Optional[str] = None
    notification_sent: bool = False
    escalation_level: int = 1


class ErrorRecoveryAction(BaseModel):
    action_id: str
    error_type: ErrorType
    action_type: str  # "retry", "fallback", "circuit_breaker", "manual_intervention"
    description: str
    parameters: Optional[Dict[str, Any]] = None
    success_rate: Optional[float] = None
    average_recovery_time: Optional[float] = None
    last_used: Optional[str] = None


class ErrorPreventionRule(BaseModel):
    rule_id: str
    name: str
    description: str
    error_type: ErrorType
    conditions: List[Dict[str, Any]]
    prevention_action: str
    enabled: bool = True
    priority: int = 1
    created_at: str
    updated_at: str
