from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

# Request/Response Models (copied from api/v1/llm_management.py for consistency)
class PolicySelectionRequest(BaseModel):
    policy_type: str = Field(..., description="LLM policy type") # Changed to str as PolicyType enum is not available here
    task_type: str = Field(..., description="Task type (e.g., question_generation)")
    complexity: str = Field(default="medium", description="Task complexity")


class PolicySelectionResponse(BaseModel):
    selected_policy: Dict[str, Any]
    available_providers: List[str]
    estimated_cost: float
    quality_threshold: float
    latency_threshold: float


class CostLimitRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    organization_id: Optional[str] = Field(None, description="Organization ID")
    endpoint: Optional[str] = Field(None, description="Endpoint name")


class CostLimitResponse(BaseModel):
    limits: Dict[str, Any]
    current_usage: Dict[str, Any]
    remaining_budget: Dict[str, Any]
    degradation_mode: Optional[str] = None


class ContentModerationRequest(BaseModel):
    content: str = Field(..., description="Content to moderate")
    content_type: str = Field(default="user_input", description="Content type")


class ContentModerationResponse(BaseModel):
    safe: bool
    risk_level: str
    issues: List[Dict[str, Any]]
    injection_detected: bool
    moderated_content: str


class LLMHealthRequest(BaseModel):
    include_provider_details: bool = Field(default=False, description="Include provider details")


class LLMHealthResponse(BaseModel):
    overall_healthy: bool
    providers: Dict[str, Any]
    cost_status: Dict[str, Any]
    policy_stats: Dict[str, Any]
    moderation_stats: Dict[str, Any]


# New Response Models

class GetAllPoliciesResponse(BaseModel):
    policies: Dict[str, Dict[str, Any]]

class UsageReportEntry(BaseModel):
    user_id: str
    organization_id: Optional[str]
    endpoint: str
    total_cost: float
    total_tokens: int
    request_count: int
    last_updated: datetime

class GetUsageReportResponse(BaseModel):
    report: Dict[str, Any] # This will be the structure returned by cost_monitoring_service.get_usage_report

class ModerationStatsEntry(BaseModel):
    total_requests: int
    flagged_requests: int
    risk_levels: Dict[str, int]
    content_types: Dict[str, int]
    time_period: str

class GetModerationStatsResponse(BaseModel):
    stats: Dict[str, Any] # This will be the structure returned by content_moderation_service.get_moderation_stats

class CheckUserFlagStatusResponse(BaseModel):
    user_id: str
    is_flagged: bool
    timestamp: str

class ProviderStatusEntry(BaseModel):
    provider_name: str
    status: str
    last_checked: datetime
    latency_ms: Optional[float]
    error_rate: Optional[float]
    available_models: List[str]

class GetProviderStatusResponse(BaseModel):
    overall_status: str
    providers: Dict[str, ProviderStatusEntry]

class TestPolicySelectionResponse(BaseModel):
    policy_type: str
    task_type: str
    test_result: Dict[str, Any]
    timestamp: str
