from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import logging
from datetime import datetime

from app.core.database import get_async_session
from app.middleware.auth import get_current_teacher, get_current_student
from app.models.user import User
from app.services.llm_providers.policy_manager import policy_manager, PolicyType
from app.services.cost_monitoring_service import cost_monitoring_service
from app.services.content_moderation_service import content_moderation_service
from app.services.llm_providers.llm_router import llm_router

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/llm", tags=["llm-management"])


# Request/Response Models
class PolicySelectionRequest(BaseModel):
    policy_type: PolicyType = Field(..., description="LLM policy type")
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


# Policy Management Endpoints
@router.get("/policies", response_model=Dict[str, Dict[str, Any]], status_code=status.HTTP_200_OK)
async def get_all_policies(
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_teacher),
):
    """Get all available LLM policies"""
    try:
        policies = await policy_manager.get_all_policies()
        return policies
    except Exception as e:
        logger.error(f"Error getting policies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get policies: {str(e)}"
        )


@router.post("/select-policy", response_model=PolicySelectionResponse, status_code=status.HTTP_200_OK)
async def select_policy_for_task(
    request: PolicySelectionRequest,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_teacher),
):
    """Select appropriate policy for a task"""
    try:
        policy_stats = await policy_manager.get_policy_stats(request.policy_type)
        
        # Get available providers for this policy
        available_providers = []
        for provider_name in policy_stats.get("preferred_providers", []):
            if provider_name in llm_router.providers:
                available_providers.append(provider_name)
        
        # Estimate cost
        estimated_cost = llm_router._estimate_cost_by_policy(request.policy_type, 1000)
        
        return PolicySelectionResponse(
            selected_policy=policy_stats,
            available_providers=available_providers,
            estimated_cost=estimated_cost,
            quality_threshold=policy_stats.get("quality_threshold", 0.8),
            latency_threshold=policy_stats.get("latency_threshold", 5.0)
        )
        
    except Exception as e:
        logger.error(f"Error selecting policy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to select policy: {str(e)}"
        )


# Cost Monitoring Endpoints
@router.post("/cost-limits", response_model=CostLimitResponse, status_code=status.HTTP_200_OK)
async def check_cost_limits(
    request: CostLimitRequest,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_teacher),
):
    """Check cost limits for user/organization/endpoint"""
    try:
        # Check limits
        limit_check = await cost_monitoring_service.check_limits(
            user_id=request.user_id,
            organization_id=request.organization_id,
            endpoint=request.endpoint,
            estimated_tokens=1000,
            estimated_cost=0.01
        )
        
        # Get usage report
        usage_report = await cost_monitoring_service.get_usage_report(
            user_id=request.user_id,
            organization_id=request.organization_id,
            endpoint=request.endpoint
        )
        
        return CostLimitResponse(
            limits=usage_report.get("usage", {}),
            current_usage=usage_report.get("usage", {}),
            remaining_budget={},  # Calculate from limits and usage
            degradation_mode=limit_check.get("degradation_mode")
        )
        
    except Exception as e:
        logger.error(f"Error checking cost limits: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check cost limits: {str(e)}"
        )


@router.get("/usage-report/{user_id}", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_usage_report(
    user_id: str,
    organization_id: Optional[str] = Query(None, description="Organization ID"),
    endpoint: Optional[str] = Query(None, description="Endpoint name"),
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_teacher),
):
    """Get detailed usage report"""
    try:
        report = await cost_monitoring_service.get_usage_report(
            user_id=user_id,
            organization_id=organization_id,
            endpoint=endpoint
        )
        return report
        
    except Exception as e:
        logger.error(f"Error getting usage report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage report: {str(e)}"
        )


# Content Moderation Endpoints
@router.post("/moderate-content", response_model=ContentModerationResponse, status_code=status.HTTP_200_OK)
async def moderate_content(
    request: ContentModerationRequest,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_student),
):
    """Moderate content for safety and injection attempts"""
    try:
        result = await content_moderation_service.moderate_content(
            content=request.content,
            content_type=request.content_type,
            user_id=str(user.id)
        )
        
        return ContentModerationResponse(**result)
        
    except Exception as e:
        logger.error(f"Error moderating content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to moderate content: {str(e)}"
        )


@router.get("/moderation-stats", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_moderation_stats(
    user_id: Optional[str] = Query(None, description="User ID"),
    time_period: str = Query("24h", description="Time period"),
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_teacher),
):
    """Get content moderation statistics"""
    try:
        stats = await content_moderation_service.get_moderation_stats(
            user_id=user_id,
            time_period=time_period
        )
        return stats
        
    except Exception as e:
        logger.error(f"Error getting moderation stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get moderation stats: {str(e)}"
        )


@router.get("/user-flagged/{user_id}", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def check_user_flag_status(
    user_id: str,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_teacher),
):
    """Check if user is flagged for violations"""
    try:
        is_flagged = await content_moderation_service.is_user_flagged(user_id)
        return {
            "user_id": user_id,
            "is_flagged": is_flagged,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking user flag status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check user flag status: {str(e)}"
        )


# LLM Health and Status Endpoints
@router.post("/health", response_model=LLMHealthResponse, status_code=status.HTTP_200_OK)
async def get_llm_health(
    request: LLMHealthRequest,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_teacher),
):
    """Get comprehensive LLM system health"""
    try:
        # Get provider health
        health_status = await llm_router.health_check()
        
        # Get policy stats
        policy_stats = await policy_manager.get_all_policies()
        
        # Get moderation stats
        moderation_stats = await content_moderation_service.get_moderation_stats()
        
        return LLMHealthResponse(
            overall_healthy=health_status.get("overall_healthy", False),
            providers=health_status.get("providers", {}),
            cost_status=health_status.get("cost_status", {}),
            policy_stats=policy_stats,
            moderation_stats=moderation_stats
        )
        
    except Exception as e:
        logger.error(f"Error getting LLM health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get LLM health: {str(e)}"
        )


@router.get("/provider-status", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_provider_status(
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_teacher),
):
    """Get detailed provider status"""
    try:
        status = await llm_router.get_provider_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting provider status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider status: {str(e)}"
        )


# Test Endpoints
@router.post("/test-policy", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def test_policy_selection(
    request: PolicySelectionRequest,
    db: AsyncSession = Depends(get_async_session),
    user: User = Depends(get_current_teacher),
):
    """Test policy selection with a sample request"""
    try:
        # Test prompt
        test_prompt = "Generate a simple math question for grade 5 students."
        
        # Test with selected policy
        result = await llm_router.generate_with_fallback(
            task_type=request.task_type,
            prompt=test_prompt,
            system_prompt="You are a helpful math teacher.",
            complexity=request.complexity,
            policy_type=request.policy_type,
            user_id=str(user.id)
        )
        
        return {
            "policy_type": request.policy_type.value,
            "task_type": request.task_type,
            "test_result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error testing policy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test policy: {str(e)}"
        )
