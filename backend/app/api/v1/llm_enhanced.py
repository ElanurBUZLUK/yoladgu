"""
Enhanced LLM API endpoints with advanced error handling and optimization
"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.llm_enhanced_service import enhanced_llm_service, RetryConfig
from app.core.error_handling import ErrorHandler, ErrorCode, ErrorSeverity

router = APIRouter(prefix="/api/v1/llm-enhanced", tags=["LLM Enhanced"])
error_handler = ErrorHandler()


class LLMRequest(BaseModel):
    prompt: str = Field(..., description="User prompt")
    system_prompt: str = Field("", description="System prompt")
    max_tokens: int = Field(1000, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Temperature for generation")
    model: Optional[str] = Field(None, description="Preferred model")
    fallback_providers: Optional[List[str]] = Field(None, description="Fallback providers")
    max_retries: int = Field(3, description="Maximum retry attempts")
    timeout: float = Field(30.0, description="Request timeout in seconds")


class TokenUsageResponse(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float


class LLMResponse(BaseModel):
    content: str
    success: bool
    provider: str
    model: str
    token_usage: TokenUsageResponse
    response_time: float
    attempt_count: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PerformanceMetricsResponse(BaseModel):
    overview: Dict[str, Any]
    provider_distribution: Dict[str, int]
    error_analysis: Dict[str, Any]
    retry_performance: Dict[str, Any]
    circuit_breaker_status: Dict[str, Any]


@router.post("/generate", response_model=LLMResponse)
async def generate_with_enhanced_features(request: LLMRequest):
    """
    Generate LLM response with enhanced error handling, retry logic, and optimization
    """
    try:
        # Create retry configuration
        retry_config = RetryConfig(
            max_retries=request.max_retries,
            initial_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=60.0,
            jitter=True
        )
        
        # Generate response with enhanced service
        response = await enhanced_llm_service.generate_with_retry(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            model=request.model,
            retry_config=retry_config,
            fallback_providers=request.fallback_providers
        )
        
        # Convert to API response format
        return LLMResponse(
            content=response.content,
            success=response.success,
            provider=response.provider,
            model=response.model,
            token_usage=TokenUsageResponse(
                prompt_tokens=response.token_usage.prompt_tokens,
                completion_tokens=response.token_usage.completion_tokens,
                total_tokens=response.token_usage.total_tokens,
                estimated_cost=response.token_usage.estimated_cost
            ),
            response_time=response.response_time,
            attempt_count=response.attempt_count,
            error_type=response.error_type.value if response.error_type else None,
            error_message=response.error_message,
            metadata=response.metadata
        )
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.LLM_SERVICE_ERROR,
            message="Failed to generate LLM response",
            severity=ErrorSeverity.HIGH,
            context={
                "prompt_length": len(request.prompt),
                "model": request.model,
                "max_tokens": request.max_tokens
            }
        )


@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_llm_performance_metrics():
    """
    Get comprehensive LLM service performance metrics
    """
    try:
        metrics = await enhanced_llm_service.get_performance_metrics()
        return PerformanceMetricsResponse(**metrics)
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.LLM_SERVICE_ERROR,
            message="Failed to get LLM performance metrics",
            severity=ErrorSeverity.LOW,
            context={}
        )


class TokenCountRequest(BaseModel):
    text: str = Field(..., description="Text to count tokens for")
    model: str = Field("gpt-4", description="Model to count tokens for")

@router.post("/token-count")
async def count_tokens(request: TokenCountRequest):
    """
    Count tokens in text for specific model
    """
    try:
        token_count = enhanced_llm_service.token_counter.count_tokens(request.text, request.model)
        
        return {
            "text_length": len(request.text),
            "token_count": token_count,
            "model": request.model,
            "ratio": round(len(request.text) / token_count if token_count > 0 else 0, 2)
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.LLM_SERVICE_ERROR,
            message="Failed to count tokens",
            severity=ErrorSeverity.LOW,
            context={
                "text_length": len(request.text),
                "model": request.model
            }
        )


class OptimizeContextRequest(BaseModel):
    system_prompt: str = Field(..., description="System prompt")
    user_prompt: str = Field(..., description="User prompt")
    max_tokens: int = Field(4000, description="Maximum tokens")
    model: str = Field("gpt-4", description="Model to optimize for")

@router.post("/optimize-context")
async def optimize_context(request: OptimizeContextRequest):
    """
    Optimize context to fit within token limits
    """
    try:
        optimized_system, optimized_user = enhanced_llm_service.token_counter.optimize_context(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            max_tokens=request.max_tokens,
            model=request.model
        )
        
        original_tokens = (
            enhanced_llm_service.token_counter.count_tokens(request.system_prompt, request.model) +
            enhanced_llm_service.token_counter.count_tokens(request.user_prompt, request.model)
        )
        
        optimized_tokens = (
            enhanced_llm_service.token_counter.count_tokens(optimized_system, request.model) +
            enhanced_llm_service.token_counter.count_tokens(optimized_user, request.model)
        )
        
        return {
            "original": {
                "system_prompt": request.system_prompt,
                "user_prompt": request.user_prompt,
                "total_tokens": original_tokens
            },
            "optimized": {
                "system_prompt": optimized_system,
                "user_prompt": optimized_user,
                "total_tokens": optimized_tokens
            },
            "optimization": {
                "tokens_saved": original_tokens - optimized_tokens,
                "reduction_percentage": round(
                    ((original_tokens - optimized_tokens) / original_tokens * 100) 
                    if original_tokens > 0 else 0, 2
                )
            }
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.LLM_SERVICE_ERROR,
            message="Failed to optimize context",
            severity=ErrorSeverity.MEDIUM,
            context={
                "system_prompt_length": len(request.system_prompt),
                "user_prompt_length": len(request.user_prompt),
                "max_tokens": request.max_tokens,
                "model": request.model
            }
        )


@router.post("/circuit-breaker/reset")
async def reset_circuit_breaker(provider: str):
    """
    Reset circuit breaker for specific provider
    """
    try:
        if provider in enhanced_llm_service.circuit_breakers:
            enhanced_llm_service.circuit_breakers[provider].failure_count = 0
            enhanced_llm_service.circuit_breakers[provider].state = "CLOSED"
            
            return {
                "success": True,
                "message": f"Circuit breaker reset for provider: {provider}",
                "provider": provider
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider not found: {provider}"
            )
            
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.LLM_SERVICE_ERROR,
            message="Failed to reset circuit breaker",
            severity=ErrorSeverity.MEDIUM,
            context={"provider": provider}
        )


@router.get("/health")
async def llm_service_health():
    """
    Get LLM service health status
    """
    try:
        metrics = await enhanced_llm_service.get_performance_metrics()
        
        # Determine health status
        success_rate = metrics["overview"]["success_rate"]
        avg_response_time = metrics["overview"]["average_response_time"]
        
        health_status = "healthy"
        if success_rate < 0.8:
            health_status = "degraded"
        if success_rate < 0.5 or avg_response_time > 10:
            health_status = "unhealthy"
        
        # Check circuit breaker status
        open_circuit_breakers = [
            provider for provider, status in metrics["circuit_breaker_status"].items()
            if status["state"] == "OPEN"
        ]
        
        return {
            "status": health_status,
            "timestamp": metrics["overview"],
            "circuit_breakers": {
                "open_count": len(open_circuit_breakers),
                "open_providers": open_circuit_breakers
            },
            "performance": {
                "success_rate": success_rate,
                "average_response_time": avg_response_time,
                "total_requests": metrics["overview"]["total_requests"]
            }
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.LLM_SERVICE_ERROR,
            message="Failed to get LLM service health",
            severity=ErrorSeverity.LOW,
            context={}
        )
