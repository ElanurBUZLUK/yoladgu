"""
Enhanced LLM Service with Error Handling, Retry Mechanisms, and Token Management
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import json
import tiktoken
from dataclasses import dataclass, field

from app.services.llm_providers.llm_router import llm_router as llm_gateway
from app.services.llm_load_balancer import llm_load_balancer
from app.services.llm_rate_limiter import llm_rate_limiter
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMErrorType(Enum):
    """LLM Error Categories"""
    RATE_LIMIT = "rate_limit"
    QUOTA_EXCEEDED = "quota_exceeded"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    CONTENT_POLICY_VIOLATION = "content_policy_violation"
    UNKNOWN_ERROR = "unknown_error"


class LLMProvider(Enum):
    """Supported LLM Providers"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    LOCAL_MODEL = "local_model"


@dataclass
class RetryConfig:
    """Retry configuration for LLM calls"""
    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 60.0
    jitter: bool = True


@dataclass
class TokenUsage:
    """Token usage tracking"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0


@dataclass
class LLMResponse:
    """Enhanced LLM response with metadata"""
    content: str
    success: bool
    provider: str
    model: str
    token_usage: TokenUsage
    response_time: float
    attempt_count: int
    error_type: Optional[LLMErrorType] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker pattern for LLM providers"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class TokenCounter:
    """Token counting and optimization"""
    
    def __init__(self):
        self.encoders = {}
        self._load_encoders()
    
    def _load_encoders(self):
        """Load tokenizers for different models"""
        try:
            self.encoders["gpt-4"] = tiktoken.encoding_for_model("gpt-4")
            self.encoders["gpt-3.5-turbo"] = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.encoders["text-embedding-ada-002"] = tiktoken.encoding_for_model("text-embedding-ada-002")
        except Exception as e:
            logger.warning(f"Failed to load some encoders: {e}")
            # Fallback to cl100k_base encoding
            self.encoders["default"] = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text for specific model"""
        try:
            encoder = self.encoders.get(model, self.encoders.get("default"))
            if encoder:
                return len(encoder.encode(text))
            else:
                # Rough estimation: 1 token ≈ 4 characters
                return len(text) // 4
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            return len(text) // 4
    
    def optimize_context(self, 
                        system_prompt: str, 
                        user_prompt: str, 
                        max_tokens: int = 4000,
                        model: str = "gpt-4") -> tuple[str, str]:
        """Optimize context to fit within token limits"""
        
        system_tokens = self.count_tokens(system_prompt, model)
        user_tokens = self.count_tokens(user_prompt, model)
        total_tokens = system_tokens + user_tokens
        
        if total_tokens <= max_tokens:
            return system_prompt, user_prompt
        
        # If system prompt is too long, truncate it
        if system_tokens > max_tokens // 2:
            target_system_tokens = max_tokens // 2
            system_prompt = self._truncate_text(system_prompt, target_system_tokens, model)
        
        # Recalculate and truncate user prompt if needed
        system_tokens = self.count_tokens(system_prompt, model)
        remaining_tokens = max_tokens - system_tokens - 100  # Buffer for response
        
        if self.count_tokens(user_prompt, model) > remaining_tokens:
            user_prompt = self._truncate_text(user_prompt, remaining_tokens, model)
        
        return system_prompt, user_prompt
    
    def _truncate_text(self, text: str, max_tokens: int, model: str) -> str:
        """Truncate text to fit within token limit"""
        encoder = self.encoders.get(model, self.encoders.get("default"))
        if not encoder:
            # Rough truncation
            return text[:max_tokens * 4]
        
        tokens = encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return encoder.decode(truncated_tokens)


class LLMCostCalculator:
    """Calculate LLM usage costs"""
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }
    
    def calculate_cost(self, 
                      model: str, 
                      prompt_tokens: int, 
                      completion_tokens: int) -> float:
        """Calculate cost for LLM usage"""
        
        pricing = self.PRICING.get(model, {"input": 0.001, "output": 0.002})
        
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost


class EnhancedLLMService:
    """Enhanced LLM service with advanced error handling and optimization"""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.token_counter = TokenCounter()
        self.cost_calculator = LLMCostCalculator()
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0,
            "provider_usage": {},
            "error_distribution": {},
            "retry_statistics": {
                "total_retries": 0,
                "successful_retries": 0,
                "failed_retries": 0
            }
        }
        
        # Initialize circuit breakers for each provider
        for provider in LLMProvider:
            self.circuit_breakers[provider.value] = CircuitBreaker()
    
    async def generate_with_retry(self,
                                 prompt: str,
                                 system_prompt: str = "",
                                 max_tokens: int = 1000,
                                 temperature: float = 0.7,
                                 model: Optional[str] = None,
                                 retry_config: Optional[RetryConfig] = None,
                                 fallback_providers: Optional[List[str]] = None,
                                 user_id: Optional[str] = None) -> LLMResponse:
        """
        Generate LLM response with retry logic and fallback providers
        """
        
        retry_config = retry_config or RetryConfig()
        fallback_providers = fallback_providers or ["openai_gpt4", "openai_gpt35", "anthropic_claude"]
        
        start_time = time.time()
        last_error = None
        attempt_count = 0
        
        # Use load balancer to select optimal provider
        if not model:
            selected_provider = await llm_load_balancer.select_provider(
                requirements=["text-generation"],
                preferred_provider=model
            )
            if not selected_provider:
                return LLMResponse(
                    content="",
                    success=False,
                    provider="none",
                    model="none",
                    token_usage=TokenUsage(),
                    response_time=time.time() - start_time,
                    attempt_count=1,
                    error_type=LLMErrorType.UNKNOWN_ERROR,
                    error_message="No available providers"
                )
            providers_to_try = [selected_provider] + (fallback_providers or [])
        else:
            providers_to_try = [model] + (fallback_providers or [])
        
        for provider in providers_to_try:
            circuit_breaker = self.circuit_breakers.get(provider)
            
            if circuit_breaker and not circuit_breaker.can_execute():
                logger.warning(f"Circuit breaker OPEN for provider {provider}")
                continue
            
            # Check rate limits before attempting
            tokens_estimate = self.token_counter.count_tokens(system_prompt + prompt, provider)
            cost_estimate = self.cost_calculator.calculate_cost(provider, tokens_estimate, max_tokens)
            
            rate_limit_result = await llm_rate_limiter.check_rate_limit(
                provider=provider,
                user_id=user_id,
                tokens=tokens_estimate,
                cost=cost_estimate
            )
            
            if not rate_limit_result.allowed:
                logger.warning(f"Rate limit exceeded for provider {provider}: {rate_limit_result.message}")
                continue
            
            # Record request start with load balancer
            await llm_load_balancer.record_request_start(provider)
            
            # Retry logic for current provider
            for attempt in range(retry_config.max_retries + 1):
                attempt_count += 1
                
                try:
                    # Optimize context for token limits
                    optimized_system, optimized_prompt = self.token_counter.optimize_context(
                        system_prompt, prompt, max_tokens * 3, provider
                    )
                    
                    # Make LLM call
                    response = await self._make_llm_call(
                        prompt=optimized_prompt,
                        system_prompt=optimized_system,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        provider=provider
                    )
                    
                    if response.success:
                        # Record success
                        if circuit_breaker:
                            circuit_breaker.record_success()
                        
                        response.attempt_count = attempt_count
                        response.response_time = time.time() - start_time
                        
                        # Record with load balancer
                        await llm_load_balancer.record_request_end(
                            provider=provider,
                            success=True,
                            response_time=response.response_time,
                            cost=response.token_usage.estimated_cost
                        )
                        
                        # Record with rate limiter
                        await llm_rate_limiter.record_request(
                            provider=provider,
                            user_id=user_id,
                            tokens=response.token_usage.total_tokens,
                            cost=response.token_usage.estimated_cost
                        )
                        
                        # Update metrics
                        self._update_metrics(response, success=True, attempt_count=attempt_count)
                        
                        return response
                    else:
                        last_error = response.error_message
                        error_type = self._classify_error(response.error_message)
                        
                        # Check if retry is appropriate
                        if not self._should_retry(error_type, attempt):
                            break
                        
                        # Wait before retry
                        if attempt < retry_config.max_retries:
                            delay = self._calculate_retry_delay(
                                attempt, retry_config
                            )
                            await asyncio.sleep(delay)
                
                except Exception as e:
                    last_error = str(e)
                    error_type = self._classify_error(str(e))
                    
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    
                    # Check if retry is appropriate
                    if not self._should_retry(error_type, attempt):
                        break
                    
                    # Wait before retry
                    if attempt < retry_config.max_retries:
                        delay = self._calculate_retry_delay(attempt, retry_config)
                        await asyncio.sleep(delay)
        
        # All providers and retries failed
        response = LLMResponse(
            content="",
            success=False,
            provider="none",
            model="none",
            token_usage=TokenUsage(),
            response_time=time.time() - start_time,
            attempt_count=attempt_count,
            error_type=LLMErrorType.UNKNOWN_ERROR,
            error_message=f"All providers failed. Last error: {last_error}"
        )
        
        self._update_metrics(response, success=False, attempt_count=attempt_count)
        return response
    
    async def _make_llm_call(self,
                            prompt: str,
                            system_prompt: str,
                            max_tokens: int,
                            temperature: float,
                            provider: str) -> LLMResponse:
        """Make actual LLM API call"""
        
        try:
            # Count tokens before call
            prompt_tokens = self.token_counter.count_tokens(
                system_prompt + prompt, provider
            )
            
            # Call LLM gateway
            response = await llm_gateway.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                provider=provider
            )
            
            if response.get("success", False):
                content = response.get("content", "")
                completion_tokens = self.token_counter.count_tokens(content, provider)
                
                # Calculate cost
                cost = self.cost_calculator.calculate_cost(
                    model=provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
                
                token_usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    estimated_cost=cost
                )
                
                return LLMResponse(
                    content=content,
                    success=True,
                    provider=provider,
                    model=provider,
                    token_usage=token_usage,
                    response_time=0.0,  # Will be set by caller
                    attempt_count=0,    # Will be set by caller
                    metadata=response.get("metadata", {})
                )
            else:
                return LLMResponse(
                    content="",
                    success=False,
                    provider=provider,
                    model=provider,
                    token_usage=TokenUsage(),
                    response_time=0.0,
                    attempt_count=0,
                    error_message=response.get("error", "Unknown error")
                )
        
        except Exception as e:
            return LLMResponse(
                content="",
                success=False,
                provider=provider,
                model=provider,
                token_usage=TokenUsage(),
                response_time=0.0,
                attempt_count=0,
                error_message=str(e)
            )
    
    def _classify_error(self, error_message: str) -> LLMErrorType:
        """Classify error type based on error message"""
        
        error_lower = error_message.lower()
        
        if "rate limit" in error_lower or "rate_limit" in error_lower:
            return LLMErrorType.RATE_LIMIT
        elif "quota" in error_lower or "billing" in error_lower:
            return LLMErrorType.QUOTA_EXCEEDED
        elif "timeout" in error_lower:
            return LLMErrorType.TIMEOUT
        elif "network" in error_lower or "connection" in error_lower:
            return LLMErrorType.NETWORK_ERROR
        elif "context_length" in error_lower or "too long" in error_lower:
            return LLMErrorType.CONTEXT_LENGTH_EXCEEDED
        elif "content policy" in error_lower or "inappropriate" in error_lower:
            return LLMErrorType.CONTENT_POLICY_VIOLATION
        elif "validation" in error_lower:
            return LLMErrorType.VALIDATION_ERROR
        else:
            return LLMErrorType.UNKNOWN_ERROR
    
    def _should_retry(self, error_type: LLMErrorType, attempt: int) -> bool:
        """Determine if error is retryable"""
        
        # Don't retry these error types
        non_retryable = {
            LLMErrorType.QUOTA_EXCEEDED,
            LLMErrorType.CONTENT_POLICY_VIOLATION,
            LLMErrorType.VALIDATION_ERROR,
            LLMErrorType.CONTEXT_LENGTH_EXCEEDED
        }
        
        if error_type in non_retryable:
            return False
        
        # Rate limit errors get longer delays
        if error_type == LLMErrorType.RATE_LIMIT and attempt >= 2:
            return False
        
        return True
    
    def _calculate_retry_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate exponential backoff delay"""
        
        delay = config.initial_delay * (config.backoff_multiplier ** attempt)
        delay = min(delay, config.max_delay)
        
        if config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay
    
    def _update_metrics(self, response: LLMResponse, success: bool, attempt_count: int):
        """Update performance metrics"""
        
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
            self.metrics["total_tokens"] += response.token_usage.total_tokens
            self.metrics["total_cost"] += response.token_usage.estimated_cost
        else:
            self.metrics["failed_requests"] += 1
        
        # Update provider usage
        provider = response.provider
        if provider not in self.metrics["provider_usage"]:
            self.metrics["provider_usage"][provider] = 0
        self.metrics["provider_usage"][provider] += 1
        
        # Update error distribution
        if not success and response.error_type:
            error_type = response.error_type.value
            if error_type not in self.metrics["error_distribution"]:
                self.metrics["error_distribution"][error_type] = 0
            self.metrics["error_distribution"][error_type] += 1
        
        # Update retry statistics
        if attempt_count > 1:
            self.metrics["retry_statistics"]["total_retries"] += attempt_count - 1
            if success:
                self.metrics["retry_statistics"]["successful_retries"] += 1
            else:
                self.metrics["retry_statistics"]["failed_retries"] += 1
        
        # Update average response time
        total_requests = self.metrics["total_requests"]
        current_avg = self.metrics["average_response_time"]
        new_avg = ((current_avg * (total_requests - 1)) + response.response_time) / total_requests
        self.metrics["average_response_time"] = new_avg
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        success_rate = (
            self.metrics["successful_requests"] / self.metrics["total_requests"]
            if self.metrics["total_requests"] > 0 else 0.0
        )
        
        retry_success_rate = (
            self.metrics["retry_statistics"]["successful_retries"] / 
            self.metrics["retry_statistics"]["total_retries"]
            if self.metrics["retry_statistics"]["total_retries"] > 0 else 0.0
        )
        
        return {
            "overview": {
                "total_requests": self.metrics["total_requests"],
                "success_rate": round(success_rate, 4),
                "average_response_time": round(self.metrics["average_response_time"], 3),
                "total_tokens_used": self.metrics["total_tokens"],
                "estimated_total_cost": round(self.metrics["total_cost"], 4)
            },
            "provider_distribution": self.metrics["provider_usage"],
            "error_analysis": {
                "error_distribution": self.metrics["error_distribution"],
                "failed_requests": self.metrics["failed_requests"]
            },
            "retry_performance": {
                "total_retries": self.metrics["retry_statistics"]["total_retries"],
                "retry_success_rate": round(retry_success_rate, 4),
                "failed_retries": self.metrics["retry_statistics"]["failed_retries"]
            },
            "circuit_breaker_status": {
                provider: {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "last_failure": cb.last_failure_time
                }
                for provider, cb in self.circuit_breakers.items()
            }
        }


# Global instance
enhanced_llm_service = EnhancedLLMService()
