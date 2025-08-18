from typing import Dict, Any, List, Optional, Union
from enum import Enum
import asyncio
import logging
from datetime import datetime, timedelta

from app.core.config import settings
from app.core.cache import cache_service
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class PolicyType(str, Enum):
    """LLM Provider Policy Types"""
    CHEAP_FAST = "cheap-fast"      # OpenAI small models, low cost, fast
    HIGH_QUALITY = "high-quality"  # Claude Sonnet/GPT-4, high quality, moderate cost
    OFFLINE_FALLBACK = "offline-fallback"  # Local models or templates
    BALANCED = "balanced"          # Default balanced approach


class ProviderPolicy:
    """Provider selection policy configuration"""
    
    def __init__(
        self,
        policy_type: PolicyType,
        preferred_providers: List[str],
        max_cost_per_request: float,
        quality_threshold: float,
        latency_threshold: float,
        fallback_strategy: str = "template"
    ):
        self.policy_type = policy_type
        self.preferred_providers = preferred_providers
        self.max_cost_per_request = max_cost_per_request
        self.quality_threshold = quality_threshold
        self.latency_threshold = latency_threshold
        self.fallback_strategy = fallback_strategy


class PolicyManager:
    """LLM Provider Policy Manager"""
    
    def __init__(self):
        self.policies = self._initialize_policies()
        self.cache_prefix = "llm_policy"
        
    def _initialize_policies(self) -> Dict[PolicyType, ProviderPolicy]:
        """Initialize policy configurations"""
        return {
            PolicyType.CHEAP_FAST: ProviderPolicy(
                policy_type=PolicyType.CHEAP_FAST,
                preferred_providers=["claude_haiku", "gpt35"],
                max_cost_per_request=0.01,  # $0.01 per request
                quality_threshold=0.7,
                latency_threshold=2.0,  # 2 seconds
                fallback_strategy="template"
            ),
            
            PolicyType.HIGH_QUALITY: ProviderPolicy(
                policy_type=PolicyType.HIGH_QUALITY,
                preferred_providers=["claude_sonnet", "gpt4", "claude_opus"],
                max_cost_per_request=0.10,  # $0.10 per request
                quality_threshold=0.9,
                latency_threshold=10.0,  # 10 seconds
                fallback_strategy="rule_based"
            ),
            
            PolicyType.OFFLINE_FALLBACK: ProviderPolicy(
                policy_type=PolicyType.OFFLINE_FALLBACK,
                preferred_providers=["local_model", "template"],
                max_cost_per_request=0.00,  # Free
                quality_threshold=0.6,
                latency_threshold=1.0,  # 1 second
                fallback_strategy="template"
            ),
            
            PolicyType.BALANCED: ProviderPolicy(
                policy_type=PolicyType.BALANCED,
                preferred_providers=["claude_sonnet", "gpt35", "claude_haiku"],
                max_cost_per_request=0.05,  # $0.05 per request
                quality_threshold=0.8,
                latency_threshold=5.0,  # 5 seconds
                fallback_strategy="hybrid"
            )
        }
    
    async def select_provider_for_policy(
        self,
        policy_type: PolicyType,
        available_providers: Dict[str, BaseLLMProvider],
        task_type: str,
        estimated_tokens: int = 1000
    ) -> Optional[BaseLLMProvider]:
        """Select provider based on policy"""
        
        policy = self.policies.get(policy_type, self.policies[PolicyType.BALANCED])
        
        # Check cost constraints
        estimated_cost = self._estimate_cost(estimated_tokens, policy_type)
        if estimated_cost > policy.max_cost_per_request:
            logger.warning(f"Estimated cost {estimated_cost} exceeds policy limit {policy.max_cost_per_request}")
            return None
        
        # Try preferred providers in order
        for provider_name in policy.preferred_providers:
            if provider_name in available_providers:
                provider = available_providers[provider_name]
                
                # Check if provider is healthy
                if await self._is_provider_healthy(provider):
                    return provider
        
        return None
    
    def _estimate_cost(self, tokens: int, policy_type: PolicyType) -> float:
        """Estimate cost based on tokens and policy"""
        # Rough cost estimates per 1K tokens
        cost_per_1k = {
            PolicyType.CHEAP_FAST: 0.001,      # $0.001 per 1K tokens
            PolicyType.HIGH_QUALITY: 0.03,     # $0.03 per 1K tokens
            PolicyType.OFFLINE_FALLBACK: 0.0,  # Free
            PolicyType.BALANCED: 0.015         # $0.015 per 1K tokens
        }
        
        return (tokens / 1000) * cost_per_1k.get(policy_type, 0.015)
    
    async def _is_provider_healthy(self, provider: BaseLLMProvider) -> bool:
        """Check if provider is healthy and responsive"""
        try:
            # Simple health check - could be enhanced with actual API call
            cache_key = f"{self.cache_prefix}:health:{provider.name}"
            health_status = await cache_service.get(cache_key)
            
            if health_status is None:
                # Assume healthy if no recent failure
                return True
            
            # Check if last failure was more than 5 minutes ago
            last_failure = datetime.fromisoformat(health_status)
            if datetime.now() - last_failure > timedelta(minutes=5):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Health check error for {provider.name}: {e}")
            return False
    
    async def mark_provider_failure(self, provider_name: str):
        """Mark provider as failed"""
        try:
            cache_key = f"{self.cache_prefix}:health:{provider_name}"
            await cache_service.set(cache_key, datetime.now().isoformat(), expire=300)  # 5 minutes
            logger.warning(f"Marked provider {provider_name} as failed")
        except Exception as e:
            logger.error(f"Error marking provider failure: {e}")
    
    async def get_policy_stats(self, policy_type: PolicyType) -> Dict[str, Any]:
        """Get statistics for a policy"""
        policy = self.policies.get(policy_type)
        if not policy:
            return {}
        
        return {
            "policy_type": policy.policy_type.value,
            "preferred_providers": policy.preferred_providers,
            "max_cost_per_request": policy.max_cost_per_request,
            "quality_threshold": policy.quality_threshold,
            "latency_threshold": policy.latency_threshold,
            "fallback_strategy": policy.fallback_strategy
        }
    
    async def get_all_policies(self) -> Dict[str, Dict[str, Any]]:
        """Get all policy configurations"""
        stats = {}
        for policy_type in PolicyType:
            stats[policy_type.value] = await self.get_policy_stats(policy_type)
        return stats


# Global instance
policy_manager = PolicyManager()
