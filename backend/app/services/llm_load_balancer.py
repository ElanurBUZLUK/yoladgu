"""
LLM Load Balancer with intelligent provider selection and health management
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import random
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    COST_OPTIMIZED = "cost_optimized"
    ADAPTIVE = "adaptive"


@dataclass
class ProviderConfig:
    """Provider configuration"""
    name: str
    weight: float = 1.0
    max_concurrent: int = 10
    timeout: float = 30.0
    cost_per_1k_tokens: float = 0.002
    priority: int = 1  # Lower number = higher priority
    enabled: bool = True
    model_capabilities: List[str] = field(default_factory=list)


@dataclass
class ProviderMetrics:
    """Provider performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    current_connections: int = 0
    total_response_time: float = 0.0
    total_cost: float = 0.0
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    recent_response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count_24h: int = 0
    last_error_reset: float = field(default_factory=time.time)


class LLMLoadBalancer:
    """Intelligent LLM provider load balancer"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.providers: Dict[str, ProviderConfig] = {}
        self.metrics: Dict[str, ProviderMetrics] = defaultdict(ProviderMetrics)
        self.round_robin_index = 0
        
        # Health check settings
        self.health_check_interval = 60  # seconds
        self.max_failures_before_disable = 5
        self.recovery_check_interval = 300  # 5 minutes
        
        # Initialize default providers
        self._initialize_default_providers()
        
        # Start background tasks - will be called during initialization
        # asyncio.create_task(self._health_check_loop())
    
    async def initialize(self):
        """Initialize the load balancer with background tasks"""
        asyncio.create_task(self._health_check_loop())
    
    def _initialize_default_providers(self):
        """Initialize default provider configurations"""
        
        default_providers = [
            ProviderConfig(
                name="openai_gpt4",
                weight=1.0,
                max_concurrent=5,
                timeout=45.0,
                cost_per_1k_tokens=0.03,
                priority=1,
                model_capabilities=["text-generation", "json-mode", "function-calling"]
            ),
            ProviderConfig(
                name="openai_gpt35",
                weight=1.5,
                max_concurrent=10,
                timeout=30.0,
                cost_per_1k_tokens=0.0015,
                priority=2,
                model_capabilities=["text-generation", "json-mode"]
            ),
            ProviderConfig(
                name="anthropic_claude_opus",
                weight=0.8,
                max_concurrent=3,
                timeout=60.0,
                cost_per_1k_tokens=0.015,
                priority=1,
                model_capabilities=["text-generation", "long-context"]
            ),
            ProviderConfig(
                name="anthropic_claude_sonnet",
                weight=1.2,
                max_concurrent=8,
                timeout=40.0,
                cost_per_1k_tokens=0.003,
                priority=2,
                model_capabilities=["text-generation", "code-generation"]
            ),
            ProviderConfig(
                name="anthropic_claude_haiku",
                weight=2.0,
                max_concurrent=15,
                timeout=20.0,
                cost_per_1k_tokens=0.00025,
                priority=3,
                model_capabilities=["text-generation", "fast-response"]
            ),
            ProviderConfig(
                name="local_model",
                weight=0.5,
                max_concurrent=5,
                timeout=15.0,
                cost_per_1k_tokens=0.0,
                priority=4,
                enabled=False,  # Disabled by default
                model_capabilities=["text-generation", "offline"]
            )
        ]
        
        for provider in default_providers:
            self.providers[provider.name] = provider
    
    async def select_provider(self, 
                            requirements: Optional[List[str]] = None,
                            preferred_provider: Optional[str] = None,
                            budget_limit: Optional[float] = None) -> Optional[str]:
        """
        Select the best provider based on strategy and requirements
        """
        
        # Filter available providers
        available_providers = self._get_available_providers(
            requirements, budget_limit
        )
        
        if not available_providers:
            logger.warning("No available providers found")
            return None
        
        # If preferred provider is available, use it
        if preferred_provider and preferred_provider in available_providers:
            return preferred_provider
        
        # Select based on strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_providers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(available_providers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_providers)
        elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
            return self._response_time_selection(available_providers)
        elif self.strategy == LoadBalancingStrategy.SUCCESS_RATE:
            return self._success_rate_selection(available_providers)
        elif self.strategy == LoadBalancingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_selection(available_providers)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_selection(available_providers)
        else:
            return random.choice(available_providers)
    
    def _get_available_providers(self, 
                               requirements: Optional[List[str]] = None,
                               budget_limit: Optional[float] = None) -> List[str]:
        """Get list of available providers based on requirements"""
        
        available = []
        
        for name, config in self.providers.items():
            if not config.enabled:
                continue
            
            # Check current connections
            metrics = self.metrics[name]
            if metrics.current_connections >= config.max_concurrent:
                continue
            
            # Check requirements
            if requirements:
                if not all(req in config.model_capabilities for req in requirements):
                    continue
            
            # Check budget
            if budget_limit and config.cost_per_1k_tokens > budget_limit:
                continue
            
            # Check recent failures
            if self._is_provider_healthy(name):
                available.append(name)
        
        return available
    
    def _is_provider_healthy(self, provider_name: str) -> bool:
        """Check if provider is healthy"""
        
        metrics = self.metrics[provider_name]
        
        # Reset 24h error count if needed
        if time.time() - metrics.last_error_reset > 86400:  # 24 hours
            metrics.error_count_24h = 0
            metrics.last_error_reset = time.time()
        
        # Check error rate
        if metrics.error_count_24h >= self.max_failures_before_disable:
            return False
        
        # Check if recently failed and in recovery period
        if metrics.last_failure_time:
            time_since_failure = time.time() - metrics.last_failure_time
            if time_since_failure < self.recovery_check_interval:
                return False
        
        return True
    
    def _round_robin_selection(self, providers: List[str]) -> str:
        """Round robin provider selection"""
        
        self.round_robin_index = (self.round_robin_index + 1) % len(providers)
        return providers[self.round_robin_index]
    
    def _weighted_round_robin_selection(self, providers: List[str]) -> str:
        """Weighted round robin selection based on provider weights"""
        
        weights = [self.providers[p].weight for p in providers]
        return random.choices(providers, weights=weights)[0]
    
    def _least_connections_selection(self, providers: List[str]) -> str:
        """Select provider with least active connections"""
        
        return min(providers, key=lambda p: self.metrics[p].current_connections)
    
    def _response_time_selection(self, providers: List[str]) -> str:
        """Select provider with best average response time"""
        
        def avg_response_time(provider: str) -> float:
            metrics = self.metrics[provider]
            if not metrics.recent_response_times:
                return float('inf')
            return sum(metrics.recent_response_times) / len(metrics.recent_response_times)
        
        return min(providers, key=avg_response_time)
    
    def _success_rate_selection(self, providers: List[str]) -> str:
        """Select provider with highest success rate"""
        
        def success_rate(provider: str) -> float:
            metrics = self.metrics[provider]
            total = metrics.total_requests
            if total == 0:
                return 0.5  # Default for providers with no history
            return metrics.successful_requests / total
        
        return max(providers, key=success_rate)
    
    def _cost_optimized_selection(self, providers: List[str]) -> str:
        """Select most cost-effective provider"""
        
        def cost_score(provider: str) -> float:
            config = self.providers[provider]
            metrics = self.metrics[provider]
            
            # Factor in both cost and success rate
            success_rate = (metrics.successful_requests / max(metrics.total_requests, 1))
            cost_effectiveness = success_rate / max(config.cost_per_1k_tokens, 0.0001)
            
            return cost_effectiveness
        
        return max(providers, key=cost_score)
    
    def _adaptive_selection(self, providers: List[str]) -> str:
        """Adaptive selection combining multiple factors"""
        
        def adaptive_score(provider: str) -> float:
            config = self.providers[provider]
            metrics = self.metrics[provider]
            
            # Success rate (0-1)
            success_rate = (
                metrics.successful_requests / max(metrics.total_requests, 1)
                if metrics.total_requests > 0 else 0.5
            )
            
            # Response time score (0-1, inverted)
            avg_response_time = (
                sum(metrics.recent_response_times) / len(metrics.recent_response_times)
                if metrics.recent_response_times else 30.0
            )
            response_time_score = max(0, 1 - (avg_response_time / 60.0))  # Normalize to 60s
            
            # Cost score (0-1, inverted)
            cost_score = max(0, 1 - (config.cost_per_1k_tokens / 0.1))  # Normalize to $0.1
            
            # Load score (0-1, inverted)
            load_score = 1 - (metrics.current_connections / config.max_concurrent)
            
            # Priority score (0-1, inverted)
            priority_score = 1 - (config.priority / 5.0)  # Normalize to priority 5
            
            # Weighted combination
            adaptive_score = (
                success_rate * 0.3 +
                response_time_score * 0.25 +
                cost_score * 0.2 +
                load_score * 0.15 +
                priority_score * 0.1
            )
            
            return adaptive_score
        
        return max(providers, key=adaptive_score)
    
    async def record_request_start(self, provider: str):
        """Record the start of a request"""
        
        metrics = self.metrics[provider]
        metrics.current_connections += 1
        metrics.total_requests += 1
    
    async def record_request_end(self, 
                               provider: str, 
                               success: bool, 
                               response_time: float,
                               cost: float = 0.0,
                               error_message: Optional[str] = None):
        """Record the end of a request"""
        
        metrics = self.metrics[provider]
        metrics.current_connections = max(0, metrics.current_connections - 1)
        metrics.total_response_time += response_time
        metrics.recent_response_times.append(response_time)
        metrics.total_cost += cost
        
        if success:
            metrics.successful_requests += 1
            metrics.last_success_time = time.time()
        else:
            metrics.failed_requests += 1
            metrics.last_failure_time = time.time()
            metrics.error_count_24h += 1
            
            logger.warning(f"Provider {provider} request failed: {error_message}")
    
    async def _health_check_loop(self):
        """Background health check for providers"""
        
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all providers"""
        
        for provider_name, config in self.providers.items():
            if not config.enabled:
                continue
            
            try:
                # Simple health check - just try to get provider status
                # In a real implementation, you might ping the actual API
                metrics = self.metrics[provider_name]
                
                # Re-enable provider if it has been healthy for recovery period
                if (metrics.last_failure_time and 
                    time.time() - metrics.last_failure_time > self.recovery_check_interval):
                    
                    logger.info(f"Provider {provider_name} recovery period completed")
                    
            except Exception as e:
                logger.error(f"Health check failed for provider {provider_name}: {e}")
    
    async def get_load_balancer_stats(self) -> Dict[str, any]:
        """Get comprehensive load balancer statistics"""
        
        stats = {
            "strategy": self.strategy.value,
            "total_providers": len(self.providers),
            "enabled_providers": sum(1 for p in self.providers.values() if p.enabled),
            "providers": {}
        }
        
        for name, config in self.providers.items():
            metrics = self.metrics[name]
            
            success_rate = (
                metrics.successful_requests / max(metrics.total_requests, 1)
                if metrics.total_requests > 0 else 0.0
            )
            
            avg_response_time = (
                sum(metrics.recent_response_times) / len(metrics.recent_response_times)
                if metrics.recent_response_times else 0.0
            )
            
            stats["providers"][name] = {
                "config": {
                    "weight": config.weight,
                    "max_concurrent": config.max_concurrent,
                    "timeout": config.timeout,
                    "cost_per_1k_tokens": config.cost_per_1k_tokens,
                    "priority": config.priority,
                    "enabled": config.enabled,
                    "capabilities": config.model_capabilities
                },
                "metrics": {
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "success_rate": round(success_rate, 4),
                    "current_connections": metrics.current_connections,
                    "average_response_time": round(avg_response_time, 3),
                    "total_cost": round(metrics.total_cost, 4),
                    "error_count_24h": metrics.error_count_24h,
                    "is_healthy": self._is_provider_healthy(name)
                }
            }
        
        return stats
    
    def update_provider_config(self, 
                             provider_name: str, 
                             config_updates: Dict[str, any]):
        """Update provider configuration"""
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not found")
        
        config = self.providers[provider_name]
        
        for key, value in config_updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")
        
        logger.info(f"Updated configuration for provider {provider_name}")


# Global instance
llm_load_balancer = LLMLoadBalancer()
