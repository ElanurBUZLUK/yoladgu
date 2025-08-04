"""
Services Package - Organized service layer
All services are categorized and exposed through this package
"""

# Core Services
from .async_neo4j_service import async_neo4j_service
from .async_redis_service import async_redis_service
from .embedding_service import EmbeddingService
from .llm_service import LLMService

# Orchestrator Services
from .orchestrators.recommendation_orchestrator import recommendation_orchestrator

# Feature Services
from .feature_extraction import feature_extraction_service

# Utility Services
from .cache_service import CacheService
from .metrics_service import MetricsService
from .rate_limiter import RateLimiter
from .task_queue import TaskQueue

# Batch Operations
from .batch_operations import batch_operations_service
from .async_stream_consumer import AsyncStreamConsumer

# Domain Services
from .domain.question_service import QuestionDomainService

# Service instances
__all__ = [
    # Core Services
    "async_neo4j_service",
    "async_redis_service", 
    "EmbeddingService",
    "LLMService",
    
    # Orchestrators
    "recommendation_orchestrator",
    
    # Feature Services
    "feature_extraction_service",
    
    # Utility Services
    "CacheService",
    "MetricsService", 
    "RateLimiter",
    "TaskQueue",
    
    # Batch Operations
    "batch_operations_service",
    "AsyncStreamConsumer",
    
    # Domain Services
    "QuestionDomainService",
]

# Service categories for easy access
CORE_SERVICES = {
    "neo4j": async_neo4j_service,
    "redis": async_redis_service,
    "embedding": EmbeddingService(),
    "llm": LLMService(),
}

ORCHESTRATORS = {
    "recommendation": recommendation_orchestrator,
}

FEATURE_SERVICES = {
    "extraction": feature_extraction_service,
}

UTILITY_SERVICES = {
    "cache": CacheService(),
    "metrics": MetricsService(),
    "rate_limiter": RateLimiter(),
    "task_queue": TaskQueue(),
}

BATCH_SERVICES = {
    "operations": batch_operations_service,
    "stream_consumer": AsyncStreamConsumer(),
}

DOMAIN_SERVICES = {
    "question": QuestionDomainService(),
}

# Service registry for dependency injection
SERVICE_REGISTRY = {
    **CORE_SERVICES,
    **ORCHESTRATORS,
    **FEATURE_SERVICES,
    **UTILITY_SERVICES,
    **BATCH_SERVICES,
    **DOMAIN_SERVICES,
}
