"""
Services Package - Organized service layer
All services are categorized and exposed through this package
"""

# Core Services
from .async_neo4j_service import async_neo4j_service
from .async_redis_service import async_redis_service
from .llm_service import LLMService

# Orchestrator Services
from .orchestrators.recommendation_orchestrator import recommendation_orchestrator

# Utility Services
from .cache_service import CacheService

# Domain Services
from .domain.question_service import QuestionDomainService

# Service instances
__all__ = [
    # Core Services
    "async_neo4j_service",
    "async_redis_service", 
    "LLMService",
    
    # Orchestrators
    "recommendation_orchestrator",
    
    # Utility Services
    "CacheService",
    
    # Domain Services
    "QuestionDomainService",
]

# Service categories for easy access
CORE_SERVICES = {
    "neo4j": async_neo4j_service,
    "redis": async_redis_service,
    "llm": LLMService(),
}

ORCHESTRATORS = {
    "recommendation": recommendation_orchestrator,
}

UTILITY_SERVICES = {
    "cache": CacheService(),
}

DOMAIN_SERVICES = {
    "question": QuestionDomainService(),
}

# Service registry for dependency injection
SERVICE_REGISTRY = {
    **CORE_SERVICES,
    **ORCHESTRATORS,
    **UTILITY_SERVICES,
    **DOMAIN_SERVICES,
}
