"""
Services Package - Organized service layer
All services are categorized and exposed through this package
"""

# Core Services
from .neo4j_consolidated import neo4j_service
from .cache_service import cache_service
from .llm_service import LLMService

# Orchestrator Services
from .orchestrators.recommendation_orchestrator import recommendation_orchestrator

# Service instances
__all__ = [
    # Core Services
    "neo4j_service",
    "cache_service", 
    "LLMService",
    
    # Orchestrators
    "recommendation_orchestrator",
]

# Service categories for easy access
CORE_SERVICES = {
    "neo4j": neo4j_service,
    "cache": cache_service,
    "llm": LLMService(),
}

ORCHESTRATORS = {
    "recommendation": recommendation_orchestrator,
}

# Service registry for dependency injection
SERVICE_REGISTRY = {
    **CORE_SERVICES,
    **ORCHESTRATORS,
}
