"""
Cache module for the application.
Provides access to the cache service.
"""

from app.services.cache_service import SemanticCache

# Create a global cache service instance
cache_service = SemanticCache()

__all__ = ["cache_service", "SemanticCache"]
