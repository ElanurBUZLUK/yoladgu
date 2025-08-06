"""
Topic Schemas - Import from common to avoid duplication
"""

from .common import (
    TopicBase,
    TopicCreate,
    TopicUpdate,
    TopicResponse,
    PaginationRequest,
    FilterRequest,
    ListResponse
)

# Re-export for backward compatibility
__all__ = [
    "TopicBase",
    "TopicCreate",
    "TopicUpdate", 
    "TopicResponse",
    "PaginationRequest",
    "FilterRequest",
    "ListResponse"
]
