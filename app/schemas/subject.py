"""
Subject Schemas - Import from common to avoid duplication
"""

from .common import (
    SubjectBase,
    SubjectCreate,
    SubjectUpdate,
    SubjectResponse,
    PaginationRequest,
    FilterRequest,
    ListResponse
)

# Re-export for backward compatibility
__all__ = [
    "SubjectBase",
    "SubjectCreate", 
    "SubjectUpdate",
    "SubjectResponse",
    "PaginationRequest",
    "FilterRequest",
    "ListResponse"
]
