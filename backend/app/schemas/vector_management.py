from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

# Request/Response Models (copied from api/v1/vector_management.py for consistency)
class BatchUpsertRequest(BaseModel):
    items: List[Dict[str, Any]] = Field(..., description="Items to upsert")
    table_name: str = Field(..., description="Table name (questions or error_patterns)")
    namespace: Optional[str] = Field(None, description="Namespace")


class BatchUpsertResponse(BaseModel):
    success: bool
    processed: int
    namespace: str
    slot: int
    errors: List[str]


class RebuildIndexRequest(BaseModel):
    namespace: str = Field(..., description="Namespace to rebuild")
    force: bool = Field(default=False, description="Force rebuild even if in progress")


class RebuildIndexResponse(BaseModel):
    success: bool
    namespace: str
    questions_updated: int
    patterns_updated: int
    total_processed: int


class CleanupRequest(BaseModel):
    namespace: str = Field(..., description="Namespace to clean")
    keep_slots: int = Field(default=3, ge=1, le=10, description="Number of slots to keep")


class CleanupResponse(BaseModel):
    success: bool
    slots_cleaned: int
    cleaned_slots: List[int]


class LockStatusResponse(BaseModel):
    locked: bool
    lock_key: str
    ttl_ms: int


# New Response Models

class GetVectorStatisticsResponse(BaseModel):
    total_indexes: int
    total_vectors: int
    index_details: Dict[str, Any]
    last_updated: datetime

class ForceReleaseLockResponse(BaseModel):
    success: bool
    lock_key: str
    message: str

class RebuildIndexManualResponse(BaseModel):
    success: bool
    namespace: str
    questions_updated: int
    patterns_updated: int
    total_processed: int

class VectorHealthCheckResponse(BaseModel):
    status: str
    vector_indexes: bool
    redis_connection: bool
    statistics: Dict[str, Any]
    timestamp: str
    error: Optional[str] = None
