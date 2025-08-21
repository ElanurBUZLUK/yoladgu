from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import json # Added for hashing request body

from app.database_enhanced import enhanced_database_manager
from app.middleware.auth import get_current_teacher, get_current_student
from app.models.user import User
from app.services.vector_index_manager import vector_index_manager
from app.utils.distlock_idem import RedisLock, IdempotencyConfig, idempotency_decorator # Added idempotency_decorator
from redis.asyncio import Redis
from app.schemas.vector_management import (
    GetVectorStatisticsResponse, ForceReleaseLockResponse,
    RebuildIndexManualResponse, VectorHealthCheckResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/vector", tags=["vector-management"])


# Request/Response Models
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


# Vector Management Endpoints
@router.post("/batch-upsert", response_model=BatchUpsertResponse, status_code=status.HTTP_200_OK)
@idempotency_decorator(
    key_builder=lambda request, user: f"batch_upsert:{request.table_name}:{request.namespace}:{hash(json.dumps(request.dict(), sort_keys=True))}",
    config=IdempotencyConfig(scope="batch_upsert", ttl_seconds=3600)
)
async def batch_upsert_embeddings(
    request: BatchUpsertRequest,
    db: AsyncSession = Depends(enhanced_database_manager.get_session),
    user: User = Depends(get_current_teacher),
):
    """Batch upsert embeddings with distributed lock protection"""
    try:
        result = await vector_index_manager.batch_upsert_embeddings(
            items=request.items,
            table_name=request.table_name,
            namespace=request.namespace
        )
        
        return BatchUpsertResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in batch upsert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch upsert failed: {str(e)}"
        )


@router.post("/rebuild-index", response_model=RebuildIndexResponse, status_code=status.HTTP_200_OK)
@idempotency_decorator(
    key_builder=lambda request, user: f"rebuild_index:{request.namespace}",
    config=IdempotencyConfig(scope="rebuild_index", ttl_seconds=3600)
)
async def rebuild_vector_index(
    request: RebuildIndexRequest,
    db: AsyncSession = Depends(enhanced_database_manager.get_session),
    user: User = Depends(get_current_teacher),
):
    """Rebuild vector index for a namespace (idempotent operation)"""
    try:
        result = await vector_index_manager.rebuild_vector_index(request.namespace)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Rebuild failed")
            )
        
        return RebuildIndexResponse(**result)
        
    except Exception as e:
        logger.error(f"Error rebuilding vector index: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rebuild failed: {str(e)}"
        )


@router.post("/cleanup-slots", response_model=CleanupResponse, status_code=status.HTTP_200_OK)
@idempotency_decorator(
    key_builder=lambda request, user: f"cleanup_slots:{request.namespace}:{request.keep_slots}",
    config=IdempotencyConfig(scope="cleanup_slots", ttl_seconds=3600)
)
async def cleanup_old_slots(
    request: CleanupRequest,
    db: AsyncSession = Depends(enhanced_database_manager.get_session),
    user: User = Depends(get_current_teacher),
):
    """Clean up old slots, keeping only the most recent ones"""
    try:
        result = await vector_index_manager.cleanup_old_slots(
            namespace=request.namespace,
            keep_slots=request.keep_slots
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Cleanup failed")
            )
        
        return CleanupResponse(**result)
        
    except Exception as e:
        logger.error(f"Error cleaning up slots: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cleanup failed: {str(e)}"
        )


@router.get("/statistics", response_model=GetVectorStatisticsResponse, status_code=status.HTTP_200_OK)
async def get_vector_statistics(
    db: AsyncSession = Depends(enhanced_database_manager.get_session),
    user: User = Depends(get_current_teacher),
):
    """Get vector index statistics"""
    try:
        stats = await vector_index_manager.get_index_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting vector statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/lock-status/{lock_key}", response_model=LockStatusResponse, status_code=status.HTTP_200_OK)
async def check_lock_status(
    lock_key: str,
    db: AsyncSession = Depends(enhanced_database_manager.get_session),
    user: User = Depends(get_current_teacher),
):
    """Check if a specific lock is currently held"""
    try:
        redis_client = vector_index_manager._get_redis_client()
        
        # Check if lock exists
        lock_value = await redis_client.get(lock_key)
        locked = lock_value is not None
        
        # Get TTL if locked
        ttl_ms = 0
        if locked:
            ttl_ms = await redis_client.pttl(lock_key)
        
        return LockStatusResponse(
            locked=locked,
            lock_key=lock_key,
            ttl_ms=max(0, ttl_ms)
        )
        
    except Exception as e:
        logger.error(f"Error checking lock status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check lock status: {str(e)}"
        )


@router.delete("/lock/{lock_key}", response_model=ForceReleaseLockResponse, status_code=status.HTTP_200_OK)
async def force_release_lock(
    lock_key: str,
    db: AsyncSession = Depends(enhanced_database_manager.get_session),
    user: User = Depends(get_current_teacher),
):
    """Force release a lock (admin only)"""
    try:
        redis_client = vector_index_manager._get_redis_client()
        
        # Delete the lock
        deleted = await redis_client.delete(lock_key)
        
        return {
            "success": deleted > 0,
            "lock_key": lock_key,
            "message": "Lock released" if deleted > 0 else "Lock not found"
        }
        
    except Exception as e:
        logger.error(f"Error force releasing lock: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to release lock: {str(e)}"
        )


# Context Manager Example Endpoint
@router.post("/admin/rebuild-index-manual", response_model=RebuildIndexManualResponse, status_code=status.HTTP_200_OK)
@idempotency_decorator(
    key_builder=lambda namespace, user: f"manual_rebuild:{namespace}",
    config=IdempotencyConfig(scope="manual_rebuild", ttl_seconds=3600)
)
async def rebuild_index_manual(
    namespace: str,
    db: AsyncSession = Depends(enhanced_database_manager.get_session),
    user: User = Depends(get_current_teacher),
):
    """Manual rebuild using context manager (example)"""
    try:
        redis_client = vector_index_manager._get_redis_client()
        lock_key = f"lock:vector:rebuild:{namespace}"
        
        try:
            async with RedisLock(redis_client, lock_key, ttl_ms=300_000):  # 5 minutes
                logger.info(f"🔄 Manual rebuild started for namespace: {namespace}")
                
                # Perform the rebuild
                result = await vector_index_manager.rebuild_vector_index(namespace)
                
                logger.info(f"✅ Manual rebuild completed for namespace: {namespace}")
                return result
                
        except TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Another rebuild is in progress for namespace: {namespace}"
            )
            
    except Exception as e:
        logger.error(f"Error in manual rebuild: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Manual rebuild failed: {str(e)}"
        )


# Health Check Endpoint
@router.get("/health", response_model=VectorHealthCheckResponse, status_code=status.HTTP_200_OK)
async def vector_health_check(
    db: AsyncSession = Depends(enhanced_database_manager.get_session),
    user: User = Depends(get_current_teacher),
):
    """Vector system health check"""
    try:
        # Check vector indexes
        indexes_ok = await vector_index_manager.create_vector_indexes()
        
        # Get statistics
        stats = await vector_index_manager.get_index_statistics()
        
        # Check Redis connection
        redis_client = vector_index_manager._get_redis_client()
        redis_ok = await redis_client.ping()
        
        return {
            "status": "healthy" if indexes_ok and redis_ok else "unhealthy",
            "vector_indexes": indexes_ok,
            "redis_connection": redis_ok,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in vector health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }