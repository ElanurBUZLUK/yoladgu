"""
Vector operations API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import numpy as np

from app.services.vector_index_manager import vector_index_manager
from app.services.vector_service import vector_service
from app.models.vector_item import VectorItem
from app.models.mmr_search_request import MMRSearchRequest

router = APIRouter(prefix="/vector", tags=["vector"])


@router.get("/health")
async def health_check():
    """Check health of vector backends."""
    try:
        health_status = await vector_index_manager.health_check()
        return {
            "status": "healthy" if all(health_status.values()) else "degraded",
            "backends": health_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/stats")
async def get_stats(backend_name: Optional[str] = Query(None, description="Specific backend name")):
    """Get vector service statistics."""
    try:
        if backend_name:
            stats = await vector_index_manager.get_backend_stats(backend_name)
        else:
            stats = await vector_index_manager.get_manager_stats()
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/monitoring")
async def get_monitoring_dashboard():
    """Get comprehensive monitoring dashboard combining vector and search stats."""
    try:
        # Vector service stats
        vector_stats = await vector_index_manager.get_manager_stats()
        
        # Search service stats (if available)
        search_stats = {}
        try:
            from app.services.search_service import search_service
            search_stats = {
                "elasticsearch_status": "available",
                "index_count": "N/A"  # Could be enhanced with actual ES stats
            }
        except ImportError:
            search_stats = {"elasticsearch_status": "not_available"}
        
        # System health overview
        health_status = await vector_index_manager.health_check()
        overall_health = "healthy" if all(health_status.values()) else "degraded"
        
        # Performance metrics
        performance = {
            "avg_search_time_ms": vector_stats.get("avg_search_time_ms", 0),
            "total_searches": vector_stats.get("total_searches", 0),
            "cache_hit_rate": "N/A",  # Could be enhanced with Redis stats
            "backend_usage": vector_stats.get("backend_usage", {})
        }
        
        return {
            "status": "success",
            "timestamp": "2025-09-03T03:28:00Z",  # Could be enhanced with actual timestamp
            "overall_health": overall_health,
            "vector_service": {
                "health": health_status,
                "stats": vector_stats
            },
            "search_service": search_stats,
            "performance": performance,
            "recommendations": {
                "backend_optimization": "Consider FAISS for large k queries" if vector_stats.get("total_searches", 0) > 100 else "System performing well",
                "cache_optimization": "Enable Redis caching for frequently accessed vectors" if vector_stats.get("total_searches", 0) > 50 else "Cache not needed yet"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monitoring failed: {str(e)}")


@router.post("/search")
async def search_vectors(
    query: str = Query(..., description="Search query text"),
    k: int = Query(10, description="Number of results"),
    backend_name: Optional[str] = Query(None, description="Specific backend to use"),
    use_hybrid: bool = Query(False, description="Use hybrid search across backends"),
    filters: Optional[Dict[str, Any]] = None
):
    """Search for similar vectors."""
    try:
        results = await vector_service.search(
            query=query,
            limit=k,
            filters=filters,
            backend_name=backend_name,
            use_hybrid=use_hybrid
        )
        
        return {
            "query": query,
            "results": results,
            "total": len(results),
            "backend_used": backend_name or "default"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/add")
async def add_vector(
    item_id: str = Query(..., description="Item ID"),
    text: str = Query(..., description="Text content to encode"),
    metadata: Optional[Dict[str, Any]] = None,
    backend_name: Optional[str] = Query(None, description="Specific backend to use")
):
    """Add a single vector item."""
    try:
        success = await vector_service.add_item(
            item_id=item_id,
            text=text,
            metadata=metadata or {},
            backend_name=backend_name
        )
        
        if success:
            return {"message": "Item added successfully", "item_id": item_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to add item")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add item: {str(e)}")


@router.post("/add-batch")
async def add_vectors_batch(
    items: List[VectorItem],
    backend_name: Optional[str] = Query(None, description="Specific backend to use")
):
    """Add multiple vector items in batch."""
    try:
        # Convert Pydantic models to dict format
        items_dict = [
            {
                "id": item.id,
                "text": item.text,
                "metadata": item.metadata
            }
            for item in items
        ]
        
        success = await vector_service.add_batch(
            items=items_dict,
            backend_name=backend_name
        )
        
        if success:
            return {
                "message": "Batch items added successfully",
                "count": len(items),
                "items": [{"id": item.id, "status": "added"} for item in items]
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add batch items")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add batch items: {str(e)}")


@router.post("/search-mmr")
async def search_with_mmr(
    request: MMRSearchRequest
):
    """Search with MMR reranking for diversity optimization."""
    try:
        results = await vector_service.search(
            query=request.query,
            limit=request.k,
            filters=request.filters,
            backend_name=request.backend_name,
            use_hybrid=request.use_hybrid,
            use_mmr=True,
            mmr_lambda=request.mmr_lambda
        )
        
        return {
            "query": request.query,
            "results": results,
            "total": len(results),
            "backend_used": request.backend_name or "default",
            "mmr_applied": True,
            "mmr_lambda": request.mmr_lambda,
            "diversity_optimized": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MMR search failed: {str(e)}")


@router.get("/backends")
async def list_backends():
    """List available vector backends."""
    try:
        stats = await vector_index_manager.get_manager_stats()
        return {
            "available_backends": stats.get("available_backends", []),
            "default_backend": stats.get("default_backend", "qdrant"),
            "backend_details": stats.get("backend_details", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list backends: {str(e)}")


@router.post("/switch-backend")
async def switch_default_backend(backend_name: str = Query(..., description="New default backend name")):
    """Switch the default vector backend."""
    try:
        success = await vector_index_manager.switch_default_backend(backend_name)
        
        if success:
            return {
                "message": f"Default backend switched to {backend_name}",
                "new_default": backend_name
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Backend '{backend_name}' not available"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch backend: {str(e)}")


@router.get("/benchmark")
async def benchmark_search(
    query: str = Query(..., description="Search query for benchmarking"),
    k: int = Query(10, description="Number of results"),
    runs: int = Query(5, description="Number of benchmark runs")
):
    """Benchmark search performance across backends."""
    try:
        import time
        
        benchmark_results = {}
        
        # Test each available backend
        for backend_name in vector_index_manager.backends.keys():
            times = []
            for _ in range(runs):
                start_time = time.time()
                await vector_service.search(
                    query=query,
                    limit=k,
                    backend_name=backend_name
                )
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            benchmark_results[backend_name] = {
                "avg_time_ms": sum(times) / len(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "times_ms": times
            }
        
        # Test hybrid search if multiple backends available
        if len(vector_index_manager.backends) > 1:
            times = []
            for _ in range(runs):
                start_time = time.time()
                await vector_service.search(
                    query=query,
                    limit=k,
                    use_hybrid=True
                )
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            benchmark_results["hybrid"] = {
                "avg_time_ms": sum(times) / len(times),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "times_ms": times
            }
        
        return {
            "query": query,
            "k": k,
            "runs": runs,
            "results": benchmark_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")
