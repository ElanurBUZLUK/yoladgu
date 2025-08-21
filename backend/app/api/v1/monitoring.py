from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response, JSONResponse
from typing import Dict, Any
import structlog
from ...core.config import settings
from ...services.monitoring_service import monitoring_service, get_grafana_dashboard_config
from ...services.storage_service import storage_service
from ...core.security import get_current_user
from ...models.user import User
from ...main import cache_service # Import the global cache_service instance

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


@router.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    if not settings.prometheus_enabled:
        raise HTTPException(status_code=404, detail="Metrics endpoint disabled")
    
    try:
        metrics = monitoring_service.get_metrics()
        return Response(content=metrics, media_type="text/plain")
    except Exception as e:
        logger.error("Error generating metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Error generating metrics")


@router.get("/health")
async def health_check():
    """Get comprehensive health status"""
    try:
        # Get system health
        system_health = monitoring_service.get_health_status()
        
        # Get storage health
        storage_health = await storage_service.health_check()
        
        # Combine health statuses
        overall_status = "healthy"
        if (system_health["status"] != "healthy" or 
            storage_health["status"] != "healthy"):
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": system_health["timestamp"],
            "system": system_health,
            "storage": storage_health,
            "version": settings.version,
            "environment": settings.environment.value
        }
    except Exception as e:
        logger.error("Error in health check", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": system_health.get("timestamp", 0)
        }


@router.get("/health/simple")
async def simple_health_check():
    """Simple health check for load balancers"""
    return {"status": "ok", "timestamp": monitoring_service.get_health_status()["timestamp"]}


@router.get("/dashboard")
async def get_dashboard_config():
    """Get Grafana dashboard configuration"""
    try:
        dashboard_config = get_grafana_dashboard_config()
        return JSONResponse(content=dashboard_config)
    except Exception as e:
        logger.error("Error generating dashboard config", error=str(e))
        raise HTTPException(status_code=500, detail="Error generating dashboard config")


@router.get("/stats")
async def get_system_stats(current_user: User = Depends(get_current_user)):
    """Get system statistics (requires authentication)"""
    try:
        # Update system metrics
        monitoring_service.update_system_metrics()
        
        # Get current metrics
        health_status = monitoring_service.get_health_status()
        
        return {
            "system": health_status["system"],
            "disk": health_status["disk"],
            "metrics_enabled": settings.prometheus_enabled,
            "environment": settings.environment.value
        }
    except Exception as e:
        logger.error("Error getting system stats", error=str(e))
        raise HTTPException(status_code=500, detail="Error getting system stats")


@router.get("/cache/health")
async def get_cache_health():
    """Get cache service health status"""
    try:
        pong = await cache_service.ping()
        
        return {
            "status": "healthy" if pong else "unhealthy",
            "ping": bool(pong),
            "operations": ["PING"]
        }
    except Exception as e:
        logger.error("Error getting cache health", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/cache/metrics")
async def get_cache_metrics(current_user: User = Depends(get_current_user)):
    """Get cache service metrics (requires authentication)"""
    try:
        size = await cache_service.client.dbsize()
        
        return {
            "size": size,
            "connected": True
        }
    except Exception as e:
        logger.error("Error getting cache metrics", error=str(e))
        return {
            "size": 0,
            "connected": False,
            "error": str(e)
        }


@router.post("/cache/clear")
async def clear_cache(current_user: User = Depends(get_current_user)):
    """Clear all cached data (requires authentication)"""
    try:
        await cache_service.client.flushdb()
        
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error("Error clearing cache", error=str(e))
        raise HTTPException(status_code=500, detail="Error clearing cache")


@router.get("/storage/status")
async def get_storage_status(current_user: User = Depends(get_current_user)):
    """Get storage service status"""
    try:
        storage_health = await storage_service.health_check()
        return storage_health
    except Exception as e:
        logger.error("Error getting storage status", error=str(e))
        raise HTTPException(status_code=500, detail="Error getting storage status")


@router.get("/storage/files")
async def list_storage_files(
    prefix: str = "",
    current_user: User = Depends(get_current_user)
):
    """List files in storage (requires authentication)"""
    try:
        files = await storage_service.list_files(prefix)
        return {
            "files": files,
            "count": len(files),
            "prefix": prefix
        }
    except Exception as e:
        logger.error("Error listing storage files", error=str(e))
        raise HTTPException(status_code=500, detail="Error listing storage files")


@router.get("/config")
async def get_config_summary(current_user: User = Depends(get_current_user)):
    """Get configuration summary (without sensitive data)"""
    try:
        return {
            "environment": settings.environment.value,
            "debug": settings.debug,
            "api_docs_enabled": settings.api_docs_enabled,
            "prometheus_enabled": settings.prometheus_enabled,
            "rate_limit_enabled": settings.rate_limit_enabled,
            "storage_backend": settings.storage_backend.value,
            "llm_providers_available": len(settings.llm_providers_available),
            "content_moderation_enabled": settings.content_moderation_enabled,
            "cost_monitoring_enabled": settings.cost_monitoring_enabled,
            "version": settings.version
        }
    except Exception as e:
        logger.error("Error getting config summary", error=str(e))
        raise HTTPException(status_code=500, detail="Error getting config summary")
