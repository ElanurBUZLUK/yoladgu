"""
Enhanced Database API endpoints for monitoring and management
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from datetime import datetime

from app.database_enhanced import enhanced_database_manager
from app.core.error_handling import ErrorHandler, ErrorCode, ErrorSeverity

router = APIRouter(prefix="/api/v1/database-enhanced", tags=["Database Enhanced"])
error_handler = ErrorHandler()


class DatabaseHealthResponse(BaseModel):
    status: str = Field(..., description="Database health status")
    response_time: float = Field(..., description="Response time in seconds")
    connection_count: int = Field(..., description="Number of connections")
    error_rate: float = Field(..., description="Error rate percentage")
    slow_queries: int = Field(..., description="Number of slow queries")
    last_check: datetime = Field(..., description="Last health check time")
    details: Dict[str, Any] = Field(..., description="Detailed health information")


class QueryMetricsResponse(BaseModel):
    query_type: str = Field(..., description="Type of query")
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: datetime = Field(..., description="Query timestamp")
    success: bool = Field(..., description="Whether query was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class PerformanceMetricsResponse(BaseModel):
    query_optimizer: Dict[str, Any] = Field(..., description="Query optimizer statistics")
    pool_monitor: Dict[str, Any] = Field(..., description="Connection pool health")
    health_history: List[Dict[str, Any]] = Field(..., description="Health check history")
    performance_alerts: List[Dict[str, Any]] = Field(..., description="Performance alerts")
    maintenance: Dict[str, Any] = Field(..., description="Maintenance information")


class ConnectionPoolResponse(BaseModel):
    utilization_rate: float = Field(..., description="Connection pool utilization rate")
    overflow_rate: float = Field(..., description="Overflow connection rate")
    error_rate: float = Field(..., description="Connection error rate")
    metrics: Dict[str, int] = Field(..., description="Connection metrics")
    status: str = Field(..., description="Pool health status")


@router.get("/health", response_model=DatabaseHealthResponse)
async def get_database_health():
    """
    Get comprehensive database health status
    """
    try:
        # Perform health check
        await enhanced_database_manager._perform_health_check()
        
        # Get latest health status
        if enhanced_database_manager.pool_monitor.health_history:
            latest_health = enhanced_database_manager.pool_monitor.health_history[-1]
            
            return DatabaseHealthResponse(
                status=latest_health.status,
                response_time=latest_health.response_time,
                connection_count=latest_health.connection_count,
                error_rate=latest_health.error_rate,
                slow_queries=latest_health.slow_queries,
                last_check=latest_health.last_check,
                details=latest_health.details
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No health data available"
            )
            
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.DATABASE_ERROR,
            message="Failed to get database health",
            severity=ErrorSeverity.MEDIUM,
            context={}
        )


@router.get("/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics():
    """
    Get comprehensive database performance metrics
    """
    try:
        metrics = await enhanced_database_manager.get_performance_metrics()
        
        return PerformanceMetricsResponse(**metrics)
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.DATABASE_ERROR,
            message="Failed to get performance metrics",
            severity=ErrorSeverity.MEDIUM,
            context={}
        )


@router.get("/pool", response_model=ConnectionPoolResponse)
async def get_connection_pool_status():
    """
    Get connection pool status and metrics
    """
    try:
        pool_health = enhanced_database_manager.pool_monitor.get_pool_health()
        
        return ConnectionPoolResponse(**pool_health)
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.DATABASE_ERROR,
            message="Failed to get connection pool status",
            severity=ErrorSeverity.MEDIUM,
            context={}
        )


@router.get("/queries/slow")
async def get_slow_queries(limit: int = 10):
    """
    Get slowest queries with performance metrics
    """
    try:
        slow_queries = enhanced_database_manager.query_optimizer.get_slow_queries(limit)
        
        return {
            "slow_queries": [
                {
                    "query_type": q.query_type,
                    "execution_time": q.execution_time,
                    "timestamp": q.timestamp.isoformat(),
                    "success": q.success,
                    "error_message": q.error_message
                }
                for q in slow_queries
            ],
            "total_slow_queries": len(slow_queries),
            "threshold": enhanced_database_manager.query_optimizer.slow_query_threshold
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.DATABASE_ERROR,
            message="Failed to get slow queries",
            severity=ErrorSeverity.MEDIUM,
            context={"limit": limit}
        )


@router.get("/queries/statistics")
async def get_query_statistics():
    """
    Get comprehensive query statistics
    """
    try:
        stats = enhanced_database_manager.query_optimizer.get_query_statistics()
        
        return {
            "statistics": stats,
            "query_patterns": stats.get("query_patterns", {}),
            "recent_activity": stats.get("recent_activity", {}),
            "execution_time_stats": stats.get("execution_time_stats", {})
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.DATABASE_ERROR,
            message="Failed to get query statistics",
            severity=ErrorSeverity.MEDIUM,
            context={}
        )


@router.post("/maintenance/trigger")
async def trigger_maintenance():
    """
    Manually trigger database maintenance
    """
    try:
        await enhanced_database_manager._perform_maintenance()
        
        return {
            "success": True,
            "message": "Database maintenance completed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.DATABASE_ERROR,
            message="Failed to trigger maintenance",
            severity=ErrorSeverity.MEDIUM,
            context={}
        )


@router.post("/queries/execute")
async def execute_query_with_metrics(query: str, params: Optional[Dict[str, Any]] = None):
    """
    Execute a query with performance metrics
    """
    try:
        result, execution_time = await enhanced_database_manager.execute_with_metrics(query, params)
        
        return {
            "success": True,
            "execution_time": execution_time,
            "result": str(result) if result else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.DATABASE_ERROR,
            message="Failed to execute query",
            severity=ErrorSeverity.MEDIUM,
            context={"query": query[:100] + "..." if len(query) > 100 else query}
        )


@router.get("/alerts")
async def get_performance_alerts(limit: int = 10):
    """
    Get recent performance alerts
    """
    try:
        alerts = enhanced_database_manager.performance_alerts[-limit:]
        
        return {
            "alerts": alerts,
            "total_alerts": len(enhanced_database_manager.performance_alerts),
            "recent_alerts": len(alerts)
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.DATABASE_ERROR,
            message="Failed to get performance alerts",
            severity=ErrorSeverity.LOW,
            context={"limit": limit}
        )


@router.delete("/alerts/clear")
async def clear_performance_alerts():
    """
    Clear all performance alerts
    """
    try:
        alert_count = len(enhanced_database_manager.performance_alerts)
        enhanced_database_manager.performance_alerts.clear()
        
        return {
            "success": True,
            "message": f"Cleared {alert_count} performance alerts",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.DATABASE_ERROR,
            message="Failed to clear performance alerts",
            severity=ErrorSeverity.LOW,
            context={}
        )


@router.get("/connections/events")
async def get_connection_events(limit: int = 50):
    """
    Get recent connection pool events
    """
    try:
        events = enhanced_database_manager.pool_monitor.connection_events[-limit:]
        
        return {
            "events": events,
            "total_events": len(enhanced_database_manager.pool_monitor.connection_events),
            "recent_events": len(events)
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.DATABASE_ERROR,
            message="Failed to get connection events",
            severity=ErrorSeverity.LOW,
            context={"limit": limit}
        )


@router.get("/status")
async def get_database_status():
    """
    Get overall database status summary
    """
    try:
        # Get various status information
        pool_health = enhanced_database_manager.pool_monitor.get_pool_health()
        query_stats = enhanced_database_manager.query_optimizer.get_query_statistics()
        
        # Determine overall status
        overall_status = "healthy"
        issues = []
        
        if pool_health["status"] != "healthy":
            overall_status = "degraded"
            issues.append("Connection pool issues")
        
        if query_stats.get("failed_queries", 0) > 10:
            overall_status = "degraded"
            issues.append("High query failure rate")
        
        if query_stats.get("slow_queries", 0) > 20:
            overall_status = "degraded"
            issues.append("Many slow queries")
        
        return {
            "overall_status": overall_status,
            "is_initialized": enhanced_database_manager.is_initialized,
            "last_maintenance": enhanced_database_manager.last_maintenance.isoformat(),
            "pool_status": pool_health["status"],
            "query_success_rate": query_stats.get("successful_queries", 0) / max(query_stats.get("total_queries", 1), 1),
            "issues": issues,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.DATABASE_ERROR,
            message="Failed to get database status",
            severity=ErrorSeverity.MEDIUM,
            context={}
        )
