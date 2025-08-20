from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
import logging

from app.core.mcp_observability import mcp_observability
from app.middleware.auth import get_current_student
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/mcp", tags=["mcp-monitoring"])


@router.get("/health", status_code=status.HTTP_200_OK)
async def get_mcp_health():
    """Get MCP health status"""
    try:
        health_status = mcp_observability.get_health_status()
        return {
            "status": "success",
            "data": health_status
        }
    except Exception as e:
        logger.error(f"Failed to get MCP health status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCP health status"
        )


@router.get("/metrics", status_code=status.HTTP_200_OK)
async def get_mcp_metrics(time_window_minutes: int = 60):
    """Get MCP metrics"""
    try:
        metrics = {
            "latency_stats": mcp_observability.get_latency_stats(time_window_minutes=time_window_minutes),
            "usage_stats": mcp_observability.get_usage_stats(time_window_minutes=time_window_minutes),
            "error_stats": mcp_observability.get_error_stats(time_window_minutes=time_window_minutes)
        }
        
        return {
            "status": "success",
            "data": metrics,
            "time_window_minutes": time_window_minutes
        }
    except Exception as e:
        logger.error(f"Failed to get MCP metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCP metrics"
        )


@router.get("/alerts", status_code=status.HTTP_200_OK)
async def get_mcp_alerts():
    """Get MCP performance alerts"""
    try:
        alerts = mcp_observability.get_performance_alerts()
        
        return {
            "status": "success",
            "data": alerts,
            "count": len(alerts)
        }
    except Exception as e:
        logger.error(f"Failed to get MCP alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCP alerts"
        )


@router.get("/tool/{tool_name}/stats", status_code=status.HTTP_200_OK)
async def get_tool_stats(tool_name: str, time_window_minutes: int = 60):
    """Get specific tool statistics"""
    try:
        tool_stats = mcp_observability.get_latency_stats(
            tool_name=tool_name,
            time_window_minutes=time_window_minutes
        )
        
        return {
            "status": "success",
            "data": {
                "tool_name": tool_name,
                "stats": tool_stats,
                "time_window_minutes": time_window_minutes
            }
        }
    except Exception as e:
        logger.error(f"Failed to get tool stats for {tool_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get tool stats for {tool_name}"
        )


@router.get("/export", status_code=status.HTTP_200_OK)
async def export_mcp_metrics():
    """Export all MCP metrics for external monitoring"""
    try:
        exported_metrics = mcp_observability.export_metrics()
        
        return {
            "status": "success",
            "data": exported_metrics
        }
    except Exception as e:
        logger.error(f"Failed to export MCP metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export MCP metrics"
        )
