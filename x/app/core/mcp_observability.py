import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MCPLatencyMetric:
    """MCP Latency Metric"""
    tool_name: str
    latency_ms: float
    timestamp: datetime
    success: bool
    error_type: Optional[str] = None
    method: str = "mcp"


@dataclass
class MCPUsageMetric:
    """MCP Usage Metric"""
    tool_name: str
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: datetime
    arguments_size: int
    response_size: int
    method: str = "mcp"


class MCPObservability:
    """MCP Observability and Monitoring"""
    
    def __init__(self, max_metrics: int = 1000):
        self.max_metrics = max_metrics
        self.latency_metrics: deque = deque(maxlen=max_metrics)
        self.usage_metrics: deque = deque(maxlen=max_metrics)
        self.error_counts: defaultdict = defaultdict(int)
        self.tool_usage_counts: defaultdict = defaultdict(int)
        self.start_time = datetime.utcnow()
    
    def record_latency(self, tool_name: str, latency_ms: float, success: bool, 
                      error_type: Optional[str] = None, method: str = "mcp"):
        """Record MCP tool latency"""
        metric = MCPLatencyMetric(
            tool_name=tool_name,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
            success=success,
            error_type=error_type,
            method=method
        )
        
        self.latency_metrics.append(metric)
        
        if not success and error_type:
            self.error_counts[error_type] += 1
        
        self.tool_usage_counts[tool_name] += 1
        
        # Log latency for monitoring
        if latency_ms > 5000:  # Log slow requests
            logger.warning(f"Slow MCP request: {tool_name} took {latency_ms:.2f}ms")
        elif not success:
            logger.error(f"MCP request failed: {tool_name} - {error_type}")
        else:
            logger.debug(f"MCP request: {tool_name} took {latency_ms:.2f}ms")
    
    def record_usage(self, tool_name: str, user_id: Optional[str], session_id: Optional[str],
                    arguments_size: int, response_size: int, method: str = "mcp"):
        """Record MCP tool usage"""
        metric = MCPUsageMetric(
            tool_name=tool_name,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            arguments_size=arguments_size,
            response_size=response_size,
            method=method
        )
        
        self.usage_metrics.append(metric)
    
    def get_latency_stats(self, tool_name: Optional[str] = None, 
                         time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get latency statistics"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        # Filter metrics by time window and tool name
        filtered_metrics = [
            m for m in self.latency_metrics
            if m.timestamp >= cutoff_time and (tool_name is None or m.tool_name == tool_name)
        ]
        
        if not filtered_metrics:
            return {
                "count": 0,
                "avg_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
                "p95_latency_ms": 0,
                "success_rate": 0
            }
        
        latencies = [m.latency_ms for m in filtered_metrics]
        success_count = sum(1 for m in filtered_metrics if m.success)
        
        return {
            "count": len(filtered_metrics),
            "avg_latency_ms": statistics.mean(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
            "success_rate": success_count / len(filtered_metrics)
        }
    
    def get_usage_stats(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get usage statistics"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        # Filter metrics by time window
        filtered_metrics = [
            m for m in self.usage_metrics
            if m.timestamp >= cutoff_time
        ]
        
        if not filtered_metrics:
            return {
                "total_requests": 0,
                "unique_users": 0,
                "unique_sessions": 0,
                "avg_arguments_size": 0,
                "avg_response_size": 0,
                "tool_breakdown": {}
            }
        
        # Calculate statistics
        unique_users = len(set(m.user_id for m in filtered_metrics if m.user_id))
        unique_sessions = len(set(m.session_id for m in filtered_metrics if m.session_id))
        avg_arguments_size = statistics.mean(m.arguments_size for m in filtered_metrics)
        avg_response_size = statistics.mean(m.response_size for m in filtered_metrics)
        
        # Tool breakdown
        tool_breakdown = defaultdict(int)
        for metric in filtered_metrics:
            tool_breakdown[metric.tool_name] += 1
        
        return {
            "total_requests": len(filtered_metrics),
            "unique_users": unique_users,
            "unique_sessions": unique_sessions,
            "avg_arguments_size": avg_arguments_size,
            "avg_response_size": avg_response_size,
            "tool_breakdown": dict(tool_breakdown)
        }
    
    def get_error_stats(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get error statistics"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        # Filter error metrics by time window
        error_metrics = [
            m for m in self.latency_metrics
            if m.timestamp >= cutoff_time and not m.success
        ]
        
        error_breakdown = defaultdict(int)
        for metric in error_metrics:
            error_breakdown[metric.error_type or "unknown"] += 1
        
        return {
            "total_errors": len(error_metrics),
            "error_breakdown": dict(error_breakdown),
            "error_rate": len(error_metrics) / max(len([m for m in self.latency_metrics if m.timestamp >= cutoff_time]), 1)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get MCP health status"""
        # Get recent metrics (last 5 minutes)
        recent_stats = self.get_latency_stats(time_window_minutes=5)
        error_stats = self.get_error_stats(time_window_minutes=5)
        
        # Determine health status
        if recent_stats["count"] == 0:
            health_status = "unknown"
        elif error_stats["error_rate"] > 0.1:  # More than 10% error rate
            health_status = "unhealthy"
        elif recent_stats["avg_latency_ms"] > 10000:  # More than 10 seconds average
            health_status = "degraded"
        else:
            health_status = "healthy"
        
        return {
            "status": health_status,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "total_requests": len(self.latency_metrics),
            "recent_stats": recent_stats,
            "error_stats": error_stats,
            "tool_usage": dict(self.tool_usage_counts)
        }
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts"""
        alerts = []
        
        # Check for high error rates
        error_stats = self.get_error_stats(time_window_minutes=5)
        if error_stats["error_rate"] > 0.05:  # More than 5% error rate
            alerts.append({
                "type": "high_error_rate",
                "severity": "warning",
                "message": f"High error rate: {error_stats['error_rate']:.2%}",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Check for slow responses
        latency_stats = self.get_latency_stats(time_window_minutes=5)
        if latency_stats["avg_latency_ms"] > 5000:  # More than 5 seconds average
            alerts.append({
                "type": "high_latency",
                "severity": "warning",
                "message": f"High average latency: {latency_stats['avg_latency_ms']:.2f}ms",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Check for specific tool issues
        for tool_name in self.tool_usage_counts:
            tool_stats = self.get_latency_stats(tool_name=tool_name, time_window_minutes=5)
            if tool_stats["count"] > 0 and tool_stats["success_rate"] < 0.8:
                alerts.append({
                    "type": "tool_failure",
                    "severity": "error",
                    "message": f"Tool {tool_name} has low success rate: {tool_stats['success_rate']:.2%}",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return alerts
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external monitoring"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "health_status": self.get_health_status(),
            "latency_stats": self.get_latency_stats(),
            "usage_stats": self.get_usage_stats(),
            "error_stats": self.get_error_stats(),
            "performance_alerts": self.get_performance_alerts(),
            "tool_usage": dict(self.tool_usage_counts)
        }


# Global observability instance
mcp_observability = MCPObservability()


def mcp_latency_monitor(func):
    """Decorator to monitor MCP tool latency"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        success = False
        error_type = None
        
        try:
            result = await func(*args, **kwargs)
            success = True
            return result
        except Exception as e:
            error_type = type(e).__name__
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            tool_name = func.__name__
            
            mcp_observability.record_latency(
                tool_name=tool_name,
                latency_ms=latency_ms,
                success=success,
                error_type=error_type
            )
    
    return wrapper
