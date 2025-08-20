import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from app.core.mcp_observability import MCPObservability, MCPLatencyMetric, MCPUsageMetric


class TestMCPObservability:
    """MCP Observability testleri"""
    
    @pytest.fixture
    def observability(self):
        """Observability instance"""
        return MCPObservability(max_metrics=100)
    
    def test_record_latency_success(self, observability):
        """Latency recording başarı testi"""
        observability.record_latency(
            tool_name="test_tool",
            latency_ms=150.5,
            success=True
        )
        
        assert len(observability.latency_metrics) == 1
        metric = observability.latency_metrics[0]
        assert metric.tool_name == "test_tool"
        assert metric.latency_ms == 150.5
        assert metric.success is True
        assert metric.error_type is None
        assert observability.tool_usage_counts["test_tool"] == 1
    
    def test_record_latency_failure(self, observability):
        """Latency recording hata testi"""
        observability.record_latency(
            tool_name="test_tool",
            latency_ms=5000.0,
            success=False,
            error_type="ConnectionError"
        )
        
        assert len(observability.latency_metrics) == 1
        metric = observability.latency_metrics[0]
        assert metric.success is False
        assert metric.error_type == "ConnectionError"
        assert observability.error_counts["ConnectionError"] == 1
    
    def test_record_usage(self, observability):
        """Usage recording testi"""
        observability.record_usage(
            tool_name="test_tool",
            user_id="user123",
            session_id="session456",
            arguments_size=500,
            response_size=1000
        )
        
        assert len(observability.usage_metrics) == 1
        metric = observability.usage_metrics[0]
        assert metric.tool_name == "test_tool"
        assert metric.user_id == "user123"
        assert metric.session_id == "session456"
        assert metric.arguments_size == 500
        assert metric.response_size == 1000
    
    def test_get_latency_stats_empty(self, observability):
        """Boş latency stats testi"""
        stats = observability.get_latency_stats()
        
        assert stats["count"] == 0
        assert stats["avg_latency_ms"] == 0
        assert stats["success_rate"] == 0
    
    def test_get_latency_stats_with_data(self, observability):
        """Veri ile latency stats testi"""
        # Add some test metrics
        observability.record_latency("tool1", 100.0, True)
        observability.record_latency("tool1", 200.0, True)
        observability.record_latency("tool1", 300.0, False, "Error")
        
        stats = observability.get_latency_stats(tool_name="tool1")
        
        assert stats["count"] == 3
        assert stats["avg_latency_ms"] == 200.0
        assert stats["min_latency_ms"] == 100.0
        assert stats["max_latency_ms"] == 300.0
        assert stats["success_rate"] == 2/3
    
    def test_get_usage_stats(self, observability):
        """Usage stats testi"""
        # Add some test usage metrics
        observability.record_usage("tool1", "user1", "session1", 100, 200)
        observability.record_usage("tool1", "user2", "session2", 150, 300)
        observability.record_usage("tool2", "user1", "session3", 200, 400)
        
        stats = observability.get_usage_stats()
        
        assert stats["total_requests"] == 3
        assert stats["unique_users"] == 2
        assert stats["unique_sessions"] == 3
        assert stats["avg_arguments_size"] == 150
        assert stats["avg_response_size"] == 300
        assert stats["tool_breakdown"]["tool1"] == 2
        assert stats["tool_breakdown"]["tool2"] == 1
    
    def test_get_error_stats(self, observability):
        """Error stats testi"""
        # Add some test metrics with errors
        observability.record_latency("tool1", 100.0, True)
        observability.record_latency("tool1", 200.0, False, "ConnectionError")
        observability.record_latency("tool2", 300.0, False, "TimeoutError")
        
        stats = observability.get_error_stats()
        
        assert stats["total_errors"] == 2
        assert stats["error_breakdown"]["ConnectionError"] == 1
        assert stats["error_breakdown"]["TimeoutError"] == 1
        assert stats["error_rate"] == 2/3
    
    def test_get_health_status_healthy(self, observability):
        """Sağlıklı health status testi"""
        # Add healthy metrics
        observability.record_latency("tool1", 100.0, True)
        observability.record_latency("tool1", 200.0, True)
        
        health = observability.get_health_status()
        
        assert health["status"] == "healthy"
        assert health["total_requests"] == 2
        assert "uptime_seconds" in health
        assert "recent_stats" in health
        assert "error_stats" in health
        assert "tool_usage" in health
    
    def test_get_health_status_unhealthy(self, observability):
        """Sağlıksız health status testi"""
        # Add mostly failed metrics
        for i in range(10):
            observability.record_latency("tool1", 100.0, False, "Error")
        
        health = observability.get_health_status()
        
        assert health["status"] == "unhealthy"
    
    def test_get_health_status_degraded(self, observability):
        """Degraded health status testi"""
        # Add slow metrics
        observability.record_latency("tool1", 15000.0, True)  # 15 seconds
        
        health = observability.get_health_status()
        
        assert health["status"] == "degraded"
    
    def test_get_performance_alerts(self, observability):
        """Performance alerts testi"""
        # Add metrics that should trigger alerts
        for i in range(10):
            observability.record_latency("tool1", 100.0, False, "Error")
        
        alerts = observability.get_performance_alerts()
        
        assert len(alerts) > 0
        assert any(alert["type"] == "high_error_rate" for alert in alerts)
    
    def test_export_metrics(self, observability):
        """Metrics export testi"""
        # Add some test data
        observability.record_latency("tool1", 100.0, True)
        observability.record_usage("tool1", "user1", "session1", 100, 200)
        
        exported = observability.export_metrics()
        
        assert "timestamp" in exported
        assert "health_status" in exported
        assert "latency_stats" in exported
        assert "usage_stats" in exported
        assert "error_stats" in exported
        assert "performance_alerts" in exported
        assert "tool_usage" in exported
    
    def test_metric_rotation(self, observability):
        """Metric rotation testi (max_metrics limit)"""
        # Add more metrics than max_metrics
        for i in range(150):  # More than max_metrics (100)
            observability.record_latency(f"tool{i}", 100.0, True)
        
        # Should only keep the most recent metrics
        assert len(observability.latency_metrics) == 100
        assert len(observability.usage_metrics) == 0  # No usage metrics added
    
    def test_time_window_filtering(self, observability):
        """Time window filtering testi"""
        # Add old metrics
        old_time = datetime.utcnow() - timedelta(hours=2)
        old_metric = MCPLatencyMetric(
            tool_name="old_tool",
            latency_ms=100.0,
            timestamp=old_time,
            success=True
        )
        observability.latency_metrics.append(old_metric)
        
        # Add recent metrics
        observability.record_latency("recent_tool", 200.0, True)
        
        # Get stats for last 60 minutes
        stats = observability.get_latency_stats(time_window_minutes=60)
        
        # Should only include recent metrics
        assert stats["count"] == 1
        assert "recent_tool" in [m.tool_name for m in observability.latency_metrics if m.timestamp >= datetime.utcnow() - timedelta(minutes=60)]
