from typing import Dict, Any, Optional
import time
import psutil
import structlog
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess
)
from ..core.config import settings

logger = structlog.get_logger()


class MonitoringService:
    """Prometheus metrics and monitoring service"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self._initialize_metrics()
        logger.info("Monitoring service initialized")
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics"""
        # Request metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # LLM metrics
        self.llm_requests_total = Counter(
            'llm_requests_total',
            'Total LLM API requests',
            ['provider', 'model', 'status'],
            registry=self.registry
        )
        
        self.llm_request_duration = Histogram(
            'llm_request_duration_seconds',
            'LLM request duration in seconds',
            ['provider', 'model'],
            registry=self.registry
        )
        
        self.llm_tokens_used = Counter(
            'llm_tokens_used_total',
            'Total tokens used in LLM requests',
            ['provider', 'model', 'token_type'],
            registry=self.registry
        )
        
        self.llm_cost_total = Counter(
            'llm_cost_total',
            'Total cost of LLM requests',
            ['provider', 'model'],
            registry=self.registry
        )
        
        # Database metrics
        self.db_connections = Gauge(
            'database_connections_active',
            'Number of active database connections',
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration in seconds',
            ['operation'],
            registry=self.registry
        )
        
        # Vector search metrics
        self.vector_search_requests = Counter(
            'vector_search_requests_total',
            'Total vector search requests',
            ['namespace', 'status'],
            registry=self.registry
        )
        
        self.vector_search_duration = Histogram(
            'vector_search_duration_seconds',
            'Vector search duration in seconds',
            ['namespace'],
            registry=self.registry
        )
        
        # User activity metrics
        self.active_users = Gauge(
            'active_users_total',
            'Number of active users',
            registry=self.registry
        )
        
        self.user_sessions = Gauge(
            'user_sessions_total',
            'Number of active user sessions',
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_bytes',
            'Disk usage in bytes',
            ['mount_point'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'application_errors_total',
            'Total application errors',
            ['error_type', 'endpoint'],
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.rate_limit_hits = Counter(
            'rate_limit_hits_total',
            'Total rate limit hits',
            ['endpoint', 'user_id'],
            registry=self.registry
        )
        
        # Content moderation metrics
        self.content_moderation_requests = Counter(
            'content_moderation_requests_total',
            'Total content moderation requests',
            ['risk_level', 'action'],
            registry=self.registry
        )
        
        # File upload metrics
        self.file_uploads = Counter(
            'file_uploads_total',
            'Total file uploads',
            ['file_type', 'status'],
            registry=self.registry
        )
        
        self.file_upload_size = Histogram(
            'file_upload_size_bytes',
            'File upload size in bytes',
            ['file_type'],
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_llm_request(self, provider: str, model: str, status: str, duration: float, 
                          tokens_used: int = 0, cost: float = 0.0):
        """Record LLM request metrics"""
        self.llm_requests_total.labels(provider=provider, model=model, status=status).inc()
        self.llm_request_duration.labels(provider=provider, model=model).observe(duration)
        
        if tokens_used > 0:
            self.llm_tokens_used.labels(provider=provider, model=model, token_type="total").inc(tokens_used)
        
        if cost > 0:
            self.llm_cost_total.labels(provider=provider, model=model).inc(cost)
    
    def record_db_operation(self, operation: str, duration: float):
        """Record database operation metrics"""
        self.db_query_duration.labels(operation=operation).observe(duration)
    
    def set_db_connections(self, count: int):
        """Set active database connections count"""
        self.db_connections.set(count)
    
    def record_vector_search(self, namespace: str, status: str, duration: float):
        """Record vector search metrics"""
        self.vector_search_requests.labels(namespace=namespace, status=status).inc()
        self.vector_search_duration.labels(namespace=namespace).observe(duration)
    
    def set_active_users(self, count: int):
        """Set active users count"""
        self.active_users.set(count)
    
    def set_user_sessions(self, count: int):
        """Set active user sessions count"""
        self.user_sessions.set(count)
    
    def record_error(self, error_type: str, endpoint: str):
        """Record application error"""
        self.error_count.labels(error_type=error_type, endpoint=endpoint).inc()
    
    def record_rate_limit_hit(self, endpoint: str, user_id: str):
        """Record rate limit hit"""
        self.rate_limit_hits.labels(endpoint=endpoint, user_id=user_id).inc()
    
    def record_content_moderation(self, risk_level: str, action: str):
        """Record content moderation event"""
        self.content_moderation_requests.labels(risk_level=risk_level, action=action).inc()
    
    def record_file_upload(self, file_type: str, status: str, size_bytes: int = 0):
        """Record file upload metrics"""
        self.file_uploads.labels(file_type=file_type, status=status).inc()
        if size_bytes > 0:
            self.file_upload_size.labels(file_type=file_type).observe(size_bytes)
    
    def update_system_metrics(self):
        """Update system metrics (CPU, memory, disk)"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            
            # Disk usage
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    self.disk_usage.labels(mount_point=partition.mountpoint).set(usage.used)
                except PermissionError:
                    continue
                    
        except Exception as e:
            logger.error("Error updating system metrics", error=str(e))
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics as string"""
        try:
            return generate_latest(self.registry)
        except Exception as e:
            logger.error("Error generating metrics", error=str(e))
            return ""
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        try:
            # System health
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk health
            disk_health = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_health[partition.mountpoint] = {
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": (usage.used / usage.total) * 100
                    }
                except PermissionError:
                    continue
            
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used": memory.used,
                    "memory_total": memory.total
                },
                "disk": disk_health,
                "metrics_enabled": settings.prometheus_enabled
            }
            
        except Exception as e:
            logger.error("Error getting health status", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }


# Global monitoring service instance
monitoring_service = MonitoringService()


def get_grafana_dashboard_config() -> Dict[str, Any]:
    """Get Grafana dashboard configuration"""
    return {
        "dashboard": {
            "id": None,
            "title": "Adaptive Learning System Dashboard",
            "tags": ["adaptive-learning", "api", "monitoring"],
            "style": "dark",
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "HTTP Requests",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(http_requests_total[5m])",
                            "legendFormat": "{{method}} {{endpoint}}"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                },
                {
                    "id": 2,
                    "title": "LLM Requests",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(llm_requests_total[5m])",
                            "legendFormat": "{{provider}} {{model}}"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                },
                {
                    "id": 3,
                    "title": "System Resources",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "system_cpu_usage_percent",
                            "legendFormat": "CPU Usage"
                        },
                        {
                            "expr": "system_memory_usage_bytes / 1024 / 1024",
                            "legendFormat": "Memory Usage (MB)"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                },
                {
                    "id": 4,
                    "title": "Active Users",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "active_users_total",
                            "legendFormat": "Active Users"
                        }
                    ],
                    "gridPos": {"h": 4, "w": 6, "x": 12, "y": 8}
                },
                {
                    "id": 5,
                    "title": "Error Rate",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(application_errors_total[5m])",
                            "legendFormat": "{{error_type}}"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
                },
                {
                    "id": 6,
                    "title": "Vector Search Performance",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(vector_search_requests_total[5m])",
                            "legendFormat": "{{namespace}}"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
                }
            ],
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "refresh": "30s"
        }
    }
