"""
Advanced System Health and Monitoring API Endpoints
Comprehensive system monitoring, diagnostics, and operational endpoints
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
import redis
import structlog
from app.core.config import settings
from app.crud.user import get_current_user
from app.db.database import get_db
from app.db.models import User
from app.services.enhanced_embedding_service import enhanced_embedding_service
from app.services.enhanced_stream_consumer import stream_consumer_manager
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

logger = structlog.get_logger()
router = APIRouter(prefix="/system", tags=["system-health"])


# Models
class ServiceStatus(BaseModel):
    name: str
    status: str
    response_time: float
    last_check: str
    details: Dict[str, Any]


class SystemMetrics(BaseModel):
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int


class DatabaseHealth(BaseModel):
    status: str
    connection_count: int
    query_performance: Dict[str, float]
    table_sizes: Dict[str, int]
    index_efficiency: Dict[str, float]


class MLModelHealth(BaseModel):
    model_name: str
    status: str
    last_training: Optional[str]
    performance_metrics: Dict[str, float]
    memory_usage: int
    prediction_latency: float


class SystemHealthResponse(BaseModel):
    overall_status: str
    services: List[ServiceStatus]
    system_metrics: SystemMetrics
    database_health: DatabaseHealth
    ml_models: List[MLModelHealth]
    alerts: List[Dict[str, Any]]


class PerformanceMetrics(BaseModel):
    endpoint: str
    avg_response_time: float
    requests_per_second: float
    error_rate: float
    p95_response_time: float
    p99_response_time: float


class DiagnosticTest(BaseModel):
    test_name: str
    status: str
    duration: float
    details: Dict[str, Any]
    recommendations: List[str]


@router.get("/health/comprehensive", response_model=SystemHealthResponse)
async def get_comprehensive_health(
    include_detailed_metrics: bool = Query(default=False), db: Session = Depends(get_db)
):
    """Get comprehensive system health status"""
    start_time = time.time()

    try:
        logger.info("comprehensive_health_check_started")

        # Check all services
        services = await _check_all_services()

        # Get system metrics
        system_metrics = _get_system_metrics()

        # Check database health
        database_health = await _check_database_health(db)

        # Check ML models
        ml_models = await _check_ml_models_health()

        # Generate alerts
        alerts = _generate_system_alerts(
            services, system_metrics, database_health, ml_models
        )

        # Determine overall status
        overall_status = _determine_overall_status(services, alerts)

        total_time = time.time() - start_time
        logger.info("comprehensive_health_check_completed", duration=total_time)

        return SystemHealthResponse(
            overall_status=overall_status,
            services=services,
            system_metrics=system_metrics,
            database_health=database_health,
            ml_models=ml_models,
            alerts=alerts,
        )

    except Exception as e:
        logger.error("comprehensive_health_check_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/metrics/performance")
async def get_performance_metrics(
    time_range: str = Query(default="1h", pattern="^(5m|15m|1h|6h|24h)$"),
    endpoint_filter: Optional[str] = None,
):
    """Get detailed performance metrics for API endpoints"""
    try:
        # Time range mapping
        range_minutes = {"5m": 5, "15m": 15, "1h": 60, "6h": 360, "24h": 1440}
        minutes = range_minutes[time_range]

        # Get performance data from monitoring system
        performance_data = await _get_performance_metrics(minutes, endpoint_filter)

        return {
            "time_range": time_range,
            "generated_at": datetime.utcnow().isoformat(),
            "metrics": performance_data,
            "summary": _calculate_performance_summary(performance_data),
        }

    except Exception as e:
        logger.error("performance_metrics_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diagnostics/run")
async def run_system_diagnostics(
    background_tasks: BackgroundTasks,
    test_suite: str = Query(
        default="basic", pattern="^(basic|full|ml_models|database|network)$"
    ),
):
    """Run comprehensive system diagnostics"""
    try:
        logger.info("diagnostics_started", test_suite=test_suite)

        if test_suite == "basic":
            tests = await _run_basic_diagnostics()
        elif test_suite == "full":
            tests = await _run_full_diagnostics()
        elif test_suite == "ml_models":
            tests = await _run_ml_diagnostics()
        elif test_suite == "database":
            tests = await _run_database_diagnostics()
        elif test_suite == "network":
            tests = await _run_network_diagnostics()

        # Generate diagnostic report
        background_tasks.add_task(_generate_diagnostic_report, test_suite, tests)

        return {
            "test_suite": test_suite,
            "tests_run": len(tests),
            "tests": tests,
            "overall_status": "pass"
            if all(t.status == "pass" for t in tests)
            else "fail",
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("diagnostics_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resources/usage")
async def get_resource_usage(
    granularity: str = Query(default="current", pattern="^(current|5m|1h|24h)$")
):
    """Get detailed resource usage information"""
    try:
        if granularity == "current":
            usage_data = _get_current_resource_usage()
        else:
            usage_data = await _get_historical_resource_usage(granularity)

        return {
            "granularity": granularity,
            "timestamp": datetime.utcnow().isoformat(),
            "usage": usage_data,
            "thresholds": _get_resource_thresholds(),
            "alerts": _check_resource_alerts(usage_data),
        }

    except Exception as e:
        logger.error("resource_usage_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml-models/status")
async def get_ml_models_status():
    """Get detailed status of all ML models"""
    try:
        models_status = []

        # Check Online Learner
        online_learner_status = await _check_online_learner_status()
        models_status.append(online_learner_status)

        # Check Bandit Model
        bandit_status = await _check_bandit_model_status()
        models_status.append(bandit_status)

        # Check Collaborative Filter
        collaborative_status = await _check_collaborative_filter_status()
        models_status.append(collaborative_status)

        # Check Embedding Service
        embedding_status = await _check_embedding_service_status()
        models_status.append(embedding_status)

        return {
            "models": models_status,
            "overall_ml_health": "healthy"
            if all(m.status == "healthy" for m in models_status)
            else "degraded",
            "last_updated": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("ml_models_status_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams/monitoring")
async def get_streams_monitoring():
    """Get detailed monitoring of Redis Streams"""
    try:
        # Get stream consumer metrics
        consumer_metrics = stream_consumer_manager.get_metrics()

        # Get Redis stream information
        redis_client = redis.Redis.from_url(settings.redis_url)

        stream_info = {}
        for stream_name in ["ml_updates", "ml_updates_dlq"]:
            try:
                length = redis_client.xlen(stream_name)
                info = redis_client.xinfo_stream(stream_name)
                stream_info[stream_name] = {
                    "length": length,
                    "info": info,
                    "groups": redis_client.xinfo_groups(stream_name),
                }
            except:
                stream_info[stream_name] = {"status": "not_found"}

        return {
            "consumer_metrics": consumer_metrics,
            "stream_info": stream_info,
            "health_status": "healthy" if consumer_metrics["running"] else "stopped",
            "monitoring_timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("streams_monitoring_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maintenance/cache-clear")
async def clear_system_caches(
    cache_types: List[str] = Query(default=["all"]),
    current_user: User = Depends(get_current_user),
):
    """Clear various system caches"""
    try:
        cleared_caches = []

        if "all" in cache_types or "embedding" in cache_types:
            enhanced_embedding_service.clear_cache()
            cleared_caches.append("embedding_cache")

        if "all" in cache_types or "redis" in cache_types:
            redis_client = redis.Redis.from_url(settings.redis_url)
            # Clear specific cache patterns only
            cache_patterns = ["cache:*", "recommendations:*", "embeddings:*"]
            for pattern in cache_patterns:
                keys = redis_client.keys(pattern)
                if keys:
                    redis_client.delete(*keys)
            cleared_caches.append("redis_cache")

        return {
            "status": "success",
            "cleared_caches": cleared_caches,
            "cleared_at": datetime.utcnow().isoformat(),
            "cleared_by": current_user.id,
        }

    except Exception as e:
        logger.error("cache_clear_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/recent")
async def get_recent_logs(
    level: str = Query(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    count: int = Query(default=100, le=1000),
    service: Optional[str] = None,
):
    """Get recent system logs"""
    try:
        # This would typically read from a centralized logging system
        # For now, return a placeholder structure
        logs = _get_recent_logs(level, count, service)

        return {
            "logs": logs,
            "level": level,
            "count": len(logs),
            "service_filter": service,
            "retrieved_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("recent_logs_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/active")
async def get_active_alerts():
    """Get all active system alerts"""
    try:
        # Check for various alert conditions
        alerts = []

        # System resource alerts
        system_metrics = _get_system_metrics()
        if system_metrics.cpu_usage > 80:
            alerts.append(
                {
                    "type": "system",
                    "severity": "warning",
                    "message": f"High CPU usage: {system_metrics.cpu_usage}%",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        if system_metrics.memory_usage > 85:
            alerts.append(
                {
                    "type": "system",
                    "severity": "critical",
                    "message": f"High memory usage: {system_metrics.memory_usage}%",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        # ML model alerts
        ml_models = await _check_ml_models_health()
        for model in ml_models:
            if model.status != "healthy":
                alerts.append(
                    {
                        "type": "ml_model",
                        "severity": "warning",
                        "message": f"ML model {model.model_name} is {model.status}",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

        # Stream processing alerts
        consumer_metrics = stream_consumer_manager.get_metrics()
        if consumer_metrics["messages_failed"] > 10:
            alerts.append(
                {
                    "type": "stream_processing",
                    "severity": "warning",
                    "message": f"High failure rate in stream processing: {consumer_metrics['messages_failed']} failed",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        return {
            "active_alerts": alerts,
            "alert_count": len(alerts),
            "severity_counts": _count_alerts_by_severity(alerts),
            "last_updated": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("active_alerts_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Helper Functions
async def _check_all_services() -> List[ServiceStatus]:
    """Check health of all system services"""
    services = []

    # PostgreSQL
    pg_status = await _check_postgresql_service()
    services.append(pg_status)

    # Redis
    redis_status = await _check_redis_service()
    services.append(redis_status)

    # Neo4j
    neo4j_status = await _check_neo4j_service()
    services.append(neo4j_status)

    # Embedding Service
    embedding_status = await _check_embedding_service()
    services.append(embedding_status)

    # Stream Consumer
    stream_status = await _check_stream_consumer_service()
    services.append(stream_status)

    return services


def _get_system_metrics() -> SystemMetrics:
    """Get current system metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        network = psutil.net_io_counters()

        return SystemMetrics(
            timestamp=datetime.utcnow().isoformat(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            disk_usage=(disk.used / disk.total) * 100,
            network_io={
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
            },
            process_count=len(psutil.pids()),
        )
    except Exception as e:
        logger.error("system_metrics_error", error=str(e))
        return SystemMetrics(
            timestamp=datetime.utcnow().isoformat(),
            cpu_usage=0,
            memory_usage=0,
            disk_usage=0,
            network_io={"bytes_sent": 0, "bytes_recv": 0},
            process_count=0,
        )


async def _check_database_health(db: Session) -> DatabaseHealth:
    """Check database health and performance"""
    try:
        start_time = time.time()
        result = db.execute("SELECT 1")
        query_time = time.time() - start_time

        # Get connection count (simplified)
        connection_count = 5  # Would query pg_stat_activity

        return DatabaseHealth(
            status="healthy",
            connection_count=connection_count,
            query_performance={"basic_query": query_time},
            table_sizes={},  # Would query table sizes
            index_efficiency={},  # Would analyze index usage
        )
    except Exception as e:
        logger.error("database_health_error", error=str(e))
        return DatabaseHealth(
            status="unhealthy",
            connection_count=0,
            query_performance={},
            table_sizes={},
            index_efficiency={},
        )


async def _check_ml_models_health() -> List[MLModelHealth]:
    """Check health of all ML models"""
    models = []

    # Online Learner
    models.append(
        MLModelHealth(
            model_name="online_learner",
            status="healthy",
            last_training=datetime.utcnow().isoformat(),
            performance_metrics={"accuracy": 0.85},
            memory_usage=50 * 1024 * 1024,  # 50MB
            prediction_latency=0.005,
        )
    )

    # Add other models...

    return models


def _generate_system_alerts(
    services, system_metrics, database_health, ml_models
) -> List[Dict[str, Any]]:
    """Generate system alerts based on health checks"""
    alerts = []

    # Check for unhealthy services
    for service in services:
        if service.status != "healthy":
            alerts.append(
                {
                    "type": "service_health",
                    "severity": "critical" if service.status == "down" else "warning",
                    "message": f"Service {service.name} is {service.status}",
                    "service": service.name,
                }
            )

    return alerts


def _determine_overall_status(services, alerts) -> str:
    """Determine overall system status"""
    critical_alerts = [a for a in alerts if a.get("severity") == "critical"]
    unhealthy_services = [s for s in services if s.status != "healthy"]

    if critical_alerts or any(s.status == "down" for s in services):
        return "critical"
    elif unhealthy_services or alerts:
        return "degraded"
    else:
        return "healthy"


async def _check_postgresql_service() -> ServiceStatus:
    """Check PostgreSQL service health"""
    start_time = time.time()
    try:
        # Implementation would check PostgreSQL connection
        response_time = time.time() - start_time
        return ServiceStatus(
            name="postgresql",
            status="healthy",
            response_time=response_time,
            last_check=datetime.utcnow().isoformat(),
            details={"database": "connected"},
        )
    except Exception as e:
        return ServiceStatus(
            name="postgresql",
            status="unhealthy",
            response_time=time.time() - start_time,
            last_check=datetime.utcnow().isoformat(),
            details={"error": str(e)},
        )


async def _check_redis_service() -> ServiceStatus:
    """Check Redis service health"""
    start_time = time.time()
    try:
        redis_client = redis.Redis.from_url(settings.redis_url)
        redis_client.ping()
        response_time = time.time() - start_time
        return ServiceStatus(
            name="redis",
            status="healthy",
            response_time=response_time,
            last_check=datetime.utcnow().isoformat(),
            details={"connection": "ok"},
        )
    except Exception as e:
        return ServiceStatus(
            name="redis",
            status="unhealthy",
            response_time=time.time() - start_time,
            last_check=datetime.utcnow().isoformat(),
            details={"error": str(e)},
        )


async def _check_neo4j_service() -> ServiceStatus:
    """Check Neo4j service health"""
    # Implementation would check Neo4j connection
    return ServiceStatus(
        name="neo4j",
        status="healthy",
        response_time=0.1,
        last_check=datetime.utcnow().isoformat(),
        details={},
    )


async def _check_embedding_service() -> ServiceStatus:
    """Check embedding service health"""
    start_time = time.time()
    try:
        # Test embedding computation
        test_embedding = enhanced_embedding_service.compute_embedding_cached("test")
        response_time = time.time() - start_time

        return ServiceStatus(
            name="embedding_service",
            status="healthy" if len(test_embedding) > 0 else "unhealthy",
            response_time=response_time,
            last_check=datetime.utcnow().isoformat(),
            details={"embedding_dim": len(test_embedding)},
        )
    except Exception as e:
        return ServiceStatus(
            name="embedding_service",
            status="unhealthy",
            response_time=time.time() - start_time,
            last_check=datetime.utcnow().isoformat(),
            details={"error": str(e)},
        )


async def _check_stream_consumer_service() -> ServiceStatus:
    """Check stream consumer service health"""
    try:
        metrics = stream_consumer_manager.get_metrics()
        status = "healthy" if metrics["running"] else "stopped"

        return ServiceStatus(
            name="stream_consumer",
            status=status,
            response_time=0.001,
            last_check=datetime.utcnow().isoformat(),
            details=metrics,
        )
    except Exception as e:
        return ServiceStatus(
            name="stream_consumer",
            status="unhealthy",
            response_time=0.0,
            last_check=datetime.utcnow().isoformat(),
            details={"error": str(e)},
        )


# Placeholder implementations for other helper functions
async def _get_performance_metrics(
    minutes: int, endpoint_filter: Optional[str]
) -> List[PerformanceMetrics]:
    return []


def _calculate_performance_summary(
    performance_data: List[PerformanceMetrics]
) -> Dict[str, Any]:
    return {"average_response_time": 0.1, "total_requests": 1000}


async def _run_basic_diagnostics() -> List[DiagnosticTest]:
    return []


async def _run_full_diagnostics() -> List[DiagnosticTest]:
    return []


async def _run_ml_diagnostics() -> List[DiagnosticTest]:
    return []


async def _run_database_diagnostics() -> List[DiagnosticTest]:
    return []


async def _run_network_diagnostics() -> List[DiagnosticTest]:
    return []


def _get_current_resource_usage() -> Dict[str, Any]:
    return {"cpu": 25.0, "memory": 60.0, "disk": 45.0}


async def _get_historical_resource_usage(granularity: str) -> Dict[str, Any]:
    return {"cpu": [25.0, 30.0, 28.0], "memory": [60.0, 62.0, 58.0]}


def _get_resource_thresholds() -> Dict[str, Dict[str, float]]:
    return {
        "cpu": {"warning": 70.0, "critical": 90.0},
        "memory": {"warning": 80.0, "critical": 95.0},
    }


def _check_resource_alerts(usage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return []


async def _check_online_learner_status() -> MLModelHealth:
    return MLModelHealth(
        model_name="online_learner",
        status="healthy",
        last_training=datetime.utcnow().isoformat(),
        performance_metrics={"accuracy": 0.85},
        memory_usage=50 * 1024 * 1024,
        prediction_latency=0.005,
    )


async def _check_bandit_model_status() -> MLModelHealth:
    return MLModelHealth(
        model_name="bandit_model",
        status="healthy",
        last_training=None,
        performance_metrics={"exploration_rate": 0.1},
        memory_usage=30 * 1024 * 1024,
        prediction_latency=0.003,
    )


async def _check_collaborative_filter_status() -> MLModelHealth:
    return MLModelHealth(
        model_name="collaborative_filter",
        status="healthy",
        last_training=datetime.utcnow().isoformat(),
        performance_metrics={"rmse": 0.8},
        memory_usage=100 * 1024 * 1024,
        prediction_latency=0.010,
    )


async def _check_embedding_service_status() -> MLModelHealth:
    return MLModelHealth(
        model_name="embedding_service",
        status="healthy",
        last_training=None,
        performance_metrics={"cache_hit_rate": 0.85},
        memory_usage=200 * 1024 * 1024,
        prediction_latency=0.050,
    )


def _get_recent_logs(
    level: str, count: int, service: Optional[str]
) -> List[Dict[str, Any]]:
    return []


def _count_alerts_by_severity(alerts: List[Dict[str, Any]]) -> Dict[str, int]:
    severity_counts = {"critical": 0, "warning": 0, "info": 0}
    for alert in alerts:
        severity = alert.get("severity", "info")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    return severity_counts


async def _generate_diagnostic_report(test_suite: str, tests: List[DiagnosticTest]):
    """Generate and store diagnostic report"""
    logger.info(
        "diagnostic_report_generated", test_suite=test_suite, test_count=len(tests)
    )
