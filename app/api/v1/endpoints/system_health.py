"""
System Health Endpoints
Sistem sağlık izleme endpointleri
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from datetime import datetime
from app.services.health_check_service import health_check_service
from app.services.metrics_service import metrics_service

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health")
async def get_system_health() -> Dict[str, Any]:
    """Sistem sağlık kontrolü"""
    try:
        health_data = await health_check_service.check_all_health()
        
        return {
            "status": "success",
            "data": health_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


@router.get("/health/summary")
async def get_health_summary() -> Dict[str, Any]:
    """Sağlık özeti"""
    try:
        summary = await health_check_service.get_health_summary()
        
        return {
            "status": "success",
            "data": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health summary: {str(e)}")


@router.get("/health/{service_name}")
async def get_service_health(service_name: str) -> Dict[str, Any]:
    """Belirli servisin sağlık kontrolü"""
    try:
        health_data = await health_check_service.check_all_health()
        
        if service_name not in health_data.get("services", {}):
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        
        service_health = health_data["services"][service_name]
        
        return {
            "status": "success",
            "data": {
                "service_name": service_name,
                "health_status": service_health
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service health: {str(e)}")


@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """Sistem durumu"""
    try:
        # Sistem durumu bilgileri
        status = {
            "system_status": "operational",
            "uptime_seconds": 86400,  # 24 hours
            "version": "1.0.0",
            "environment": "production",
            "last_restart": (datetime.now() - timedelta(hours=24)).isoformat(),
            "services_count": 8,
            "healthy_services": 7,
            "degraded_services": 1,
            "unhealthy_services": 0
        }
        
        return {
            "status": "success",
            "data": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@router.get("/info")
async def get_system_info() -> Dict[str, Any]:
    """Sistem bilgileri"""
    try:
        # Sistem bilgileri
        info = {
            "application_name": "Yoladgu",
            "version": "1.0.0",
            "description": "AI-powered educational platform",
            "author": "Yoladgu Team",
            "license": "MIT",
            "repository": "https://github.com/yoladgu/yoladgu",
            "documentation": "https://docs.yoladgu.com",
            "support_email": "support@yoladgu.com",
            "features": [
                "Vector database integration",
                "AI-powered recommendations",
                "Real-time streaming",
                "Advanced analytics",
                "Multi-model ML system"
            ]
        }
        
        return {
            "status": "success",
            "data": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")


@router.get("/metrics")
async def get_system_metrics() -> Dict[str, Any]:
    """Sistem metrikleri"""
    try:
        # Sistem metriklerini topla
        await metrics_service.collect_system_metrics()
        
        # Metrikleri getir
        metrics = await metrics_service.get_metrics_summary()
        
        return {
            "status": "success",
            "data": {
                "system_metrics": metrics.get("current_metrics", {}).get("gauges", {}),
                "timestamp": metrics.get("timestamp")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")


@router.get("/resources")
async def get_system_resources() -> Dict[str, Any]:
    """Sistem kaynakları"""
    try:
        import psutil
        
        # CPU bilgileri
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory bilgileri
        memory = psutil.virtual_memory()
        
        # Disk bilgileri
        disk = psutil.disk_usage('/')
        
        # Network bilgileri
        network = psutil.net_io_counters()
        
        resources = {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": cpu_count,
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "usage_percent": memory.percent
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "used_gb": disk.used / (1024**3),
                "usage_percent": disk.percent
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        }
        
        return {
            "status": "success",
            "data": resources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system resources: {str(e)}")


@router.get("/alerts")
async def get_system_alerts() -> Dict[str, Any]:
    """Sistem uyarıları"""
    try:
        # Mock alerts
        alerts = [
            {
                "alert_id": "alert_001",
                "severity": "warning",
                "message": "High memory usage detected",
                "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat(),
                "service": "system",
                "resolved": False
            },
            {
                "alert_id": "alert_002",
                "severity": "info",
                "message": "Database backup completed successfully",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "service": "database",
                "resolved": True
            }
        ]
        
        return {
            "status": "success",
            "data": {
                "alerts": alerts,
                "total_alerts": len(alerts),
                "active_alerts": len([a for a in alerts if not a["resolved"]]),
                "resolved_alerts": len([a for a in alerts if a["resolved"]])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system alerts: {str(e)}")


@router.post("/health/check")
async def trigger_health_check() -> Dict[str, Any]:
    """Manuel sağlık kontrolü tetikle"""
    try:
        # Tüm servislerin sağlık kontrolünü yap
        health_data = await health_check_service.check_all_health()
        
        return {
            "status": "success",
            "message": "Health check completed",
            "data": {
                "overall_status": health_data.get("overall_status"),
                "services_checked": len(health_data.get("services", {})),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger health check: {str(e)}")


# Import timedelta for mock data
from datetime import timedelta 