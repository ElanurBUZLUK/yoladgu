"""
Performance Monitor Endpoints
Sistem performans izleme endpointleri
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from app.services.metrics_service import metrics_service
from app.services.health_check_service import health_check_service

router = APIRouter(prefix="/performance", tags=["performance"])


@router.get("/metrics")
async def get_performance_metrics() -> Dict[str, Any]:
    """Performans metriklerini getir"""
    try:
        metrics = await metrics_service.get_metrics_summary()
        return {
            "status": "success",
            "data": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/metrics/prometheus")
async def get_prometheus_metrics() -> str:
    """Prometheus formatında metrikleri getir"""
    try:
        return await metrics_service.export_prometheus_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {str(e)}")


@router.get("/health")
async def get_performance_health() -> Dict[str, Any]:
    """Performans sağlık kontrolü"""
    try:
        health_data = await health_check_service.get_health_summary()
        return {
            "status": "success",
            "data": health_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health data: {str(e)}")


@router.get("/system")
async def get_system_performance() -> Dict[str, Any]:
    """Sistem performans bilgileri"""
    try:
        # Sistem metriklerini topla
        await metrics_service.collect_system_metrics()
        
        # Sistem metriklerini getir
        metrics = await metrics_service.get_metrics_summary()
        
        return {
            "status": "success",
            "data": {
                "system_metrics": metrics.get("current_metrics", {}).get("gauges", {}),
                "timestamp": metrics.get("timestamp")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system performance: {str(e)}")


@router.get("/application")
async def get_application_performance() -> Dict[str, Any]:
    """Uygulama performans bilgileri"""
    try:
        # Uygulama metriklerini topla
        await metrics_service.collect_application_metrics()
        
        # Uygulama metriklerini getir
        metrics = await metrics_service.get_metrics_summary()
        
        return {
            "status": "success",
            "data": {
                "application_metrics": metrics.get("current_metrics", {}).get("gauges", {}),
                "timestamp": metrics.get("timestamp")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get application performance: {str(e)}")


@router.get("/business")
async def get_business_performance() -> Dict[str, Any]:
    """İş performans bilgileri"""
    try:
        # İş metriklerini topla
        await metrics_service.collect_business_metrics()
        
        # İş metriklerini getir
        metrics = await metrics_service.get_metrics_summary()
        
        return {
            "status": "success",
            "data": {
                "business_metrics": metrics.get("current_metrics", {}).get("counters", {}),
                "timestamp": metrics.get("timestamp")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get business performance: {str(e)}")


@router.get("/api/stats")
async def get_api_performance_stats() -> Dict[str, Any]:
    """API performans istatistikleri"""
    try:
        metrics = await metrics_service.get_metrics_summary()
        
        # API metriklerini filtrele
        api_metrics = {}
        for name, value in metrics.get("current_metrics", {}).get("counters", {}).items():
            if name.startswith("api_"):
                api_metrics[name] = value
        
        return {
            "status": "success",
            "data": {
                "api_metrics": api_metrics,
                "timestamp": metrics.get("timestamp")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get API stats: {str(e)}")


@router.get("/user/activity")
async def get_user_activity_stats() -> Dict[str, Any]:
    """Kullanıcı aktivite istatistikleri"""
    try:
        metrics = await metrics_service.get_metrics_summary()
        
        # Kullanıcı aktivite metriklerini filtrele
        user_metrics = {}
        for name, value in metrics.get("current_metrics", {}).get("counters", {}).items():
            if name.startswith("user_"):
                user_metrics[name] = value
        
        return {
            "status": "success",
            "data": {
                "user_activity_metrics": user_metrics,
                "timestamp": metrics.get("timestamp")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user activity stats: {str(e)}")


@router.get("/ml/stats")
async def get_ml_performance_stats() -> Dict[str, Any]:
    """ML performans istatistikleri"""
    try:
        metrics = await metrics_service.get_metrics_summary()
        
        # ML metriklerini filtrele
        ml_metrics = {}
        for name, value in metrics.get("current_metrics", {}).get("gauges", {}).items():
            if name.startswith("ml_"):
                ml_metrics[name] = value
        
        return {
            "status": "success",
            "data": {
                "ml_metrics": ml_metrics,
                "timestamp": metrics.get("timestamp")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get ML stats: {str(e)}") 