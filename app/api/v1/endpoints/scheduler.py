"""
Scheduler Management API Endpoints
Offline batch işlemler ve cron job yönetimi
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import structlog
from app.core.config import settings
from app.services.scheduler_service import offline_scheduler
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

logger = structlog.get_logger()

router = APIRouter()


class ModelMigrationRequest(BaseModel):
    new_model_key: str
    force: bool = False


class ScheduleTaskRequest(BaseModel):
    task_type: str
    scheduled_time: Optional[datetime] = None
    parameters: Optional[Dict[str, Any]] = None


@router.get("/status")
async def get_scheduler_status():
    """Scheduler durumu ve aktif job'lar"""
    try:
        status = offline_scheduler.get_scheduler_status()
        return {"scheduler_status": status, "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error("scheduler_status_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger/daily-sync")
async def trigger_daily_sync(background_tasks: BackgroundTasks):
    """Manuel günlük embedding sync tetikleme"""
    try:
        # Background task olarak çalıştır
        background_tasks.add_task(offline_scheduler.daily_embedding_sync)

        return {
            "message": "Daily embedding sync triggered",
            "status": "scheduled",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("daily_sync_trigger_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger/full-reindex")
async def trigger_full_reindex(background_tasks: BackgroundTasks):
    """Manuel full reindex tetikleme"""
    try:
        # Background task olarak çalıştır
        background_tasks.add_task(offline_scheduler.weekly_full_reindex)

        return {
            "message": "Full reindex triggered",
            "status": "scheduled",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("full_reindex_trigger_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger/cache-cleanup")
async def trigger_cache_cleanup(background_tasks: BackgroundTasks):
    """Manuel cache cleanup tetikleme"""
    try:
        background_tasks.add_task(offline_scheduler.cleanup_expired_cache)

        return {
            "message": "Cache cleanup triggered",
            "status": "scheduled",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("cache_cleanup_trigger_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model-migration")
async def trigger_model_migration(request: ModelMigrationRequest):
    """Embedding model migration tetikleme"""
    try:
        # Validate model key
        from app.services.enhanced_embedding_service import enhanced_embedding_service

        if request.new_model_key not in enhanced_embedding_service.available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model key. Available: {list(enhanced_embedding_service.available_models.keys())}",
            )

        # Trigger migration
        task_id = await offline_scheduler.trigger_model_migration(
            new_model_key=request.new_model_key, force=request.force
        )

        return {
            "message": "Model migration scheduled",
            "task_id": task_id,
            "new_model": request.new_model_key,
            "force": request.force,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("model_migration_trigger_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
async def get_recent_tasks(
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    limit: int = Query(20, ge=1, le=100, description="Number of tasks to return"),
    status: Optional[str] = Query(None, description="Filter by status"),
):
    """Son task'ları listele"""
    try:
        # Redis'ten task'ları çek
        import json

        import redis
        from app.services.scheduler_service import offline_scheduler

        redis_client = redis.from_url(settings.redis_url)
        pattern = f"{offline_scheduler.task_prefix}:*"
        keys = redis_client.keys(pattern)

        tasks = []
        for key in keys[:limit]:
            task_data = redis_client.get(key)
            if task_data:
                task = json.loads(task_data)

                # Filtrele
                if task_type and task.get("task_type") != task_type:
                    continue
                if status and task.get("status") != status:
                    continue

                tasks.append(task)

        # Created_at'a göre sırala (en yeni önce)
        tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return {
            "tasks": tasks[:limit],
            "total_found": len(tasks),
            "filters": {"task_type": task_type, "status": status, "limit": limit},
        }

    except Exception as e:
        logger.error("get_tasks_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}")
async def get_task_details(task_id: str):
    """Specific task detayları"""
    try:
        task = await offline_scheduler._get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Task'ı dict'e çevir
        from dataclasses import asdict

        task_dict = asdict(task)

        # Datetime'ları string'e çevir
        for key, value in task_dict.items():
            if isinstance(value, datetime):
                task_dict[key] = value.isoformat()

        return {"task": task_dict, "timestamp": datetime.utcnow().isoformat()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_task_details_error", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_scheduler_stats():
    """Scheduler istatistikleri"""
    try:
        import json

        import redis

        redis_client = redis.from_url(settings.redis_url)

        # Summary stats
        stats_key = f"{offline_scheduler.stats_key}:summary"
        summary_data = redis_client.get(stats_key)
        summary_stats = json.loads(summary_data) if summary_data else {}

        # Vector stats (son 24 saat)
        vector_stats = []
        now = datetime.utcnow()
        for i in range(24):
            hour_time = now - timedelta(hours=i)
            hour_key = f"{offline_scheduler.stats_key}:vector:{hour_time.strftime('%Y%m%d_%H')}"
            hour_data = redis_client.get(hour_key)

            if hour_data:
                stats = json.loads(hour_data)
                stats["hour"] = hour_time.strftime("%Y-%m-%d %H:00")
                vector_stats.append(stats)

        # Job execution stats
        jobs_status = offline_scheduler.get_scheduler_status()

        return {
            "summary_stats": summary_stats,
            "vector_stats_24h": vector_stats,
            "scheduler_status": jobs_status,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("get_scheduler_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Cron job'ı duraklat"""
    try:
        job = offline_scheduler.scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        job.pause()

        return {
            "message": f"Job {job_id} paused",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("pause_job_error", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Cron job'ı devam ettir"""
    try:
        job = offline_scheduler.scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        job.resume()

        return {
            "message": f"Job {job_id} resumed",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("resume_job_error", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def remove_job(job_id: str):
    """Cron job'ı kaldır"""
    try:
        job = offline_scheduler.scheduler.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Default job'ları silmeyi engelle
        protected_jobs = [
            "daily_embedding_sync",
            "weekly_full_reindex",
            "cleanup_expired_cache",
            "update_vector_stats",
        ]
        if job_id in protected_jobs:
            raise HTTPException(
                status_code=403, detail="Cannot remove protected system job"
            )

        job.remove()

        return {
            "message": f"Job {job_id} removed",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("remove_job_error", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def scheduler_health_check():
    """Scheduler sağlık kontrolü"""
    try:
        # Scheduler running mu?
        is_running = (
            offline_scheduler.scheduler.running
            if offline_scheduler.scheduler
            else False
        )

        # Redis bağlantısı?
        redis_healthy = False
        try:
            offline_scheduler.redis_client.ping()
            redis_healthy = True
        except:
            pass

        # Vector store healthy?
        from app.services.enhanced_embedding_service import enhanced_embedding_service

        vector_healthy = enhanced_embedding_service.vector_store_initialized

        # Overall health
        overall_healthy = is_running and redis_healthy and vector_healthy

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "components": {
                "scheduler": "healthy" if is_running else "unhealthy",
                "redis": "healthy" if redis_healthy else "unhealthy",
                "vector_store": "healthy" if vector_healthy else "unhealthy",
            },
            "scheduler_running": is_running,
            "jobs_count": len(offline_scheduler.scheduler.get_jobs())
            if offline_scheduler.scheduler
            else 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("scheduler_health_error", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
