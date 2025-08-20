from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List
from pydantic import BaseModel

from app.core.database import get_async_session
from app.services.background_scheduler import background_scheduler
from app.middleware.auth import get_current_admin
from app.models.user import User

router = APIRouter(prefix="/api/v1/scheduler", tags=["scheduler"])


class TaskScheduleRequest(BaseModel):
    task_id: str
    delay_seconds: int = 0
    repeat_interval: int = None
    task_type: str = "custom"


class TaskStatusResponse(BaseModel):
    scheduler_running: bool
    active_tasks: int
    scheduled_tasks: int
    task_details: Dict[str, Any]


@router.get("/status", response_model=TaskStatusResponse)
async def get_scheduler_status(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_session)
):
    """Get background scheduler status"""
    
    try:
        status = await background_scheduler.get_task_status()
        return TaskStatusResponse(**status)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get scheduler status: {str(e)}"
        )


@router.post("/start")
async def start_scheduler(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_session)
):
    """Start background scheduler"""
    
    try:
        await background_scheduler.start()
        
        return {
            "success": True,
            "message": "Background scheduler started successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start scheduler: {str(e)}"
        )


@router.post("/stop")
async def stop_scheduler(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_session)
):
    """Stop background scheduler"""
    
    try:
        await background_scheduler.stop()
        
        return {
            "success": True,
            "message": "Background scheduler stopped successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop scheduler: {str(e)}"
        )


@router.post("/tasks/schedule")
async def schedule_task(
    request: TaskScheduleRequest,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_session)
):
    """Schedule a new background task"""
    
    try:
        # Task fonksiyonunu belirle
        task_func = await _get_task_function(request.task_type)
        
        await background_scheduler.schedule_task(
            task_id=request.task_id,
            task_func=task_func,
            delay_seconds=request.delay_seconds,
            repeat_interval=request.repeat_interval
        )
        
        return {
            "success": True,
            "message": f"Task {request.task_id} scheduled successfully",
            "task_id": request.task_id,
            "delay_seconds": request.delay_seconds,
            "repeat_interval": request.repeat_interval
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to schedule task: {str(e)}"
        )


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_session)
):
    """Cancel a scheduled task"""
    
    try:
        await background_scheduler.cancel_task(task_id)
        
        return {
            "success": True,
            "message": f"Task {task_id} cancelled successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )


@router.get("/tasks")
async def list_scheduled_tasks(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_session)
):
    """List all scheduled tasks"""
    
    try:
        status = await background_scheduler.get_task_status()
        
        scheduled_tasks = []
        for task_id, task_info in status["task_details"].items():
            if task_info["type"] == "scheduled":
                scheduled_tasks.append({
                    "task_id": task_id,
                    **task_info
                })
        
        return {
            "scheduled_tasks": scheduled_tasks,
            "total_count": len(scheduled_tasks)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tasks: {str(e)}"
        )


@router.post("/tasks/health-check")
async def trigger_health_check(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_session)
):
    """Manually trigger health check"""
    
    try:
        # Health check görevini hemen çalıştır
        await background_scheduler.schedule_task(
            task_id="manual_health_check",
            task_func=_manual_health_check,
            delay_seconds=0,
            repeat_interval=None
        )
        
        return {
            "success": True,
            "message": "Health check triggered successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger health check: {str(e)}"
        )


@router.post("/tasks/cleanup")
async def trigger_cleanup(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_session)
):
    """Manually trigger database cleanup"""
    
    try:
        # Cleanup görevini hemen çalıştır
        await background_scheduler.schedule_task(
            task_id="manual_cleanup",
            task_func=_manual_cleanup,
            delay_seconds=0,
            repeat_interval=None
        )
        
        return {
            "success": True,
            "message": "Database cleanup triggered successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger cleanup: {str(e)}"
        )


async def _get_task_function(task_type: str):
    """Task tipine göre fonksiyon döndür"""
    
    task_functions = {
        "health_check": _manual_health_check,
        "cleanup": _manual_cleanup,
        "pdf_processing": _manual_pdf_processing,
        "custom": _custom_task
    }
    
    return task_functions.get(task_type, _custom_task)


async def _manual_health_check():
    """Manuel sağlık kontrolü"""
    from app.services.background_scheduler import background_scheduler
    await background_scheduler._check_system_health()


async def _manual_cleanup():
    """Manuel veritabanı temizleme"""
    from app.services.background_scheduler import background_scheduler
    await background_scheduler._cleanup_old_data()


async def _manual_pdf_processing():
    """Manuel PDF işleme"""
    from app.services.background_scheduler import background_scheduler
    await background_scheduler._process_pending_pdfs()


async def _custom_task():
    """Özel görev"""
    print("Custom task executed")


@router.get("/health")
async def scheduler_health_check():
    """Scheduler health check endpoint"""
    return {
        "status": "healthy",
        "scheduler_running": background_scheduler.is_running,
        "active_tasks": len(background_scheduler.tasks),
        "scheduled_tasks": len(background_scheduler.scheduled_tasks)
    }
