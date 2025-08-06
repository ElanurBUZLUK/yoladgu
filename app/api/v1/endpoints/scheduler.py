"""
Scheduler Endpoints
Görev zamanlama ve yönetim endpointleri
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
from datetime import datetime, timedelta
from pydantic import BaseModel

router = APIRouter(prefix="/scheduler", tags=["scheduler"])


class TaskRequest(BaseModel):
    """Görev isteği"""
    task_name: str
    schedule: str  # cron format
    parameters: Dict[str, Any] = {}
    enabled: bool = True


class TaskResponse(BaseModel):
    """Görev yanıtı"""
    task_id: str
    task_name: str
    schedule: str
    status: str
    last_run: str
    next_run: str
    parameters: Dict[str, Any]


@router.get("/tasks")
async def get_scheduled_tasks() -> Dict[str, Any]:
    """Zamanlanmış görevleri getir"""
    try:
        # Mock data - gerçek uygulamada scheduler'dan alınır
        tasks = [
            {
                "task_id": "task_001",
                "task_name": "daily_metrics_collection",
                "schedule": "0 0 * * *",
                "status": "active",
                "last_run": (datetime.now() - timedelta(hours=12)).isoformat(),
                "next_run": (datetime.now() + timedelta(hours=12)).isoformat(),
                "parameters": {"collect_system_metrics": True}
            },
            {
                "task_id": "task_002", 
                "task_name": "model_retraining",
                "schedule": "0 2 * * 0",
                "status": "active",
                "last_run": (datetime.now() - timedelta(days=7)).isoformat(),
                "next_run": (datetime.now() + timedelta(days=1)).isoformat(),
                "parameters": {"model_type": "recommendation"}
            }
        ]
        
        return {
            "status": "success",
            "data": {
                "tasks": tasks,
                "total_tasks": len(tasks),
                "active_tasks": len([t for t in tasks if t["status"] == "active"])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tasks: {str(e)}")


@router.post("/tasks")
async def create_scheduled_task(task: TaskRequest) -> Dict[str, Any]:
    """Yeni görev oluştur"""
    try:
        # Mock task creation
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        new_task = {
            "task_id": task_id,
            "task_name": task.task_name,
            "schedule": task.schedule,
            "status": "active" if task.enabled else "inactive",
            "last_run": None,
            "next_run": (datetime.now() + timedelta(minutes=5)).isoformat(),
            "parameters": task.parameters
        }
        
        return {
            "status": "success",
            "data": new_task,
            "message": "Task created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")


@router.get("/tasks/{task_id}")
async def get_task_details(task_id: str) -> Dict[str, Any]:
    """Görev detaylarını getir"""
    try:
        # Mock task details
        task = {
            "task_id": task_id,
            "task_name": "daily_metrics_collection",
            "schedule": "0 0 * * *",
            "status": "active",
            "last_run": (datetime.now() - timedelta(hours=12)).isoformat(),
            "next_run": (datetime.now() + timedelta(hours=12)).isoformat(),
            "parameters": {"collect_system_metrics": True},
            "execution_history": [
                {
                    "run_id": "run_001",
                    "started_at": (datetime.now() - timedelta(hours=12)).isoformat(),
                    "completed_at": (datetime.now() - timedelta(hours=11, minutes=55)).isoformat(),
                    "status": "success",
                    "duration_seconds": 300
                }
            ]
        }
        
        return {
            "status": "success",
            "data": task
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task details: {str(e)}")


@router.put("/tasks/{task_id}")
async def update_task(task_id: str, task: TaskRequest) -> Dict[str, Any]:
    """Görev güncelle"""
    try:
        # Mock task update
        updated_task = {
            "task_id": task_id,
            "task_name": task.task_name,
            "schedule": task.schedule,
            "status": "active" if task.enabled else "inactive",
            "last_run": (datetime.now() - timedelta(hours=6)).isoformat(),
            "next_run": (datetime.now() + timedelta(hours=6)).isoformat(),
            "parameters": task.parameters
        }
        
        return {
            "status": "success",
            "data": updated_task,
            "message": "Task updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update task: {str(e)}")


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str) -> Dict[str, Any]:
    """Görev sil"""
    try:
        return {
            "status": "success",
            "message": f"Task {task_id} deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")


@router.post("/tasks/{task_id}/run")
async def run_task_now(task_id: str) -> Dict[str, Any]:
    """Görevi şimdi çalıştır"""
    try:
        # Mock task execution
        execution_result = {
            "task_id": task_id,
            "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "message": "Task execution started"
        }
        
        return {
            "status": "success",
            "data": execution_result,
            "message": "Task execution initiated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run task: {str(e)}")


@router.get("/tasks/{task_id}/history")
async def get_task_execution_history(task_id: str, limit: int = 10) -> Dict[str, Any]:
    """Görev çalıştırma geçmişini getir"""
    try:
        # Mock execution history
        history = [
            {
                "run_id": f"run_{i:03d}",
                "started_at": (datetime.now() - timedelta(hours=i*6)).isoformat(),
                "completed_at": (datetime.now() - timedelta(hours=i*6, minutes=5)).isoformat(),
                "status": "success" if i % 3 != 0 else "failed",
                "duration_seconds": 300 + (i * 10),
                "error_message": None if i % 3 != 0 else "Task execution failed"
            }
            for i in range(1, min(limit + 1, 11))
        ]
        
        return {
            "status": "success",
            "data": {
                "task_id": task_id,
                "execution_history": history,
                "total_executions": len(history),
                "successful_executions": len([h for h in history if h["status"] == "success"]),
                "failed_executions": len([h for h in history if h["status"] == "failed"])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get execution history: {str(e)}")


@router.get("/status")
async def get_scheduler_status() -> Dict[str, Any]:
    """Scheduler durumunu getir"""
    try:
        # Mock scheduler status
        status = {
            "scheduler_status": "running",
            "total_tasks": 5,
            "active_tasks": 4,
            "inactive_tasks": 1,
            "last_heartbeat": datetime.now().isoformat(),
            "uptime_seconds": 86400,  # 24 hours
            "next_execution": (datetime.now() + timedelta(minutes=15)).isoformat()
        }
        
        return {
            "status": "success",
            "data": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scheduler status: {str(e)}")


@router.get("/stats")
async def get_scheduler_stats() -> Dict[str, Any]:
    """Scheduler istatistiklerini getir"""
    try:
        # Mock scheduler stats
        stats = {
            "total_executions_today": 48,
            "successful_executions_today": 45,
            "failed_executions_today": 3,
            "average_execution_time_seconds": 245,
            "total_execution_time_today_hours": 3.2,
            "most_frequent_task": "daily_metrics_collection",
            "least_frequent_task": "model_retraining"
        }
        
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scheduler stats: {str(e)}") 