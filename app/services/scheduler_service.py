"""
Scheduler Service
Görev zamanlama ve yönetim servisi
"""

import structlog
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = structlog.get_logger()

class OfflineScheduler:
    """Mock offline scheduler (gerçek uygulamada Celery, APScheduler, vs. ile entegre edilir)"""

    def __init__(self):
        self.tasks = []

    async def add_task(self, task_name: str, schedule: str, parameters: Optional[Dict[str, Any]] = None):
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task = {
            "task_id": task_id,
            "task_name": task_name,
            "schedule": schedule,
            "parameters": parameters or {},
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        self.tasks.append(task)
        logger.info("task_added", task=task)
        return task

    async def get_tasks(self) -> List[Dict[str, Any]]:
        return self.tasks

    async def remove_task(self, task_id: str) -> bool:
        before = len(self.tasks)
        self.tasks = [t for t in self.tasks if t["task_id"] != task_id]
        after = len(self.tasks)
        logger.info("task_removed", task_id=task_id)
        return before != after

    async def get_status(self) -> Dict[str, Any]:
        return {
            "scheduler_status": "running",
            "total_tasks": len(self.tasks),
            "active_tasks": len([t for t in self.tasks if t["status"] == "active"]),
            "last_heartbeat": datetime.now().isoformat(),
            "uptime_seconds": 86400
        }

offline_scheduler = OfflineScheduler()