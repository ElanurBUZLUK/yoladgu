"""
Robust Process Management for Background Services
"""

import asyncio
import signal
import sys
from typing import Optional, Dict, Any, Callable
import structlog
from contextlib import asynccontextmanager
from datetime import datetime
import psutil
import os

logger = structlog.get_logger()

class ProcessManager:
    """Manages background processes with proper lifecycle management"""
    
    def __init__(self):
        self.processes: Dict[str, Dict[str, Any]] = {}
        self.shutdown_event = asyncio.Event()
        self.health_check_interval = 30  # seconds
        
    async def start_background_task(
        self, 
        name: str, 
        coro: Callable,
        restart_on_failure: bool = True,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ):
        """Start a background task with monitoring and auto-restart"""
        
        async def managed_task():
            retries = 0
            while not self.shutdown_event.is_set() and retries <= max_retries:
                try:
                    logger.info("starting_background_task", name=name, attempt=retries + 1)
                    
                    # Start the actual coroutine
                    task = asyncio.create_task(coro())
                    self.processes[name] = {
                        'task': task,
                        'started_at': datetime.utcnow(),
                        'retries': retries,
                        'status': 'running'
                    }
                    
                    # Wait for completion or shutdown
                    done, pending = await asyncio.wait(
                        [task, asyncio.create_task(self.shutdown_event.wait())],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel any pending tasks
                    for p in pending:
                        p.cancel()
                        try:
                            await p
                        except asyncio.CancelledError:
                            pass
                    
                    # Check if shutdown was requested
                    if self.shutdown_event.is_set():
                        logger.info("background_task_shutdown_requested", name=name)
                        break
                    
                    # Task completed/failed
                    if task in done:
                        try:
                            await task  # This will raise if the task failed
                            logger.info("background_task_completed", name=name)
                            break  # Successful completion
                        except Exception as e:
                            logger.error("background_task_failed", 
                                       name=name, 
                                       error=str(e),
                                       retries=retries)
                            
                            if not restart_on_failure:
                                break
                            
                            retries += 1
                            if retries <= max_retries:
                                logger.info("background_task_retrying", 
                                          name=name, 
                                          retry_in=retry_delay)
                                await asyncio.sleep(retry_delay)
                            else:
                                logger.error("background_task_max_retries_exceeded", name=name)
                                break
                
                except Exception as e:
                    logger.error("background_task_manager_error", 
                               name=name, 
                               error=str(e))
                    break
            
            # Mark as stopped
            if name in self.processes:
                self.processes[name]['status'] = 'stopped'
                self.processes[name]['stopped_at'] = datetime.utcnow()
        
        # Start the managed task
        manager_task = asyncio.create_task(managed_task())
        return manager_task
    
    async def stop_background_task(self, name: str, timeout: float = 10.0):
        """Stop a specific background task"""
        if name not in self.processes:
            logger.warning("background_task_not_found", name=name)
            return
        
        process_info = self.processes[name]
        task = process_info.get('task')
        
        if task and not task.done():
            logger.info("stopping_background_task", name=name)
            task.cancel()
            
            try:
                await asyncio.wait_for(task, timeout=timeout)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.warning("background_task_force_stopped", name=name)
        
        process_info['status'] = 'stopped'
        process_info['stopped_at'] = datetime.utcnow()
    
    async def stop_all_tasks(self, timeout: float = 30.0):
        """Stop all managed background tasks"""
        logger.info("stopping_all_background_tasks")
        self.shutdown_event.set()
        
        # Cancel all tasks
        tasks_to_wait = []
        for name, process_info in self.processes.items():
            task = process_info.get('task')
            if task and not task.done():
                task.cancel()
                tasks_to_wait.append(task)
        
        # Wait for all tasks to complete
        if tasks_to_wait:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_wait, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("some_background_tasks_did_not_stop_gracefully")
        
        logger.info("all_background_tasks_stopped")
    
    def get_process_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of managed processes"""
        if name:
            return self.processes.get(name, {'status': 'not_found'})
        return self.processes
    
    async def health_check_loop(self):
        """Continuous health monitoring of managed processes"""
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                
                for name, process_info in self.processes.items():
                    task = process_info.get('task')
                    if task:
                        if task.done():
                            if task.exception():
                                logger.error("background_task_health_check_failed", 
                                           name=name, 
                                           error=str(task.exception()))
                            process_info['status'] = 'completed'
                        else:
                            process_info['status'] = 'running'
                            process_info['last_health_check'] = current_time
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error("health_check_loop_error", error=str(e))
                await asyncio.sleep(5)  # Short delay before retry

# Global process manager instance
process_manager = ProcessManager()

@asynccontextmanager
async def managed_background_service(name: str, coro: Callable, **kwargs):
    """Context manager for background services"""
    try:
        task = await process_manager.start_background_task(name, coro, **kwargs)
        yield task
    finally:
        await process_manager.stop_background_task(name)

# Signal handlers for graceful shutdown
def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info("received_shutdown_signal", signal=signum)
        asyncio.create_task(process_manager.stop_all_tasks())
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

# Docker health check helper
def health_check_endpoint():
    """Simple health check for Docker/Kubernetes"""
    try:
        status = process_manager.get_process_status()
        healthy_processes = sum(1 for p in status.values() if p.get('status') == 'running')
        total_processes = len(status)
        
        if total_processes == 0:
            return {"status": "no_processes", "healthy": True}
        
        health_ratio = healthy_processes / total_processes
        
        return {
            "status": "healthy" if health_ratio >= 0.8 else "degraded",
            "healthy": health_ratio >= 0.8,
            "processes": {
                "total": total_processes,
                "healthy": healthy_processes,
                "ratio": health_ratio
            },
            "details": status
        }
    except Exception as e:
        logger.error("health_check_error", error=str(e))
        return {"status": "unhealthy", "healthy": False, "error": str(e)}