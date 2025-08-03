"""
Offline Batch Scheduler Service
Embedding güncelleme, indexleme ve bakım işlemleri için cron job scheduler
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED
import redis
import json
from dataclasses import dataclass, asdict

from app.core.config import settings
from app.services.enhanced_embedding_service import enhanced_embedding_service
from app.services.vector_store_service import vector_store_service
from app.db.database import SessionLocal
from app.db.models import Question

logger = structlog.get_logger()

@dataclass
class SchedulerTask:
    """Scheduler görevi metadata'sı"""
    task_id: str
    task_type: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict] = None
    progress: float = 0.0
    estimated_duration: Optional[int] = None  # seconds

class OfflineBatchScheduler:
    """
    Offline batch işlemler için scheduler service
    """
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler(
            timezone='UTC',
            job_defaults={
                'coalesce': False,
                'max_instances': 1,
                'misfire_grace_time': 300  # 5 minutes
            }
        )
        
        self.redis_client = redis.from_url(settings.redis_url)
        self.task_prefix = "scheduler:task"
        self.lock_prefix = "scheduler:lock"
        self.stats_key = "scheduler:stats"
        
        # Task type configuration
        self.task_configs = {
            'daily_embedding_sync': {
                'max_duration': 3600,  # 1 hour max
                'batch_size': 100,
                'priority': 'high'
            },
            'weekly_full_reindex': {
                'max_duration': 7200,  # 2 hours max
                'batch_size': 200,
                'priority': 'medium'
            },
            'model_migration': {
                'max_duration': 10800,  # 3 hours max
                'batch_size': 50,
                'priority': 'critical'
            },
            'cleanup_expired_cache': {
                'max_duration': 600,  # 10 minutes max
                'batch_size': 1000,
                'priority': 'low'
            }
        }
        
        # Event listeners
        self.scheduler.add_listener(self._job_listener, 
                                  EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED)
    
    async def initialize(self):
        """Scheduler'ı başlat ve cron job'ları ekle"""
        try:
            # Vector store'u initialize et
            await enhanced_embedding_service.initialize_vector_store()
            
            # Default cron jobs'ları ekle
            await self._add_default_cron_jobs()
            
            # Scheduler'ı başlat
            self.scheduler.start()
            
            logger.info("scheduler_initialized", 
                       timezone=self.scheduler.timezone,
                       jobs_count=len(self.scheduler.get_jobs()))
            
        except Exception as e:
            logger.error("scheduler_init_error", error=str(e))
            raise
    
    async def _add_default_cron_jobs(self):
        """Default cron job'ları ekle"""
        
        # 1. Her gece 02:00'da embedding sync
        self.scheduler.add_job(
            func=self.daily_embedding_sync,
            trigger=CronTrigger(hour=2, minute=0),
            id='daily_embedding_sync',
            name='Daily Embedding Synchronization',
            replace_existing=True,
            max_instances=1
        )
        
        # 2. Her Pazar 03:00'da full reindex
        self.scheduler.add_job(
            func=self.weekly_full_reindex,
            trigger=CronTrigger(day_of_week=6, hour=3, minute=0),  # Sunday
            id='weekly_full_reindex',
            name='Weekly Full Reindex',
            replace_existing=True,
            max_instances=1
        )
        
        # 3. Her 6 saatte cache cleanup
        self.scheduler.add_job(
            func=self.cleanup_expired_cache,
            trigger=IntervalTrigger(hours=6),
            id='cleanup_expired_cache',
            name='Cleanup Expired Cache',
            replace_existing=True,
            max_instances=1
        )
        
        # 4. Vector store istatistik güncellemesi (her saat)
        self.scheduler.add_job(
            func=self.update_vector_stats,
            trigger=IntervalTrigger(hours=1),
            id='update_vector_stats',
            name='Update Vector Store Statistics',
            replace_existing=True,
            max_instances=1
        )
        
        logger.info("default_cron_jobs_added", jobs_count=4)
    
    async def daily_embedding_sync(self):
        """Günlük embedding senkronizasyonu"""
        task_id = f"daily_sync_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        task = SchedulerTask(
            task_id=task_id,
            task_type='daily_embedding_sync',
            status='running',
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow()
        )
        
        try:
            await self._save_task(task)
            
            # Lock al
            lock_key = f"{self.lock_prefix}:daily_sync"
            if not await self._acquire_lock(lock_key, timeout=3600):
                raise Exception("Could not acquire lock for daily sync")
            
            try:
                logger.info("daily_embedding_sync_started", task_id=task_id)
                
                # Son 24 saatte değişen soruları bul
                db = SessionLocal()
                yesterday = datetime.utcnow() - timedelta(days=1)
                
                # Yeni/güncellenmiş sorular
                new_questions = db.query(Question).filter(
                    Question.created_at >= yesterday
                ).all()
                
                updated_questions = db.query(Question).filter(
                    Question.updated_at >= yesterday,
                    Question.created_at < yesterday
                ).all()
                
                db.close()
                
                # Progress tracking
                total_questions = len(new_questions) + len(updated_questions)
                processed = 0
                
                # Yeni soruları işle
                for question in new_questions:
                    success = await enhanced_embedding_service.store_question_embedding(
                        question_id=question.id,
                        question_text=question.question_text,
                        subject_id=question.subject_id,
                        topic_id=question.topic_id,
                        difficulty_level=question.difficulty_level
                    )
                    
                    processed += 1
                    task.progress = (processed / total_questions) * 100 if total_questions > 0 else 100
                    await self._save_task(task)
                    
                    if not success:
                        logger.warning("failed_to_store_embedding", question_id=question.id)
                
                # Güncellenmiş soruları işle
                for question in updated_questions:
                    success = await enhanced_embedding_service.store_question_embedding(
                        question_id=question.id,
                        question_text=question.question_text,
                        subject_id=question.subject_id,
                        topic_id=question.topic_id,
                        difficulty_level=question.difficulty_level
                    )
                    
                    processed += 1
                    task.progress = (processed / total_questions) * 100 if total_questions > 0 else 100
                    await self._save_task(task)
                    
                    if not success:
                        logger.warning("failed_to_update_embedding", question_id=question.id)
                
                # Başarılı tamamlandı
                task.status = 'completed'
                task.completed_at = datetime.utcnow()
                task.result = {
                    'new_questions_processed': len(new_questions),
                    'updated_questions_processed': len(updated_questions),
                    'total_processed': processed,
                    'duration_seconds': (task.completed_at - task.started_at).total_seconds()
                }
                
                await self._save_task(task)
                await self._update_stats('daily_embedding_sync', 'success')
                
                logger.info("daily_embedding_sync_completed", 
                           task_id=task_id,
                           **task.result)
                
            finally:
                await self._release_lock(lock_key)
                
        except Exception as e:
            task.status = 'failed'
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            await self._save_task(task)
            await self._update_stats('daily_embedding_sync', 'error')
            
            logger.error("daily_embedding_sync_failed", 
                        task_id=task_id, 
                        error=str(e))
            raise
    
    async def weekly_full_reindex(self):
        """Haftalık tam yeniden indexleme"""
        task_id = f"weekly_reindex_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        task = SchedulerTask(
            task_id=task_id,
            task_type='weekly_full_reindex',
            status='running',
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow()
        )
        
        try:
            await self._save_task(task)
            
            # Lock al
            lock_key = f"{self.lock_prefix}:weekly_reindex"
            if not await self._acquire_lock(lock_key, timeout=7200):  # 2 hours
                raise Exception("Could not acquire lock for weekly reindex")
            
            try:
                logger.info("weekly_full_reindex_started", task_id=task_id)
                
                # Tüm embedding'leri yeniden hesapla
                stats = await enhanced_embedding_service.batch_update_embeddings(
                    batch_size=self.task_configs['weekly_full_reindex']['batch_size'],
                    force_recompute=True
                )
                
                # Vector store stats güncelle
                vector_stats = await enhanced_embedding_service.get_vector_store_stats()
                
                task.status = 'completed'
                task.completed_at = datetime.utcnow()
                task.progress = 100.0
                task.result = {
                    **stats,
                    'vector_stats': vector_stats,
                    'duration_seconds': (task.completed_at - task.started_at).total_seconds()
                }
                
                await self._save_task(task)
                await self._update_stats('weekly_full_reindex', 'success')
                
                logger.info("weekly_full_reindex_completed", 
                           task_id=task_id,
                           **task.result)
                
            finally:
                await self._release_lock(lock_key)
                
        except Exception as e:
            task.status = 'failed'
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            await self._save_task(task)
            await self._update_stats('weekly_full_reindex', 'error')
            
            logger.error("weekly_full_reindex_failed", 
                        task_id=task_id, 
                        error=str(e))
            raise
    
    async def cleanup_expired_cache(self):
        """Süresi dolmuş cache'leri temizle"""
        task_id = f"cache_cleanup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info("cache_cleanup_started", task_id=task_id)
            
            # Embedding cache temizleme
            success = enhanced_embedding_service.clear_cache()
            
            # Redis'te eski task kayıtlarını temizle (30 gün)
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            cleaned_tasks = await self._cleanup_old_tasks(cutoff_date)
            
            result = {
                'cache_cleared': success,
                'old_tasks_cleaned': cleaned_tasks,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self._update_stats('cleanup_expired_cache', 'success')
            
            logger.info("cache_cleanup_completed", 
                       task_id=task_id,
                       **result)
            
        except Exception as e:
            await self._update_stats('cleanup_expired_cache', 'error')
            logger.error("cache_cleanup_failed", 
                        task_id=task_id, 
                        error=str(e))
    
    async def update_vector_stats(self):
        """Vector store istatistiklerini güncelle"""
        try:
            stats = await enhanced_embedding_service.get_vector_store_stats()
            
            # Redis'e kaydet
            stats_key = f"{self.stats_key}:vector:{datetime.utcnow().strftime('%Y%m%d_%H')}"
            self.redis_client.setex(stats_key, 86400, json.dumps(stats))  # 24 saat
            
            logger.debug("vector_stats_updated", stats=stats)
            
        except Exception as e:
            logger.error("vector_stats_update_failed", error=str(e))
    
    async def trigger_model_migration(self, new_model_key: str, force: bool = False):
        """Model değişikliği sonrası migration tetikle"""
        task_id = f"model_migration_{new_model_key}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        task = SchedulerTask(
            task_id=task_id,
            task_type='model_migration',
            status='running',
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow()
        )
        
        try:
            await self._save_task(task)
            
            # Immediate job scheduling
            self.scheduler.add_job(
                func=self._execute_model_migration,
                trigger=DateTrigger(run_date=datetime.utcnow() + timedelta(seconds=10)),
                args=[task_id, new_model_key, force],
                id=f'model_migration_{new_model_key}',
                name=f'Model Migration to {new_model_key}',
                replace_existing=True,
                max_instances=1
            )
            
            logger.info("model_migration_scheduled", 
                       task_id=task_id,
                       new_model=new_model_key,
                       force=force)
            
            return task_id
            
        except Exception as e:
            task.status = 'failed'
            task.error_message = str(e)
            await self._save_task(task)
            
            logger.error("model_migration_schedule_failed", 
                        task_id=task_id,
                        error=str(e))
            raise
    
    async def _execute_model_migration(self, task_id: str, new_model_key: str, force: bool):
        """Model migration'ı çalıştır"""
        task = await self._get_task(task_id)
        
        try:
            # Model switch
            success = enhanced_embedding_service.switch_model(new_model_key)
            if not success:
                raise Exception(f"Failed to switch to model: {new_model_key}")
            
            # Full reindex with new model
            stats = await enhanced_embedding_service.batch_update_embeddings(
                batch_size=self.task_configs['model_migration']['batch_size'],
                force_recompute=True
            )
            
            task.status = 'completed'
            task.completed_at = datetime.utcnow()
            task.progress = 100.0
            task.result = {
                'new_model': new_model_key,
                'migration_stats': stats,
                'duration_seconds': (task.completed_at - task.started_at).total_seconds()
            }
            
            await self._save_task(task)
            await self._update_stats('model_migration', 'success')
            
            logger.info("model_migration_completed", 
                       task_id=task_id,
                       new_model=new_model_key,
                       **stats)
            
        except Exception as e:
            task.status = 'failed'
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            await self._save_task(task)
            await self._update_stats('model_migration', 'error')
            
            logger.error("model_migration_failed", 
                        task_id=task_id,
                        new_model=new_model_key,
                        error=str(e))
    
    async def _acquire_lock(self, lock_key: str, timeout: int = 3600) -> bool:
        """Distributed lock al"""
        try:
            result = self.redis_client.set(lock_key, "locked", nx=True, ex=timeout)
            return result is True
        except Exception as e:
            logger.error("lock_acquire_error", lock_key=lock_key, error=str(e))
            return False
    
    async def _release_lock(self, lock_key: str) -> bool:
        """Distributed lock bırak"""
        try:
            result = self.redis_client.delete(lock_key)
            return result > 0
        except Exception as e:
            logger.error("lock_release_error", lock_key=lock_key, error=str(e))
            return False
    
    async def _save_task(self, task: SchedulerTask):
        """Task durumunu Redis'e kaydet"""
        try:
            task_key = f"{self.task_prefix}:{task.task_id}"
            task_data = asdict(task)
            
            # Datetime objeleri string'e çevir
            for key, value in task_data.items():
                if isinstance(value, datetime):
                    task_data[key] = value.isoformat()
            
            self.redis_client.setex(task_key, 86400 * 7, json.dumps(task_data))  # 7 gün
            
        except Exception as e:
            logger.error("task_save_error", task_id=task.task_id, error=str(e))
    
    async def _get_task(self, task_id: str) -> Optional[SchedulerTask]:
        """Task durumunu Redis'den al"""
        try:
            task_key = f"{self.task_prefix}:{task_id}"
            task_data = self.redis_client.get(task_key)
            
            if task_data:
                data = json.loads(task_data)
                
                # String'leri datetime'a çevir
                for key in ['created_at', 'started_at', 'completed_at']:
                    if data.get(key):
                        data[key] = datetime.fromisoformat(data[key])
                
                return SchedulerTask(**data)
            
            return None
            
        except Exception as e:
            logger.error("task_get_error", task_id=task_id, error=str(e))
            return None
    
    async def _cleanup_old_tasks(self, cutoff_date: datetime) -> int:
        """Eski task kayıtlarını temizle"""
        try:
            pattern = f"{self.task_prefix}:*"
            keys = self.redis_client.keys(pattern)
            cleaned = 0
            
            for key in keys:
                task_data = self.redis_client.get(key)
                if task_data:
                    data = json.loads(task_data)
                    created_at = datetime.fromisoformat(data.get('created_at', ''))
                    
                    if created_at < cutoff_date:
                        self.redis_client.delete(key)
                        cleaned += 1
            
            return cleaned
            
        except Exception as e:
            logger.error("cleanup_old_tasks_error", error=str(e))
            return 0
    
    async def _update_stats(self, task_type: str, status: str):
        """Scheduler istatistiklerini güncelle"""
        try:
            stats_key = f"{self.stats_key}:summary"
            current_stats = self.redis_client.get(stats_key)
            
            if current_stats:
                stats = json.loads(current_stats)
            else:
                stats = {}
            
            # İstatistik güncelle
            if task_type not in stats:
                stats[task_type] = {'success': 0, 'error': 0, 'total': 0}
            
            stats[task_type][status] += 1
            stats[task_type]['total'] += 1
            stats['last_updated'] = datetime.utcnow().isoformat()
            
            self.redis_client.setex(stats_key, 86400 * 30, json.dumps(stats))  # 30 gün
            
        except Exception as e:
            logger.error("stats_update_error", task_type=task_type, error=str(e))
    
    def _job_listener(self, event):
        """Job event listener"""
        if event.exception:
            logger.error("scheduled_job_error", 
                        job_id=event.job_id,
                        exception=str(event.exception))
        else:
            logger.debug("scheduled_job_completed", 
                        job_id=event.job_id,
                        scheduled_run_time=event.scheduled_run_time)
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Scheduler durumu"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        
        return {
            'running': self.scheduler.running,
            'jobs_count': len(jobs),
            'jobs': jobs,
            'timezone': str(self.scheduler.timezone)
        }
    
    async def shutdown(self):
        """Scheduler'ı temiz bir şekilde kapat"""
        try:
            self.scheduler.shutdown(wait=True)
            logger.info("scheduler_shutdown_completed")
        except Exception as e:
            logger.error("scheduler_shutdown_error", error=str(e))

# Global instance
offline_scheduler = OfflineBatchScheduler()