import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Callable, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.core.database import get_async_session
from app.models.pdf_upload import PDFUpload, ProcessingStatus
from app.services.pdf_processing_service import pdf_processing_service
from app.services.vector_index_manager import vector_index_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class BackgroundScheduler:
    """Background task scheduler - Arka plan görevlerini yönetir"""
    
    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}
        self.scheduled_tasks: Dict[str, Dict[str, Any]] = {}
        self.is_running = False
    
    async def start(self):
        """Scheduler'ı başlat"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Background scheduler started")
        
        # Temel görevleri başlat
        await self._start_core_tasks()
    
    async def stop(self):
        """Scheduler'ı durdur"""
        self.is_running = False
        
        # Tüm görevleri iptal et
        for task_id, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Background scheduler stopped")
    
    async def _start_core_tasks(self):
        """Temel görevleri başlat"""
        
        # Vektör indekslerini oluştur
        await vector_index_manager.create_vector_indexes()

        # PDF işleme görevleri
        self.tasks["pdf_processor"] = asyncio.create_task(
            self._pdf_processing_worker()
        )
        
        # Veritabanı temizleme görevleri
        self.tasks["db_cleanup"] = asyncio.create_task(
            self._database_cleanup_worker()
        )
        
        # Sağlık kontrolü görevleri
        self.tasks["health_check"] = asyncio.create_task(
            self._health_check_worker()
        )
        
        logger.info("Core background tasks started")
    
    async def _pdf_processing_worker(self):
        """PDF işleme worker'ı"""
        
        while self.is_running:
            try:
                # Bekleyen PDF'leri işle
                await self._process_pending_pdfs()
                
                # 30 saniye bekle
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"PDF processing worker error: {e}")
                await asyncio.sleep(60)  # Hata durumunda daha uzun bekle
    
    async def _process_pending_pdfs(self):
        """Bekleyen PDF'leri işle"""
        
        async for db in get_async_session():
            try:
                # Pending durumundaki PDF'leri bul
                result = await db.execute(
                    select(PDFUpload).where(
                        PDFUpload.processing_status == ProcessingStatus.PENDING
                    ).limit(5)  # Aynı anda maksimum 5 PDF işle
                )
                pending_uploads = result.scalars().all()
                
                for upload in pending_uploads:
                    try:
                        # PDF işleme başlat
                        await pdf_processing_service.process_pdf_upload(
                            db, str(upload.id), upload.uploaded_by
                        )
                        
                        logger.info(f"PDF processed successfully: {upload.id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process PDF {upload.id}: {e}")
                        
                        # Status'u failed olarak güncelle
                        upload.processing_status = ProcessingStatus.FAILED
                        upload.processing_metadata = {
                            "error": str(e),
                            "failed_at": datetime.utcnow().isoformat()
                        }
                        await db.commit()
                
                break  # Session'ı kapat
                
            except Exception as e:
                logger.error(f"Database error in PDF processing: {e}")
                break
    
    async def _database_cleanup_worker(self):
        """Veritabanı temizleme worker'ı"""
        
        while self.is_running:
            try:
                # Günlük temizlik görevleri
                await self._cleanup_old_data()
                
                # 1 saat bekle
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Database cleanup worker error: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_data(self):
        """Eski verileri temizle"""
        
        async for db in get_async_session():
            try:
                # 30 günden eski failed upload'ları temizle
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                
                result = await db.execute(
                    select(PDFUpload).where(
                        and_(
                            PDFUpload.processing_status == ProcessingStatus.FAILED,
                            PDFUpload.created_at < cutoff_date
                        )
                    )
                )
                old_failed_uploads = result.scalars().all()
                
                for upload in old_failed_uploads:
                    await db.delete(upload)
                
                await db.commit()
                
                if old_failed_uploads:
                    logger.info(f"Cleaned up {len(old_failed_uploads)} old failed uploads")
                
                break
                
            except Exception as e:
                logger.error(f"Database cleanup error: {e}")
                break
    
    async def _health_check_worker(self):
        """Sağlık kontrolü worker'ı"""
        
        while self.is_running:
            try:
                # Sistem sağlığını kontrol et
                await self._check_system_health()
                
                # 5 dakika bekle
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Health check worker error: {e}")
                await asyncio.sleep(300)
    
    async def _check_system_health(self):
        """Sistem sağlığını kontrol et"""
        
        async for db in get_async_session():
            try:
                # Processing durumundaki PDF'leri kontrol et
                result = await db.execute(
                    select(PDFUpload).where(
                        PDFUpload.processing_status == ProcessingStatus.PROCESSING
                    )
                )
                processing_uploads = result.scalars().all()
                
                # 1 saatten uzun süredir processing durumunda olanları kontrol et
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                stuck_uploads = [
                    upload for upload in processing_uploads
                    if upload.processing_started_at and upload.processing_started_at < cutoff_time
                ]
                
                for upload in stuck_uploads:
                    logger.warning(f"Stuck PDF upload detected: {upload.id}")
                    
                    # Status'u failed olarak güncelle
                    upload.processing_status = ProcessingStatus.FAILED
                    upload.processing_metadata = {
                        "error": "Processing timeout",
                        "failed_at": datetime.utcnow().isoformat(),
                        "stuck_duration": (datetime.utcnow() - upload.processing_started_at).total_seconds()
                    }
                
                await db.commit()
                
                if stuck_uploads:
                    logger.info(f"Reset {len(stuck_uploads)} stuck uploads")
                
                break
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                break
    
    async def schedule_task(
        self,
        task_id: str,
        task_func: Callable,
        delay_seconds: int = 0,
        repeat_interval: Optional[int] = None
    ):
        """Görev planla"""
        
        if task_id in self.scheduled_tasks:
            # Mevcut görevi iptal et
            existing_task = self.scheduled_tasks[task_id]["task"]
            if not existing_task.done():
                existing_task.cancel()
        
        async def scheduled_task_wrapper():
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
            
            while self.is_running:
                try:
                    await task_func()
                    
                    if repeat_interval is None:
                        break
                    
                    await asyncio.sleep(repeat_interval)
                    
                except Exception as e:
                    logger.error(f"Scheduled task {task_id} error: {e}")
                    if repeat_interval is None:
                        break
                    await asyncio.sleep(repeat_interval)
        
        task = asyncio.create_task(scheduled_task_wrapper())
        
        self.scheduled_tasks[task_id] = {
            "task": task,
            "func": task_func,
            "delay": delay_seconds,
            "repeat_interval": repeat_interval,
            "created_at": datetime.utcnow()
        }
        
        logger.info(f"Scheduled task: {task_id}")
    
    async def cancel_task(self, task_id: str):
        """Görevi iptal et"""
        
        if task_id in self.scheduled_tasks:
            task = self.scheduled_tasks[task_id]["task"]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            del self.scheduled_tasks[task_id]
            logger.info(f"Cancelled task: {task_id}")
    
    async def get_task_status(self) -> Dict[str, Any]:
        """Görev durumlarını al"""
        
        status = {
            "scheduler_running": self.is_running,
            "active_tasks": len(self.tasks),
            "scheduled_tasks": len(self.scheduled_tasks),
            "task_details": {}
        }
        
        # Core task durumları
        for task_id, task in self.tasks.items():
            status["task_details"][task_id] = {
                "type": "core",
                "running": not task.done(),
                "cancelled": task.cancelled(),
                "exception": str(task.exception()) if task.exception() else None
            }
        
        # Scheduled task durumları
        for task_id, task_info in self.scheduled_tasks.items():
            task = task_info["task"]
            status["task_details"][task_id] = {
                "type": "scheduled",
                "running": not task.done(),
                "cancelled": task.cancelled(),
                "exception": str(task.exception()) if task.exception() else None,
                "delay": task_info["delay"],
                "repeat_interval": task_info["repeat_interval"],
                "created_at": task_info["created_at"].isoformat()
            }
        
        return status


# Global scheduler instance
background_scheduler = BackgroundScheduler()
