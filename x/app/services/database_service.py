import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError

from app.core.config import settings
from app.core.database import get_async_session

logger = logging.getLogger(__name__)


class DatabaseService:
    """Database service layer - Veritabanı işlemlerini yönetir"""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        self.connection_pool = None
        self.is_initialized = False
        self.health_check_interval = 300  # 5 dakika
        self.max_retries = 3
        self.retry_delay = 1  # saniye
    
    async def initialize(self):
        """Database service'i başlat"""
        if self.is_initialized:
            return
        
        try:
            # Engine oluştur
            self.engine = create_async_engine(
                settings.database_url,
                echo=settings.debug,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600,  # 1 saat
                pool_timeout=30,
                connect_args={
                    "server_settings": {
                        "application_name": "adaptive_learning_system"
                    }
                }
            )
            
            # Session factory oluştur
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            
            # Bağlantı havuzunu test et
            await self._test_connection()
            
            self.is_initialized = True
            logger.info("Database service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database service: {e}")
            raise
    
    async def shutdown(self):
        """Database service'i kapat"""
        if self.engine:
            await self.engine.dispose()
            self.is_initialized = False
            logger.info("Database service shutdown")
    
    @asynccontextmanager
    async def get_session(self):
        """Database session context manager"""
        if not self.is_initialized:
            await self.initialize()
        
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def execute_transaction(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Transaction içinde operasyon çalıştır"""
        
        for attempt in range(self.max_retries):
            try:
                async with self.get_session() as session:
                    result = await operation(session, *args, **kwargs)
                    return result
                    
            except (DisconnectionError, SQLAlchemyError) as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Database operation failed after {self.max_retries} attempts: {e}")
                    raise
                
                logger.warning(f"Database operation failed, retrying... ({attempt + 1}/{self.max_retries})")
                await asyncio.sleep(self.retry_delay * (attempt + 1))
    
    async def execute_batch_operations(
        self,
        operations: list[Callable],
        batch_size: int = 100
    ) -> list[Any]:
        """Toplu operasyonları çalıştır"""
        
        results = []
        
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            
            try:
                async with self.get_session() as session:
                    batch_results = []
                    for operation in batch:
                        result = await operation(session)
                        batch_results.append(result)
                    
                    results.extend(batch_results)
                    
            except Exception as e:
                logger.error(f"Batch operation failed at batch {i//batch_size + 1}: {e}")
                raise
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Database sağlık kontrolü"""
        
        if not self.is_initialized:
            return {
                "status": "not_initialized",
                "message": "Database service not initialized"
            }
        
        try:
            async with self.get_session() as session:
                # Basit sorgu testi
                result = await session.execute(text("SELECT 1"))
                result.scalar()
                
                # Bağlantı havuzu durumu
                pool_status = self.engine.pool.status()
                
                return {
                    "status": "healthy",
                    "pool_size": pool_status.size,
                    "checked_in": pool_status.checkedin,
                    "checked_out": pool_status.checkedout,
                    "overflow": pool_status.overflow,
                    "invalid": pool_status.invalid
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def optimize_queries(self):
        """Query optimizasyonu"""
        
        try:
            async with self.get_session() as session:
                # İstatistikleri güncelle
                await session.execute(text("ANALYZE"))
                
                # İndeksleri yeniden oluştur
                await session.execute(text("REINDEX DATABASE adaptive_learning"))
                
                logger.info("Database optimization completed")
                
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            raise
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Veritabanı istatistiklerini al"""
        
        try:
            async with self.get_session() as session:
                # Tablo boyutları
                result = await session.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation
                    FROM pg_stats 
                    WHERE schemaname = 'public'
                    ORDER BY tablename, attname
                """))
                
                stats = result.fetchall()
                
                # Tablo sayıları
                result = await session.execute(text("""
                    SELECT 
                        table_name,
                        (SELECT count(*) FROM information_schema.columns 
                         WHERE table_name = t.table_name) as column_count
                    FROM information_schema.tables t
                    WHERE table_schema = 'public'
                """))
                
                table_info = result.fetchall()
                
                return {
                    "table_stats": [
                        {
                            "table": row.table_name,
                            "columns": row.column_count
                        }
                        for row in table_info
                    ],
                    "column_stats": [
                        {
                            "table": row.tablename,
                            "column": row.attname,
                            "distinct_values": row.n_distinct,
                            "correlation": row.correlation
                        }
                        for row in stats
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}
    
    async def backup_database(self, backup_path: str) -> bool:
        """Veritabanı yedeği al"""
        
        try:
            import subprocess
            import os
            
            # pg_dump komutu
            cmd = [
                "pg_dump",
                "-h", "localhost",
                "-U", "postgres",
                "-d", "adaptive_learning",
                "-f", backup_path,
                "--format=custom",
                "--verbose"
            ]
            
            # Environment variables
            env = os.environ.copy()
            env["PGPASSWORD"] = "password"  # Güvenlik için environment variable kullan
            
            # Backup işlemini çalıştır
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Database backup completed: {backup_path}")
                return True
            else:
                logger.error(f"Database backup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    async def restore_database(self, backup_path: str) -> bool:
        """Veritabanı yedeğini geri yükle"""
        
        try:
            import subprocess
            import os
            
            # pg_restore komutu
            cmd = [
                "pg_restore",
                "-h", "localhost",
                "-U", "postgres",
                "-d", "adaptive_learning",
                "--clean",
                "--if-exists",
                "--verbose",
                backup_path
            ]
            
            # Environment variables
            env = os.environ.copy()
            env["PGPASSWORD"] = "password"
            
            # Restore işlemini çalıştır
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Database restore completed: {backup_path}")
                return True
            else:
                logger.error(f"Database restore failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
    
    async def _test_connection(self):
        """Bağlantıyı test et"""
        
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT version()"))
                version = result.scalar()
                logger.info(f"Database connection test successful: {version}")
                
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    async def get_connection_pool_status(self) -> Dict[str, Any]:
        """Bağlantı havuzu durumunu al"""
        
        if not self.engine:
            return {"error": "Engine not initialized"}
        
        pool = self.engine.pool
        status = pool.status()
        
        return {
            "pool_size": status.size,
            "checked_in": status.checkedin,
            "checked_out": status.checkedout,
            "overflow": status.overflow,
            "invalid": status.invalid,
            "total_connections": status.size + status.overflow
        }
    
    async def clear_connection_pool(self):
        """Bağlantı havuzunu temizle"""
        
        if self.engine:
            self.engine.pool.dispose()
            logger.info("Connection pool cleared")


# Global database service instance
database_service = DatabaseService()
