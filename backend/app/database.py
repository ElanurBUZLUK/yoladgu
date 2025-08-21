import asyncio
import logging
from typing import AsyncGenerator, Optional, Dict, Any
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    create_async_engine, 
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.pool import QueuePool
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

# Create declarative base for models
Base = declarative_base()

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection manager with connection pooling"""
    
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.async_session_maker: Optional[async_sessionmaker] = None
        self.pool_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "overflow_connections": 0,
            "checked_out_connections": 0,
            "checked_in_connections": 0
        }
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        self.health_check_interval = 300  # 5 minutes
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize database connection pool"""
        
        try:
            logger.info("🚀 Initializing database connection pool...")
            
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                settings.database_url,
                echo=settings.database_echo,
                poolclass=QueuePool,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_pre_ping=True,  # Enable connection health checks
                pool_recycle=settings.database_pool_recycle,
                pool_timeout=settings.database_pool_timeout,
                pool_reset_on_return='commit',  # Reset connections on return
                connect_args={
                    "server_settings": {
                        "application_name": "yoladgu_backend"
                    }
                }
            )
            
            # Create async session maker
            self.async_session_maker = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            
            # Set up connection pool event listeners
            self._setup_pool_event_listeners()
            
            # Test connection
            await self._test_connection()
            
            # Start health check task
            self._start_health_check_task()
            
            logger.info("✅ Database connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database connection pool: {e}")
            raise
    
    async def _test_connection(self):
        """Test database connection"""
        
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute("SELECT 1")
                await result.fetchone()
            
            logger.info("✅ Database connection test successful")
            
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            raise
    
    def _setup_pool_event_listeners(self):
        """Set up connection pool event listeners for monitoring"""
        
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            """Called when a new connection is created"""
            self.pool_stats["total_connections"] += 1
            self.pool_stats["idle_connections"] += 1
            logger.debug(f"🔌 New database connection created. Total: {self.pool_stats['total_connections']}")
        
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Called when a connection is checked out from the pool"""
            self.pool_stats["checked_out_connections"] += 1
            self.pool_stats["idle_connections"] = max(0, self.pool_stats["idle_connections"] - 1)
            self.pool_stats["active_connections"] += 1
            logger.debug(f"📤 Database connection checked out. Active: {self.pool_stats['active_connections']}")
        
        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Called when a connection is returned to the pool"""
            self.pool_stats["checked_in_connections"] += 1
            self.pool_stats["active_connections"] = max(0, self.pool_stats["active_connections"] - 1)
            self.pool_stats["idle_connections"] += 1
            logger.debug(f"📥 Database connection checked in. Idle: {self.pool_stats['idle_connections']}")
        
        @event.listens_for(self.engine, "overflow")
        def receive_overflow(dbapi_connection, connection_record):
            """Called when a connection is created due to pool overflow"""
            self.pool_stats["overflow_connections"] += 1
            logger.warning(f"⚠️ Database connection pool overflow. Overflow: {self.pool_stats['overflow_connections']}")
    
    def _start_health_check_task(self):
        """Start periodic health check task"""
        
        async def health_check_loop():
            while True:
                try:
                    await asyncio.sleep(self.health_check_interval)
                    await self._perform_health_check()
                except Exception as e:
                    logger.error(f"❌ Health check failed: {e}")
        
        self._health_check_task = asyncio.create_task(health_check_loop())
        logger.info(f"🔄 Started database health check task (interval: {self.health_check_interval}s)")
    
    async def _perform_health_check(self):
        """Perform database health check"""
        
        try:
            if not self.engine:
                logger.warning("⚠️ Database engine not initialized")
                return
            
            # Check pool status
            pool = self.engine.pool
            if pool:
                pool_status = {
                    "size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "invalid": pool.invalid()
                }
                
                logger.debug(f"📊 Database pool status: {pool_status}")
                
                # Alert if pool is getting full
                if pool.checkedout() > pool.size() * 0.8:
                    logger.warning(f"⚠️ Database pool usage high: {pool.checkedout()}/{pool.size()} connections in use")
                
                # Alert if many invalid connections
                if pool.invalid() > 0:
                    logger.warning(f"⚠️ Database pool has {pool.invalid()} invalid connections")
            
            # Test connection
            await self._test_connection()
            
        except Exception as e:
            logger.error(f"❌ Database health check failed: {e}")
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session from pool"""
        
        if not self.async_session_maker:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        session = self.async_session_maker()
        
        try:
            yield session
        except Exception as e:
            logger.error(f"❌ Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()
    
    @asynccontextmanager
    async def get_session_context(self):
        """Get database session with context manager"""
        
        if not self.async_session_maker:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        session = self.async_session_maker()
        
        try:
            yield session
        except Exception as e:
            logger.error(f"❌ Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def execute_in_session(self, operation_func, *args, **kwargs):
        """Execute operation in a managed database session"""
        
        async with self.get_session_context() as session:
            try:
                result = await operation_func(session, *args, **kwargs)
                await session.commit()
                return result
            except Exception as e:
                await session.rollback()
                raise
    
    async def execute_in_transaction(self, operation_func, *args, **kwargs):
        """Execute operation in a transaction with rollback on error"""
        
        async with self.get_session_context() as session:
            async with session.begin():
                try:
                    result = await operation_func(session, *args, **kwargs)
                    return result
                except Exception as e:
                    # Transaction will automatically rollback
                    raise
    
    async def close(self):
        """Close database connections and cleanup"""
        
        try:
            logger.info("🔄 Closing database connections...")
            
            # Stop health check task
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Close engine
            if self.engine:
                await self.engine.dispose()
                self.engine = None
            
            # Reset session maker
            self.async_session_maker = None
            
            logger.info("✅ Database connections closed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error closing database connections: {e}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        
        if not self.engine or not self.engine.pool:
            return self.pool_stats
        
        pool = self.engine.pool
        
        # Update stats from pool
        self.pool_stats.update({
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
            "total_connections": pool.size() + pool.overflow(),
            "active_connections": pool.checkedout(),
            "idle_connections": pool.checkedin()
        })
        
        return self.pool_stats
    
    async def reset_pool(self):
        """Reset connection pool (useful for maintenance)"""
        
        try:
            logger.info("🔄 Resetting database connection pool...")
            
            if self.engine:
                # Dispose and recreate engine
                await self.engine.dispose()
                
                # Recreate engine
                self.engine = create_async_engine(
                    settings.database_url,
                    echo=settings.database_echo,
                    poolclass=QueuePool,
                    pool_size=settings.database_pool_size,
                    max_overflow=settings.database_max_overflow,
                    pool_pre_ping=True,
                    pool_recycle=settings.database_pool_recycle,
                    pool_timeout=settings.database_pool_timeout,
                    pool_reset_on_return='commit'
                )
                
                # Update session maker
                self.async_session_maker = async_sessionmaker(
                    bind=self.engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autoflush=False,
                    autocommit=False
                )
                
                # Reset stats
                self.pool_stats = {
                    "total_connections": 0,
                    "active_connections": 0,
                    "idle_connections": 0,
                    "overflow_connections": 0,
                    "checked_out_connections": 0,
                    "checked_in_connections": 0
                }
                
                # Set up event listeners again
                self._setup_pool_event_listeners()
                
                logger.info("✅ Database connection pool reset successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to reset database connection pool: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status"""
        
        try:
            if not self.engine:
                return {
                    "status": "unhealthy",
                    "reason": "Database not initialized",
                    "timestamp": None
                }
            
            # Test connection
            start_time = asyncio.get_event_loop().time()
            await self._test_connection()
            response_time = asyncio.get_event_loop().time() - start_time
            
            # Get pool stats
            pool_stats = self.get_pool_stats()
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "pool_stats": pool_stats,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }


# Global database manager instance
database_manager = DatabaseManager()


# Legacy compatibility functions
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Legacy function for backward compatibility"""
    logger.warning("⚠️ Using legacy get_db() function. Use database_manager.get_session() instead.")
    async for session in database_manager.get_session():
        yield session


async def get_db_session() -> AsyncSession:
    """Get a single database session"""
    if not database_manager.async_session_maker:
        raise RuntimeError("Database not initialized. Call database_manager.initialize() first.")
    
    return database_manager.async_session_maker()


# Database dependency for FastAPI
async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Database dependency for FastAPI"""
    async for session in database_manager.get_session():
        yield session


# Initialize database on startup
async def init_database():
    """Initialize database on application startup"""
    await database_manager.initialize()


# Cleanup database on shutdown
async def cleanup_database():
    """Cleanup database on application shutdown"""
    await database_manager.close()
