"""
Enhanced Database Manager with advanced connection pooling, query optimization, and monitoring
"""
import asyncio
import logging
import time
from typing import AsyncGenerator, Optional, Dict, Any, List, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    create_async_engine, 
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy import event, text, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError, OperationalError, TimeoutError

from app.core.config import settings

# Create declarative base for models
Base = declarative_base()

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_type: str
    execution_time: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    rows_affected: Optional[int] = None
    connection_id: Optional[str] = None


@dataclass
class ConnectionMetrics:
    """Connection pool metrics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    overflow_connections: int = 0
    checked_out_connections: int = 0
    checked_in_connections: int = 0
    connection_errors: int = 0
    connection_timeouts: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DatabaseHealth:
    """Database health status"""
    status: str  # healthy, degraded, unhealthy
    response_time: float
    connection_count: int
    error_rate: float
    slow_queries: int
    last_check: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class QueryOptimizer:
    """Query optimization and analysis"""
    
    def __init__(self):
        self.query_history: deque = deque(maxlen=1000)  # Last 1000 queries
        self.slow_query_threshold = 1.0  # 1 second
        self.query_patterns: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "errors": 0,
            "last_seen": None
        })
    
    def record_query(self, query_metrics: QueryMetrics):
        """Record query metrics for analysis"""
        self.query_history.append(query_metrics)
        
        # Update pattern statistics
        pattern = self._extract_query_pattern(query_metrics.query_type)
        stats = self.query_patterns[pattern]
        
        stats["count"] += 1
        stats["total_time"] += query_metrics.execution_time
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["min_time"] = min(stats["min_time"], query_metrics.execution_time)
        stats["max_time"] = max(stats["max_time"], query_metrics.execution_time)
        stats["last_seen"] = query_metrics.timestamp
        
        if not query_metrics.success:
            stats["errors"] += 1
    
    def _extract_query_pattern(self, query_type: str) -> str:
        """Extract query pattern for analysis"""
        # Simple pattern extraction - can be enhanced
        if "SELECT" in query_type.upper():
            return "SELECT"
        elif "INSERT" in query_type.upper():
            return "INSERT"
        elif "UPDATE" in query_type.upper():
            return "UPDATE"
        elif "DELETE" in query_type.upper():
            return "DELETE"
        else:
            return "OTHER"
    
    def get_slow_queries(self, limit: int = 10) -> List[QueryMetrics]:
        """Get slowest queries"""
        slow_queries = [q for q in self.query_history if q.execution_time > self.slow_query_threshold]
        return sorted(slow_queries, key=lambda x: x.execution_time, reverse=True)[:limit]
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get comprehensive query statistics"""
        if not self.query_history:
            return {}
        
        execution_times = [q.execution_time for q in self.query_history]
        
        return {
            "total_queries": len(self.query_history),
            "successful_queries": len([q for q in self.query_history if q.success]),
            "failed_queries": len([q for q in self.query_history if not q.success]),
            "slow_queries": len([q for q in self.query_history if q.execution_time > self.slow_query_threshold]),
            "execution_time_stats": {
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            },
            "query_patterns": dict(self.query_patterns),
            "recent_activity": {
                "last_hour": len([q for q in self.query_history if q.timestamp > datetime.utcnow() - timedelta(hours=1)]),
                "last_24h": len([q for q in self.query_history if q.timestamp > datetime.utcnow() - timedelta(days=1)])
            }
        }


class ConnectionPoolMonitor:
    """Connection pool monitoring and management"""
    
    def __init__(self):
        self.metrics = ConnectionMetrics()
        self.health_history: deque = deque(maxlen=100)
        self.connection_events: List[Dict[str, Any]] = []
    
    def update_metrics(self, pool):
        """Update connection pool metrics"""
        try:
            self.metrics.total_connections = pool.size()
            self.metrics.active_connections = pool.checkedin()
            self.metrics.idle_connections = pool.checkedout()
            self.metrics.overflow_connections = pool.overflow()
            self.metrics.checked_out_connections = pool.checkedout()
            self.metrics.checked_in_connections = pool.checkedin()
            self.metrics.last_updated = datetime.utcnow()
        except Exception as e:
            logger.error(f"Error updating pool metrics: {e}")
    
    def record_connection_event(self, event_type: str, details: Dict[str, Any]):
        """Record connection pool events"""
        event_data = {
            "timestamp": datetime.utcnow(),
            "event_type": event_type,
            "details": details
        }
        self.connection_events.append(event_data)
        
        # Keep only last 100 events
        if len(self.connection_events) > 100:
            self.connection_events = self.connection_events[-100:]
    
    def get_pool_health(self) -> Dict[str, Any]:
        """Get connection pool health status"""
        utilization = self.metrics.active_connections / max(self.metrics.total_connections, 1)
        
        return {
            "utilization_rate": utilization,
            "overflow_rate": self.metrics.overflow_connections / max(self.metrics.total_connections, 1),
            "error_rate": self.metrics.connection_errors / max(self.metrics.total_connections + self.metrics.connection_errors, 1),
            "metrics": {
                "total_connections": self.metrics.total_connections,
                "active_connections": self.metrics.active_connections,
                "idle_connections": self.metrics.idle_connections,
                "overflow_connections": self.metrics.overflow_connections
            },
            "status": "healthy" if utilization < 0.8 and self.metrics.connection_errors == 0 else "degraded"
        }


class EnhancedDatabaseManager:
    """Enhanced database manager with advanced features"""
    
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.async_session_maker: Optional[async_sessionmaker] = None
        self.query_optimizer = QueryOptimizer()
        self.pool_monitor = ConnectionPoolMonitor()
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        self.health_check_interval = 300  # 5 minutes
        self._health_check_task: Optional[asyncio.Task] = None
        self._maintenance_task: Optional[asyncio.Task] = None
        self.is_initialized = False
        
        # Performance settings
        self.slow_query_threshold = 1.0  # seconds
        self.max_connection_lifetime = 3600  # 1 hour
        self.connection_retry_delay = 1.0  # seconds
        
        # Monitoring
        self.performance_alerts: List[Dict[str, Any]] = []
        self.last_maintenance = datetime.utcnow()
    
    async def initialize(self):
        """Initialize enhanced database connection pool"""
        
        try:
            logger.info("🚀 Initializing enhanced database connection pool...")
            
            # Create async engine with advanced connection pooling
            self.engine = create_async_engine(
                settings.database_url,
                echo=settings.database_echo,
                poolclass=QueuePool,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_pre_ping=True,
                pool_recycle=self.max_connection_lifetime,
                pool_timeout=settings.database_pool_timeout,
                pool_reset_on_return='commit',
                connect_args={
                    "server_settings": {
                        "application_name": "yoladgu_backend_enhanced",
                        "statement_timeout": "30000",  # 30 seconds
                        "lock_timeout": "10000",  # 10 seconds
                        "idle_in_transaction_session_timeout": "60000"  # 60 seconds
                    }
                }
            )
            
            # Create async session maker with enhanced configuration
            self.async_session_maker = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            
            # Set up enhanced event listeners
            self._setup_enhanced_event_listeners()
            
            # Test connection with metrics
            await self._test_connection_with_metrics()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.is_initialized = True
            logger.info("✅ Enhanced database connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced database connection pool: {e}")
            raise
    
    def _setup_enhanced_event_listeners(self):
        """Set up enhanced database event listeners"""
        
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            """Handle new connection creation"""
            self.pool_monitor.record_connection_event("connect", {
                "connection_id": id(dbapi_connection),
                "timestamp": datetime.utcnow().isoformat()
            })
            logger.debug(f"New database connection created: {id(dbapi_connection)}")
        
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout"""
            self.pool_monitor.record_connection_event("checkout", {
                "connection_id": id(dbapi_connection),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Handle connection checkin"""
            self.pool_monitor.record_connection_event("checkin", {
                "connection_id": id(dbapi_connection),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        @event.listens_for(self.engine, "disconnect")
        def receive_disconnect(dbapi_connection, connection_record):
            """Handle connection disconnect"""
            self.pool_monitor.record_connection_event("disconnect", {
                "connection_id": id(dbapi_connection),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _test_connection_with_metrics(self):
        """Test database connection with performance metrics"""
        
        start_time = time.time()
        success = False
        error_message = None
        
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                await result.fetchone()
            
            success = True
            execution_time = time.time() - start_time
            
            # Record metrics
            query_metrics = QueryMetrics(
                query_type="SELECT 1",
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
                success=success,
                error_message=error_message
            )
            self.query_optimizer.record_query(query_metrics)
            
            logger.info(f"✅ Database connection test successful (took {execution_time:.3f}s)")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            # Record failed query
            query_metrics = QueryMetrics(
                query_type="SELECT 1",
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
                success=success,
                error_message=error_message
            )
            self.query_optimizer.record_query(query_metrics)
            
            logger.error(f"❌ Database connection test failed: {e}")
            raise
    
    def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        
        # Health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # Maintenance task
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        logger.info("✅ Background tasks started")
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_check()
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _maintenance_loop(self):
        """Background maintenance loop"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._perform_maintenance()
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        
        try:
            start_time = time.time()
            
            # Test basic connectivity
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                await result.fetchone()
            
            response_time = time.time() - start_time
            
            # Update pool metrics
            if hasattr(self.engine, 'pool'):
                self.pool_monitor.update_metrics(self.engine.pool)
            
            # Check for performance issues
            slow_queries = len(self.query_optimizer.get_slow_queries())
            error_rate = self._calculate_error_rate()
            
            health_status = DatabaseHealth(
                status="healthy" if response_time < 1.0 and error_rate < 0.1 else "degraded",
                response_time=response_time,
                connection_count=self.pool_monitor.metrics.total_connections,
                error_rate=error_rate,
                slow_queries=slow_queries,
                last_check=datetime.utcnow(),
                details={
                    "pool_health": self.pool_monitor.get_pool_health(),
                    "query_stats": self.query_optimizer.get_query_statistics()
                }
            )
            
            self.pool_monitor.health_history.append(health_status)
            
            # Log health status
            if health_status.status != "healthy":
                logger.warning(f"Database health check: {health_status.status} (response_time={response_time:.3f}s, error_rate={error_rate:.2%})")
            else:
                logger.debug(f"Database health check: {health_status.status} (response_time={response_time:.3f}s)")
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def _perform_maintenance(self):
        """Perform database maintenance tasks"""
        
        try:
            logger.info("🔧 Starting database maintenance...")
            
            # Clean up old metrics
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            self.query_optimizer.query_history = deque(
                [q for q in self.query_optimizer.query_history if q.timestamp > cutoff_time],
                maxlen=1000
            )
            
            # Clean up old connection events
            self.pool_monitor.connection_events = [
                e for e in self.pool_monitor.connection_events 
                if e["timestamp"] > cutoff_time
            ]
            
            # Clean up old health history
            self.pool_monitor.health_history = deque(
                [h for h in self.pool_monitor.health_history if h.last_check > cutoff_time],
                maxlen=100
            )
            
            # Analyze and log performance insights
            await self._analyze_performance()
            
            self.last_maintenance = datetime.utcnow()
            logger.info("✅ Database maintenance completed")
            
        except Exception as e:
            logger.error(f"Maintenance failed: {e}")
    
    async def _analyze_performance(self):
        """Analyze database performance and generate insights"""
        
        stats = self.query_optimizer.get_query_statistics()
        
        if not stats:
            return
        
        # Check for performance issues
        avg_execution_time = stats["execution_time_stats"]["mean"]
        slow_query_count = stats["slow_queries"]
        error_rate = stats["failed_queries"] / max(stats["total_queries"], 1)
        
        insights = []
        
        if avg_execution_time > 0.5:
            insights.append(f"High average query time: {avg_execution_time:.3f}s")
        
        if slow_query_count > 10:
            insights.append(f"Many slow queries: {slow_query_count}")
        
        if error_rate > 0.05:
            insights.append(f"High error rate: {error_rate:.2%}")
        
        if insights:
            alert = {
                "timestamp": datetime.utcnow(),
                "type": "performance_alert",
                "insights": insights,
                "stats": stats
            }
            self.performance_alerts.append(alert)
            logger.warning(f"Performance insights: {'; '.join(insights)}")
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        if not self.query_optimizer.query_history:
            return 0.0
        
        recent_queries = [
            q for q in self.query_optimizer.query_history 
            if q.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        if not recent_queries:
            return 0.0
        
        failed_queries = len([q for q in recent_queries if not q.success])
        return failed_queries / len(recent_queries)
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with enhanced monitoring"""
        
        if not self.is_initialized:
            raise RuntimeError("Database manager not initialized")
        
        session = None
        start_time = time.time()
        
        try:
            session = self.async_session_maker()
            
            # Monitor session creation
            creation_time = time.time() - start_time
            if creation_time > 0.1:  # Log slow session creation
                logger.warning(f"Slow session creation: {creation_time:.3f}s")
            
            yield session
            
        except Exception as e:
            # Record failed session
            execution_time = time.time() - start_time
            query_metrics = QueryMetrics(
                query_type="session_creation",
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )
            self.query_optimizer.record_query(query_metrics)
            raise
            
        finally:
            if session:
                try:
                    await session.close()
                except Exception as e:
                    logger.error(f"Error closing session: {e}")
    
    async def execute_with_metrics(self, query: str, params: Optional[Dict] = None) -> Tuple[Any, float]:
        """Execute query with performance metrics"""
        
        start_time = time.time()
        success = False
        error_message = None
        result = None
        
        try:
            async with self.get_session() as session:
                result = await session.execute(text(query), params or {})
                await session.commit()
                success = True
                
        except Exception as e:
            error_message = str(e)
            raise
            
        finally:
            execution_time = time.time() - start_time
            
            # Record query metrics
            query_metrics = QueryMetrics(
                query_type=query[:50] + "..." if len(query) > 50 else query,
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
                success=success,
                error_message=error_message
            )
            self.query_optimizer.record_query(query_metrics)
            
            # Log slow queries
            if execution_time > self.slow_query_threshold:
                logger.warning(f"Slow query detected ({execution_time:.3f}s): {query[:100]}...")
        
        return result, execution_time
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        return {
            "query_optimizer": self.query_optimizer.get_query_statistics(),
            "pool_monitor": self.pool_monitor.get_pool_health(),
            "health_history": [
                {
                    "status": h.status,
                    "response_time": h.response_time,
                    "error_rate": h.error_rate,
                    "last_check": h.last_check.isoformat()
                }
                for h in list(self.pool_monitor.health_history)[-10:]  # Last 10 health checks
            ],
            "performance_alerts": self.performance_alerts[-5:],  # Last 5 alerts
            "maintenance": {
                "last_maintenance": self.last_maintenance.isoformat(),
                "is_initialized": self.is_initialized
            }
        }
    
    async def close(self):
        """Close database connections and cleanup"""
        
        try:
            logger.info("🔄 Closing enhanced database connections...")
            
            # Cancel background tasks
            if self._health_check_task:
                self._health_check_task.cancel()
            if self._maintenance_task:
                self._maintenance_task.cancel()
            
            # Close engine
            if self.engine:
                await self.engine.dispose()
            
            self.is_initialized = False
            logger.info("✅ Enhanced database connections closed")
            
        except Exception as e:
            logger.error(f"❌ Error closing database connections: {e}")


# Global enhanced database manager instance
enhanced_database_manager = EnhancedDatabaseManager()
