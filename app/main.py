from fastapi import FastAPI, Depends, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.db.database import get_db
import redis
from app.core.config import settings
from app.api.v1.endpoints import (
    auth,
    users,
    questions,
    solutions,
    study_plans,
    topics,
    subjects,
    plan_items,
    ai,
    embeddings,
    streams,
    recommendations,
    llm_assistant,
    system_health,
    quiz_sessions,
    analytics,
)
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from neo4j import GraphDatabase
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import structlog
import redis
import psycopg2
from datetime import datetime

# Structured logging setup
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Initialize Prometheus monitoring - simplified approach
instrumentator = Instrumentator()
instrumentator.instrument(app)
instrumentator.expose(app, endpoint="/metrics")

# Custom metrics for Yoladgu
yoladgu_requests_total = Counter(
    'yoladgu_requests_total', 
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

yoladgu_health_checks = Counter(
    'yoladgu_health_checks_total',
    'Total health checks performed',
    ['service']
)

yoladgu_service_health = Gauge(
    'yoladgu_service_health_status',
    'Service health status (1=healthy, 0=unhealthy)',
    ['service']
)

# Removed duplicate - handled by prometheus_monitoring

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Geliştirme ortamı için localhost:4200'e izin ver
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:4200"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(users.router, prefix=settings.API_V1_STR)
app.include_router(questions.router, prefix=settings.API_V1_STR)
app.include_router(solutions.router, prefix=settings.API_V1_STR)
app.include_router(study_plans.router, prefix=settings.API_V1_STR)
app.include_router(topics.router, prefix=settings.API_V1_STR)
app.include_router(subjects.router, prefix=settings.API_V1_STR)
app.include_router(plan_items.router, prefix=settings.API_V1_STR)
app.include_router(ai.router, prefix=settings.API_V1_STR)
app.include_router(embeddings.router, prefix=settings.API_V1_STR + "/embeddings", tags=["embeddings"])
app.include_router(streams.router, prefix=settings.API_V1_STR + "/streams", tags=["streams"])
app.include_router(recommendations.router, prefix=settings.API_V1_STR, tags=["recommendations"])
app.include_router(llm_assistant.router, prefix=settings.API_V1_STR, tags=["llm-assistant"])
app.include_router(system_health.router, prefix=settings.API_V1_STR, tags=["system-health"])
app.include_router(quiz_sessions.router, prefix=settings.API_V1_STR)
app.include_router(analytics.router, prefix=settings.API_V1_STR)

# FastAPI lifecycle events for stream consumer
import asyncio

# Note: To use lifespan, initialize FastAPI with lifespan parameter:
# app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)
# For now, we'll use the deprecated but simpler @app.on_event approach

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Starting application services...")
    
    # Initialize singleton services
    try:
        from app.services.neo4j_service import neo4j_service
        from app.services.redis_service import redis_service
        
        # Initialize Neo4j service
        neo4j_service  # This triggers __init__ if not already initialized
        logger.info("Neo4j service initialized")
        
        # Initialize Redis service  
        redis_service  # This triggers __init__ if not already initialized
        logger.info("Redis service initialized")
        
    except Exception as e:
        logger.error("Failed to initialize singleton services", error=str(e))
    
    # Start stream consumer in background
    try:
        from app.services.enhanced_stream_consumer import stream_consumer_manager
        if stream_consumer_manager and not stream_consumer_manager.running:
            # Start consumer in background task
            asyncio.create_task(stream_consumer_manager.start_consumer())
            logger.info("Stream consumer started successfully")
        else:
            logger.info("Stream consumer already running or disabled")
    except Exception as e:
        logger.error("Failed to start stream consumer", error=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Shutting down application services...")
    
    # Stop stream consumer
    try:
        from app.services.enhanced_stream_consumer import stream_consumer_manager
        if stream_consumer_manager and stream_consumer_manager.running:
            await stream_consumer_manager._cleanup_consumer()
            logger.info("Stream consumer stopped successfully")
    except Exception as e:
        logger.error("Error stopping stream consumer", error=str(e))
    
    # Close singleton services
    try:
        from app.services.neo4j_service import neo4j_service
        from app.services.redis_service import redis_service
        
        # Close Neo4j driver
        neo4j_service.close()
        logger.info("Neo4j service closed")
        
        # Close Redis client
        redis_service.close()
        logger.info("Redis service closed")
        
    except Exception as e:
        logger.error("Error closing singleton services", error=str(e))

@app.get("/health")
def health_check():
    """Gelişmiş sistem sağlık kontrolü with Prometheus metrics"""
    import time
    start_time = time.time()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # PostgreSQL health check
    try:
        import psycopg2
        conn = psycopg2.connect(settings.DATABASE_URL.replace('postgresql+psycopg2://', 'postgresql://'))
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        health_status["services"]["postgresql"] = "healthy"
    except Exception as e:
        health_status["services"]["postgresql"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Redis health check
    try:
        import redis
        r = redis.from_url(settings.redis_url)
        r.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Neo4j health check (using singleton service)
    if settings.USE_NEO4J:
        try:
            from app.services.neo4j_service import neo4j_service
            if neo4j_service._driver:
                # Test existing connection
                with neo4j_service._driver.session() as session:
                    session.run("RETURN 1")
                health_status["services"]["neo4j"] = "healthy"
            else:
                health_status["services"]["neo4j"] = "unhealthy: driver not initialized"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["neo4j"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
    
    # Embedding service health check
    if settings.USE_EMBEDDING:
        try:
            from app.services.embedding_service import compute_embedding
            test_embedding = compute_embedding("test")
            if len(test_embedding) == settings.EMBEDDING_DIM:
                health_status["services"]["embedding"] = "healthy"
            else:
                health_status["services"]["embedding"] = "unhealthy: invalid embedding dimension"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["embedding"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
    
    return health_status

# Metrics endpoint is now handled by instrumentator

@app.get("/ready")
def readiness_check():
    """Kubernetes readiness probe"""
    try:
        # Temel servis kontrolü
        from app.db.database import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

@app.get("/live")
def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()} 