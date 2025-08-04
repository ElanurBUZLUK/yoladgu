from datetime import datetime

import structlog
from app.api.v1.endpoints import (
    ai,
    analytics,
    auth,
    embeddings,
    llm_assistant,
    performance_monitor,
    plan_items,
    questions,
    quiz_sessions,
    recommendations,
    scheduler,
    solutions,
    streams,
    study_plans,
    subjects,
    system_health,
    topics,
    users,
)
from app.core.config import settings
from app.core.rate_limiter import get_rate_limiter
from app.services.scheduler_service import offline_scheduler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

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
        structlog.processors.JSONRenderer(),
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
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Rate limiter instance - will be initialized during startup
rate_limiter = None

Instrumentator().instrument(app).expose(app)


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Rate limiting ve cache middleware"""
    global rate_limiter

    # Initialize rate limiter if not already done
    if rate_limiter is None:
        rate_limiter = await get_rate_limiter()

    # 1. Cache check (sadece GET ve cacheable POST'lar için)
    cached_response = await rate_limiter.get_cached_response(request)
    if cached_response:
        return cached_response

    # 2. Rate limit check
    rate_limit_response = await rate_limiter.check_rate_limit(request)
    if rate_limit_response:
        return rate_limit_response

    # 3. Process request
    response = await call_next(request)

    # 4. Cache response (if applicable)
    await rate_limiter.cache_response(request, response)

    # 5. Add rate limit headers
    response.headers["X-RateLimit-Applied"] = "true"

    return response


# Custom metrics for Yoladgu
yoladgu_requests_total = Counter(
    "yoladgu_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

yoladgu_health_checks = Counter(
    "yoladgu_health_checks_total", "Total health checks performed", ["service"]
)

yoladgu_service_health = Gauge(
    "yoladgu_service_health_status",
    "Service health status (1=healthy, 0=unhealthy)",
    ["service"],
)

# Removed duplicate - handled by prometheus_monitoring

# IMPORTANT: Middleware are applied in reverse order of registration
# Last registered = First to process requests

# CORS Middleware (should be last registered, first to process)
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
app.include_router(
    embeddings.router, prefix=f"{settings.API_V1_STR}/embeddings", tags=["embeddings"]
)
app.include_router(
    scheduler.router, prefix=f"{settings.API_V1_STR}/scheduler", tags=["scheduler"]
)
app.include_router(
    streams.router, prefix=settings.API_V1_STR + "/streams", tags=["streams"]
)
app.include_router(
    recommendations.router, prefix=settings.API_V1_STR, tags=["recommendations"]
)
app.include_router(
    llm_assistant.router, prefix=settings.API_V1_STR, tags=["llm-assistant"]
)
app.include_router(
    system_health.router, prefix=settings.API_V1_STR, tags=["system-health"]
)
app.include_router(
    performance_monitor.router,
    prefix=f"{settings.API_V1_STR}/performance",
    tags=["performance"],
)
app.include_router(quiz_sessions.router, prefix=settings.API_V1_STR)
app.include_router(analytics.router, prefix=settings.API_V1_STR)

# FastAPI lifecycle events for stream consumer
import asyncio

# FastAPI lifecycle events for stream consumer

# Note: To use lifespan, initialize FastAPI with lifespan parameter:
# app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)
# For now, we'll use the deprecated but simpler @app.on_event approach


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup tasks - optimized order for data flow"""
    try:
        logger.info("application_startup_started")

        # Step 1: Initialize database connections first
        from app.db.database import init_async_db

        init_async_db()
        logger.info("Database connections initialized")

        # Step 2: Initialize core infrastructure services (Redis, Neo4j)
        from app.services.neo4j_service import neo4j_service
        from app.services.redis_service import redis_service

        # Initialize Redis service (needed for rate limiting, caching)
        _ = redis_service  # This triggers __init__ if not already initialized
        logger.info("Redis service initialized")

        # Initialize Neo4j service (graph database)
        _ = neo4j_service  # This triggers __init__ if not already initialized
        logger.info("Neo4j service initialized")

        # Step 3: Initialize ML/AI services that depend on core services
        await offline_scheduler.initialize()
        logger.info("Offline scheduler initialized")

        # Step 4: Initialize HTTP clients for external services
        from app.services.question_ingestion_service import (
            get_question_ingestion_service,
        )

        await get_question_ingestion_service()  # Initialize async client
        logger.info("Question ingestion service initialized")

        # Step 5: Start background consumers/workers (last)
        from app.services.enhanced_stream_consumer import stream_consumer_manager

        if stream_consumer_manager and not stream_consumer_manager.running:
            # Start consumer in background task
            asyncio.create_task(stream_consumer_manager.start_consumer())
            logger.info("Stream consumer started successfully")
        else:
            logger.info("Stream consumer already running or disabled")

        logger.info("application_startup_completed")

    except Exception as e:
        logger.error("application_startup_error", error=str(e), exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks - reverse order of startup"""
    try:
        logger.info("application_shutdown_started")

        # Step 1: Stop background workers first
        from app.services.enhanced_stream_consumer import stream_consumer_manager

        if stream_consumer_manager and stream_consumer_manager.running:
            await stream_consumer_manager._cleanup_consumer()
            logger.info("Stream consumer stopped successfully")

        # Step 2: Shutdown ML/AI services
        await offline_scheduler.shutdown()
        logger.info("Offline scheduler shutdown")

        # Step 3: Close HTTP clients
        from app.services.question_ingestion_service import (
            close_question_ingestion_service,
        )

        await close_question_ingestion_service()
        logger.info("Question ingestion service closed")

        # Step 4: Close infrastructure services (reverse order)
        from app.services.neo4j_service import neo4j_service
        from app.services.redis_service import redis_service

        # Close Neo4j driver
        neo4j_service.close()
        logger.info("Neo4j service closed")

        # Close Redis client (last infrastructure service)
        redis_service.close()
        logger.info("Redis service closed")

        # Step 5: Close database connections (last)
        global async_engine
        from app.db.database import async_engine

        if async_engine:
            await async_engine.dispose()
            logger.info("Database connections closed")

        logger.info("application_shutdown_completed")

    except Exception as e:
        logger.error("application_shutdown_error", error=str(e), exc_info=True)


@app.get("/health")
def health_check():
    """Gelişmiş sistem sağlık kontrolü with Prometheus metrics"""
    # import time  # Unused import removed

    # start_time = time.time()  # Unused variable removed

    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {},
    }

    # PostgreSQL health check
    try:
        import psycopg2

        if settings.DATABASE_URL is None:
            health_status["services"][
                "postgresql"
            ] = "unhealthy: DATABASE_URL not configured"
            health_status["status"] = "degraded"
        else:
            conn = psycopg2.connect(
                settings.DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")
            )
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

        if settings.redis_url is None:
            health_status["services"]["redis"] = "unhealthy: redis_url not configured"
            health_status["status"] = "degraded"
        else:
            r = redis.from_url(settings.redis_url)
            if r is not None:
                r.ping()
                health_status["services"]["redis"] = "healthy"
            else:
                health_status["services"][
                    "redis"
                ] = "unhealthy: redis connection failed"
                health_status["status"] = "degraded"
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
                health_status["services"][
                    "embedding"
                ] = "unhealthy: invalid embedding dimension"
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
        from sqlalchemy import text

        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@app.get("/live")
def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}
