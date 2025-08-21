from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # Added for serving static files
from contextlib import asynccontextmanager
import structlog

from app.core.config import settings
from app.database_enhanced import enhanced_database_manager as database_manager
from app.services.cache_service import cache_service

# Import middleware
from app.middleware.error_handler import error_handler
from app.middleware.rate_limiter import rate_limiter
from app.middleware.logging import logging_middleware

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting application", 
                environment=settings.environment.value,
                version=settings.version,
                debug=settings.debug)
    
    # Temporarily disable database initialization for development
    # await database_manager.initialize()
    logger.info("⚠️ Database initialization temporarily disabled for development")
    await cache_service.connect()
    
    # Background scheduler'ı başlat
    from app.services.background_scheduler import background_scheduler
    await background_scheduler.start()
    
    # Initialize MCP
    from app.core.mcp_utils import mcp_utils
    from app.services.llm_gateway import llm_gateway
    
    try:
        # Initialize MCP utils
        mcp_initialized = await mcp_utils.initialize()
        if mcp_initialized:
            logger.info("✅ MCP Utils initialized successfully")
            
            # Initialize LLM Gateway with MCP
            await llm_gateway.initialize()
            logger.info("✅ LLM Gateway initialized with MCP")
        else:
            logger.warning("⚠️ MCP initialization failed, using direct LLM calls")
    except Exception as e:
        logger.error(f"❌ MCP initialization error: {e}")
    
    # Initialize monitoring service
    from app.services.monitoring_service import monitoring_service
    logger.info("Application startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    # Cleanup MCP
    from app.core.mcp_utils import mcp_utils
    from app.services.llm_gateway import llm_gateway
    
    try:
        await mcp_utils.cleanup()
        await llm_gateway.cleanup()
        logger.info("MCP cleanup completed")
    except Exception as e:
        logger.error(f"MCP cleanup error: {e}")
    
    await background_scheduler.stop()
    await database_manager.close()
    await cache_service.close()
    logger.info("Application shutdown completed")


app = FastAPI(
    title=settings.api_docs_title,
    version=settings.api_docs_version,
    description=settings.api_docs_description,
    contact={
        "name": settings.api_docs_contact_name,
        "email": settings.api_docs_contact_email,
    },
    openapi_url="/openapi.json" if settings.api_docs_enabled else None,
    docs_url="/docs" if settings.api_docs_enabled else None,
    redoc_url="/redoc" if settings.api_docs_enabled else None,
    lifespan=lifespan
)

# Add middleware (order matters - last added is first executed)
# Error handling middleware
app.middleware("http")(error_handler)

# Rate limiting middleware (only if enabled)
if settings.rate_limit_enabled:
    app.middleware("http")(rate_limiter)
    logger.info("Rate limiting middleware enabled")

# Logging middleware
app.middleware("http")(logging_middleware)

# CORS middleware with enhanced configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

@app.get("/health", include_in_schema=False)
async def health_check():
    """Health check endpoint"""
    from app.services.monitoring_service import monitoring_service
    
    try:
        health_status = monitoring_service.get_health_status()
        return {
            "status": health_status["status"],
            "version": settings.version,
            "environment": settings.environment.value,
            "timestamp": health_status["timestamp"]
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "version": settings.version,
            "environment": settings.environment.value,
            "error": str(e)
        }

@app.get("/health/simple", include_in_schema=False)
async def simple_health_check():
    """Simple health check for load balancers"""
    return {"status": "ok"}


@app.get("/healthz", include_in_schema=False)
async def healthz():
    """Process health check (uptime)"""
    return {
        "status": "ok",
        "uptime": "running",
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.get("/readyz", include_in_schema=False)
async def readyz():
    """Readiness check (DB/Redis/pgvector)"""
    from app.database import database_manager
    from redis.asyncio import Redis
    from app.core.config import settings
    from app.services.system_initialization_service import system_initialization_service
    
    db_status = "not_ok"
    redis_status = "not_ok"
    pgvector_status = {"extension": False, "vector_column": False, "index": False}
    
    try:
        # Database check
        async with database_manager.get_session() as db:
            await db.execute("SELECT 1")
            db_status = "ok"
    except Exception as e:
        logger.error(f"Readiness check failed: Database - {e}")

    try:
        # Redis check
        redis_client = Redis.from_url(settings.redis_url)
        await redis_client.ping()
        await redis_client.close()
        redis_status = "ok"
    except Exception as e:
        logger.error(f"Readiness check failed: Redis - {e}")

    try:
        # PgVector check
        async with database_manager.get_session() as db:
            pgvector_status = await system_initialization_service.run_all(db)
    except Exception as e:
        logger.error(f"Readiness check failed: PgVector - {e}")

    overall_status = "ready"
    if db_status != "ok" or redis_status != "ok" or not all(pgvector_status.values()):
        overall_status = "not_ready"

    return {
        "status": overall_status,
        "database": db_status,
        "redis": redis_status,
        "pgvector": pgvector_status
    }


@app.get("/system/cache-health", include_in_schema=False)
async def cache_health():
    """Cache health check endpoint"""
    try:
        pong = await cache_service.ping()
        return {"ok": bool(pong)}
    except Exception as e:
        logger.error("Cache health check failed", error=str(e))
        return {"ok": False, "error": str(e)}

# Import routers
from app.api.v1 import (
    math, english, users, mcp, dashboard, answers, pdf, scheduler, 
    analytics, sample_data, system_init, english_rag, math_rag, 
    llm_management, vector_management, monitoring, assess, system, question_generation, rag, llm_enhanced, mcp_enhanced, database_enhanced
)
from app.api.v1 import mcp_monitoring, mcp_demo

# Add routers
app.include_router(math.router)
app.include_router(english.router)
app.include_router(english_rag.router)  # English RAG API
app.include_router(math_rag.router)     # Math RAG API
app.include_router(llm_management.router)  # LLM Management API
app.include_router(vector_management.router)  # Vector Management API
app.include_router(monitoring.router)   # Monitoring API
app.include_router(users.router)
app.include_router(mcp.router)
app.include_router(dashboard.router)
app.include_router(answers.router)
app.include_router(pdf.router)
app.include_router(scheduler.router)
app.include_router(analytics.router)
app.include_router(sample_data.router)
app.include_router(system_init.router)
app.include_router(assess.router) # New assess router
app.include_router(system.router) # System API (health checks)
app.include_router(question_generation.router) # Question Generation API
app.include_router(rag.router) # RAG API
app.include_router(llm_enhanced.router) # Enhanced LLM API
app.include_router(mcp_enhanced.router) # Enhanced MCP API
app.include_router(database_enhanced.router) # Enhanced Database API
app.include_router(mcp_monitoring.router) # MCP Monitoring API
app.include_router(mcp_demo.router) # MCP Demo API

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint"""
    return {
        "message": "Adaptive Question System API",
        "version": settings.version,
        "environment": settings.environment.value,
        "docs": "/docs" if settings.api_docs_enabled else "Documentation disabled",
        "health": "/health",
        "metrics": "/api/v1/monitoring/metrics" if settings.prometheus_enabled else "Metrics disabled"
    }

# Serve static files for the frontend
# IMPORTANT: The 'directory' path should point to your Angular project's 'dist' folder  
# after you build it (e.g., by running 'ng build --configuration production' in frontend)
# The default Angular build output is usually 'dist/<project-name>/'

# Static serving temporarily disabled for testing
# Uncomment when ready to serve frontend:
# app.mount(
#     "/app",
#     StaticFiles(directory="../frontend/dist/adaptive-question-system-frontend", html=True),
#     name="frontend_app"
# )

@app.get("/config", include_in_schema=False)
async def config_info():
    """Configuration information (non-sensitive)"""
    return {
        "environment": settings.environment.value,
        "version": settings.version,
        "debug": settings.debug,
        "api_docs_enabled": settings.api_docs_enabled,
        "prometheus_enabled": settings.prometheus_enabled,
        "rate_limit_enabled": settings.rate_limit_enabled,
        "storage_backend": settings.storage_backend.value,
        "content_moderation_enabled": settings.content_moderation_enabled,
        "cost_monitoring_enabled": settings.content_monitoring_enabled,
        "llm_providers_available": len(settings.llm_providers_available),
        "cors_origins_count": len(settings.cors_origins_list),
    }