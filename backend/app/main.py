from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import connect_to_db, close_db_connection
from app.core.cache import cache_service

# Import middleware
from app.middleware.error_handler import error_handler
from app.middleware.rate_limiter import rate_limiter
from app.middleware.logging import logging_middleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_db()
    await cache_service.connect()
    
    # Background scheduler'ı başlat
    from app.services.background_scheduler import background_scheduler
    await background_scheduler.start()
    
    yield
    
    # Shutdown
    await background_scheduler.stop()
    await close_db_connection()
    await cache_service.disconnect()


app = FastAPI(
    title="Adaptive Question System API",
    version="1.0.0",
    description="AI-powered adaptive learning system for mathematics and English",
    openapi_url="/openapi.json" if settings.environment != "production" else None,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan
)

# Add middleware (order matters - last added is first executed)
# Error handling middleware
app.middleware("http")(error_handler)

# Rate limiting middleware
app.middleware("http")(rate_limiter)

# Logging middleware
app.middleware("http")(logging_middleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

@app.get("/health", include_in_schema=False)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "version": settings.version,
        "environment": settings.environment
    }

# Import routers
from app.api.v1 import math, english, users, mcp, dashboard, answers, pdf, scheduler, analytics, sample_data, system_init
from app.api.v1 import english_rag

# Add routers
app.include_router(math.router)
app.include_router(english.router)
app.include_router(english_rag.router)  # New RAG API
app.include_router(users.router)
app.include_router(mcp.router)
app.include_router(dashboard.router)
app.include_router(answers.router)
app.include_router(pdf.router)
app.include_router(scheduler.router)
app.include_router(analytics.router)
app.include_router(sample_data.router)
app.include_router(system_init.router)

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint"""
    return {
        "message": "Adaptive Question System API",
        "version": settings.version,
        "docs": "/docs" if settings.environment != "production" else "Documentation disabled in production"
    }