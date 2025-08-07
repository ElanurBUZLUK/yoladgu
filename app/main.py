"""
Main Application Entry Point
Ana uygulama giriş noktası
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

import structlog
from app.core.config import settings
from app.api.v1.endpoints import (
    auth, 
    users, 
    analytics, 
    questions, 
    quiz_sessions, 
    students, 
    vectors, 
    faiss, 
    recommendations, 
    skill_graph
)
from app.core.dependencies import init_vector_index

logger = structlog.get_logger()

# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Yoladgu - AI-Powered Learning Platform",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ============================================================================
# ROUTES
# ============================================================================

# Core API routes
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(users.router, prefix=settings.API_V1_STR)
app.include_router(students.router, prefix=settings.API_V1_STR)

# Content routes
app.include_router(questions.router, prefix=settings.API_V1_STR)
app.include_router(quiz_sessions.router, prefix=settings.API_V1_STR)

# Analytics routes
app.include_router(analytics.router, prefix=settings.API_V1_STR)

# ML/AI routes
app.include_router(recommendations.router, prefix=settings.API_V1_STR)
app.include_router(vectors.router, prefix=settings.API_V1_STR)
app.include_router(faiss.router, prefix=settings.API_V1_STR)
app.include_router(skill_graph.router, prefix=settings.API_V1_STR)

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }

# ============================================================================
# ERROR HANDLING
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error("unhandled_exception", 
                path=request.url.path,
                method=request.method,
                error=str(exc))
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("application_started", 
                project_name=settings.PROJECT_NAME,
                version=settings.VERSION)
    
    # FAISS index'ini başlat
    init_vector_index()

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("application_shutdown")
