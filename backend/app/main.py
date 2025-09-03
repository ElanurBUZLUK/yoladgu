"""
FastAPI application entry point for Adaptive Question Recommendation System.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.core.config import settings
from app.core.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from app.api.v1 import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    print("🚀 Starting Adaptive Question Recommendation System...")
    
    # Initialize search and vector services
    try:
        from app.services.search_service import search_service
        from app.services.vector_service import vector_service
        
        print("📚 Initializing search services...")
        
        # Initialize Elasticsearch index
        try:
            await search_service.initialize_index()
            print("✅ Elasticsearch index initialized successfully")
        except Exception as es_error:
            print(f"⚠️ Elasticsearch initialization failed: {es_error}")
            print("   Search functionality may be limited")
        
        # Vector service is lazy-initialized, no explicit startup needed
        print("✅ Vector service ready (lazy initialization)")
        
    except Exception as e:
        print(f"⚠️ Service initialization warning: {e}")
        print("   Some features may not be available")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Adaptive Question Recommendation System...")


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Adaptive Question Recommendation System",
        description="AI-powered personalized question recommendation system for Math and English",
        version="1.0.0",
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
        servers=[
            {"url": settings.API_BASE_URL, "description": "API Server"}
        ]
    )
    
    # Add middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    
    # Include routers
    app.include_router(api_router, prefix="/api")
    
    return app


app = create_application()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "up",
        "version": "1.0.0",
        "service": "adaptive-question-system"
    }


@app.get("/version")
async def version_info():
    """Version information endpoint."""
    return {
        "app_version": "1.0.0",
        "model_versions": {
            "retrieval": "retr_v4.1",
            "rerank": "ce_int8_v2", 
            "llm": "llama3-70b-q4",
            "bandit": "linucb_v1.3"
        }
    }