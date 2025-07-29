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
)
from prometheus_fastapi_instrumentator import Instrumentator
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

Instrumentator().instrument(app).expose(app)

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

@app.get("/health")
def health_check():
    """Sistem sağlık kontrolü"""
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
    
    # Neo4j health check
    if settings.USE_NEO4J:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(settings.NEO4J_URI, 
                                        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            health_status["services"]["neo4j"] = "healthy"
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

@app.get("/metrics")
def get_metrics():
    """Prometheus metrikleri"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

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