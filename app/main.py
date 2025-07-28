from fastapi import FastAPI, Depends
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
def health(db: Session = Depends(get_db)):
    # Redis
    try:
        r = redis.Redis.from_url(settings.redis_url)
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    # DB
    try:
        db.execute("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False
    # Neo4j
    try:
        driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("RETURN 1")
        neo4j_ok = True
    except Exception:
        neo4j_ok = False
    return {"redis": redis_ok, "db": db_ok, "neo4j": neo4j_ok, "status": redis_ok and db_ok and neo4j_ok} 