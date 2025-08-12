from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import init_logging
from app.core.errors import (
    http_exception_handler,
    validation_exception_handler,
    unhandled_exception_handler,
)
from app.api_auth import router as auth_router
from app.api_vectors import router as vectors_router
from app.api_index_admin import router as index_router
from app.api_recommend import router as recommend_router
from app.api_teacher import router as teacher_router
from app.api_student import router as student_router
from app.api_questions import router as questions_router
from app.api_admin import router as admin_router
try:
    from app.api.v1.endpoints import mcp_bridge
    _HAS_MCP = True
except Exception:
    _HAS_MCP = False
try:
    from app.api.v1.endpoints import embeddings
    _HAS_EMBED = True
except Exception:
    _HAS_EMBED = False
try:
    from app.api.v1.endpoints import adaptive
    _HAS_ADAPTIVE = True
except Exception:
    _HAS_ADAPTIVE = False
try:
    from app.api.v1.endpoints import quiz
    _HAS_QUIZ = True
except Exception:
    _HAS_QUIZ = False
try:
    from app.api.v1.endpoints import ml
    _HAS_ML = True
except Exception:
    _HAS_ML = False
try:
    from app.api.v1.endpoints import retrieval
    _HAS_RETR = True
except Exception:
    _HAS_RETR = False

init_logging(settings.ENV)
app = FastAPI(title=settings.PROJECT_NAME)

allowed_origins = [
    o.strip() for o in (getattr(settings, "CORS_ALLOW_ORIGINS", "*") or "*").split(",")
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(vectors_router, prefix=settings.API_V1_STR)
app.include_router(index_router, prefix=settings.API_V1_STR)
app.include_router(recommend_router, prefix=settings.API_V1_STR)
app.include_router(teacher_router, prefix=settings.API_V1_STR)
app.include_router(student_router, prefix=settings.API_V1_STR)
app.include_router(questions_router, prefix=settings.API_V1_STR)
app.include_router(admin_router, prefix=settings.API_V1_STR)
if _HAS_MCP:
    app.include_router(mcp_bridge.router, prefix=settings.API_V1_STR)
if _HAS_QUIZ:
    app.include_router(quiz.router, prefix=settings.API_V1_STR)
if _HAS_ML:
    app.include_router(ml.router, prefix=settings.API_V1_STR)
if _HAS_ADAPTIVE:
    app.include_router(adaptive.router, prefix=settings.API_V1_STR)
if _HAS_EMBED:
    app.include_router(embeddings.router, prefix=settings.API_V1_STR)
if _HAS_RETR:
    app.include_router(retrieval.router, prefix=settings.API_V1_STR)

# Prometheus metrics
try:
    from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore
    ins = Instrumentator()
    ins.instrument(app).expose(app)
except Exception:
    pass

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness probe checking external dependencies."""
    from app.core.db import SessionLocal
    from sqlalchemy import text
    import redis
    ok = {"db": False, "redis": False}
    # DB
    try:
        async with SessionLocal() as s:  # type: ignore
            await s.execute(text("SELECT 1"))
        ok["db"] = True
    except Exception:
        ok["db"] = False
    # Redis
    try:
        r = redis.Redis.from_url(settings.REDIS_URL)
        r.ping()
        ok["redis"] = True
    except Exception:
        ok["redis"] = False
    status = "ok" if all(ok.values()) else "degraded"
    return {"status": status, **ok}
