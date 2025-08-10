from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
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
    from app.api.v1.endpoints import quiz
    _HAS_QUIZ = True
except Exception:
    _HAS_QUIZ = False
try:
    from app.api.v1.endpoints import ml
    _HAS_ML = True
except Exception:
    _HAS_ML = False

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/health")
def health():
    return {"status": "ok"}
