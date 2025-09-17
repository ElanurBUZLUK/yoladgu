"""
API v1 router configuration.
"""

from fastapi import APIRouter

from app.api.v1 import auth, recommend, generate, profile, admin, vector, recommendations, vector_advanced, langchain_rag, advanced_rag, hybrid_search, math_recommendations

api_router = APIRouter(prefix="/v1")

# Include all routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(recommend.router, prefix="/recommend", tags=["Recommendations"])
api_router.include_router(generate.router, prefix="/generate", tags=["Generation"])
api_router.include_router(profile.router, prefix="/profile", tags=["Profile"])
api_router.include_router(vector.router, prefix="/vector", tags=["Vector Operations"])
api_router.include_router(vector_advanced.router, prefix="/vector", tags=["Advanced Vector Operations"])
api_router.include_router(langchain_rag.router, prefix="/langchain", tags=["LangChain RAG Operations"])
api_router.include_router(advanced_rag.router, prefix="/advanced-rag", tags=["Advanced RAG System"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["Error-Aware Recommendations"])
api_router.include_router(hybrid_search.router, prefix="/hybrid-search", tags=["Hybrid Search (BM25 + E5 + RRF)"])
api_router.include_router(math_recommendations.router, prefix="/math", tags=["Math Recommendations (IRT + Multi-Skill Elo)"])
api_router.include_router(admin.router, prefix="/admin", tags=["Administration"])