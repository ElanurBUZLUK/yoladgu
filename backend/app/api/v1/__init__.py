"""
API v1 router configuration.
"""

from fastapi import APIRouter

from app.api.v1 import (
    auth,
    recommend,
    generate,
    math_next,
    profile,
    admin,
    vector,
    vector_advanced,
    langchain_rag,
    advanced_rag,
    recommendations,
    hybrid_search,
    math_recommendations,
    search_orchestration,
    math_rag,
    math_adaptive,
    english_adaptive,
)

api_router = APIRouter(prefix="/v1")

# Authentication & user related
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(profile.router, prefix="/profile", tags=["Profile"])
api_router.include_router(admin.router, prefix="/admin", tags=["Administration"])

# Core recommendation & generation
api_router.include_router(recommend.router, prefix="/recommend", tags=["Recommendations"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["Recommendations"])
api_router.include_router(generate.router, prefix="/generate", tags=["Generation"])

# Search related endpoints
api_router.include_router(search_orchestration.router, prefix="/search", tags=["Search"])
api_router.include_router(hybrid_search.router, prefix="/search/hybrid", tags=["Search"])

# Vector / RAG related
api_router.include_router(vector.router, prefix="/vector", tags=["Vector"])
api_router.include_router(vector_advanced.router, prefix="/vector/advanced", tags=["Advanced RAG & Vector"])
api_router.include_router(langchain_rag.router, prefix="/langchain", tags=["Advanced RAG & Vector"])
api_router.include_router(advanced_rag.router, prefix="/advanced-rag", tags=["Advanced RAG & Vector"])

# Math related endpoints
api_router.include_router(math_next.router, prefix="/math", tags=["Math"])
api_router.include_router(math_recommendations.router, prefix="/math", tags=["Math"])
api_router.include_router(math_adaptive.router, prefix="/math/adaptive", tags=["Math"])
api_router.include_router(math_rag.router, prefix="/math/rag", tags=["Math"])

# English related endpoints
api_router.include_router(english_adaptive.router, prefix="/english/adaptive", tags=["English"])
