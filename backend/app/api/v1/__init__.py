"""
API v1 router configuration.
"""

from fastapi import APIRouter

from app.api.v1 import auth, recommend, generate, profile, admin, vector, recommendations

api_router = APIRouter(prefix="/v1")

# Include all routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(recommend.router, prefix="/recommend", tags=["Recommendations"])
api_router.include_router(generate.router, prefix="/generate", tags=["Generation"])
api_router.include_router(profile.router, prefix="/profile", tags=["Profile"])
api_router.include_router(vector.router, prefix="/vector", tags=["Vector Operations"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["Error-Aware Recommendations"])
api_router.include_router(admin.router, prefix="/admin", tags=["Administration"])