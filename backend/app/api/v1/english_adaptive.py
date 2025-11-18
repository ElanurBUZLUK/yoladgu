"""Adaptive English question API endpoint."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.request import AdaptiveEnglishRequest
from app.models.response import EnglishQuestionResponse
from app.services.adaptive_english_service import adaptive_english_service

router = APIRouter()
security = HTTPBearer()


@router.post("/adaptive-question", response_model=EnglishQuestionResponse)
async def get_adaptive_english_question(
    request: AdaptiveEnglishRequest,
    token: str = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Generate an adaptive English question using student profile data."""

    english_question = await adaptive_english_service.generate_adaptive_english_question(
        session=db,
        user_id=request.user_id,
        skill_focus=request.skill_focus,
        max_difficulty_shift=request.max_difficulty_shift,
    )

    if not english_question:
        raise HTTPException(
            status_code=500,
            detail="Adaptif İngilizce sorusu üretilemedi.",
        )

    return english_question
