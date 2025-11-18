"""Adaptive math question API endpoint."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.request import AdaptiveMathRequest
from app.models.response import MathQuestionResponse
from app.services.adaptive_math_service import adaptive_math_service


router = APIRouter()
security = HTTPBearer()


@router.post("/adaptive-question", response_model=MathQuestionResponse)
async def get_adaptive_math_question(
    request: AdaptiveMathRequest,
    token: str = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """Generate an adaptive math question using the student's profile."""

    math_question = await adaptive_math_service.generate_adaptive_math_question(
        session=db,
        user_id=request.user_id,
        skill_focus=request.skill_focus,
        max_difficulty_shift=request.max_difficulty_shift,
    )

    if not math_question:
        raise HTTPException(
            status_code=500,
            detail="Adaptif math sorusu üretilemedi.",
        )

    return math_question
