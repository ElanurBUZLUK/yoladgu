"""
Math next-question endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.response import MathQuestionResponse
from app.services.math_next_question_service import math_next_question_service

router = APIRouter()
security = HTTPBearer()


@router.get("/next-question", response_model=MathQuestionResponse)
async def get_next_math_question(
    user_id: str = Query(..., description="User ID"),
    token: str = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """
    Kullanıcının profilini kullanarak bir sonraki math sorusunu döner.
    Şimdilik sadece var olan MathItem'lardan seçim yapar.
    """
    item = await math_next_question_service.get_next_question(
        session=db,
        user_id=user_id,
    )

    if not item:
        raise HTTPException(
            status_code=404,
            detail="Uygun math sorusu bulunamadı.",
        )

    return MathQuestionResponse(
        item_id=str(item.id),
        question_text=item.stem,
        choices=item.choices or None,
        correct_answer=item.answer_key,
        solution_steps=item.solution,
        skills=item.skills or [],
    )
