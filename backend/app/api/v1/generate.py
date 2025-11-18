"""
Question generation endpoints.
"""

from typing import List, Dict, Any, Optional
import logging

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import (
    Attempt,
    MathItem as MathItemDB,
    EnglishItem as EnglishItemDB,
)
from app.models.request import GenerateMathRequest, GenerateEnglishRequest
from app.models.response import (
    GenerateMathResponse,
    GenerateEnglishResponse,
    MathItem,
    ErrorResponse,
)
from app.services.math_generation_service import math_generation_service
from app.services.english_generation_service import english_generation_service
from app.services.adaptive_generation_service import adaptive_generation_service


router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# 1) Standart matematik soru üretimi
# ------------------------------------------------------------

@router.post(
    "/math",
    response_model=GenerateMathResponse,
    responses={400: {"model": ErrorResponse}},
)
async def generate_math_question(
    request: GenerateMathRequest,
    token: str = Depends(security),
):
    """
    Standart (adaptif olmayan) matematik soru üretimi.

    Not:
    - Zorluk / template seçimi request içindeki bilgilere göre yapılır.
    - Burada DB'ye kaydetmek zorunlu değil, istersen ekleyebiliriz.
    """

    try:
        question_data = math_generation_service.generate_question(
            template_id=request.template_id,
            params_hint=request.params_hint,
            target_difficulty=request.target_difficulty,
            language=request.language or "tr",
            rationale_required=request.rationale_required,
        )
    except Exception as e:
        logger.error(f"Math question generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Math generation failed: {str(e)}",
        )

    item = MathItem(
        stem=question_data["item"]["stem"],
        choices=question_data["item"].get("choices"),
        answer_key=question_data["item"]["answer_key"],
        solution=question_data["item"].get("solution"),
        skills=question_data["item"]["skills"],
        bloom_level=question_data["item"]["bloom_level"],
        difficulty_estimate=question_data["item"]["difficulty_estimate"],
        qa_checks=question_data["item"]["qa_checks"],
    )

    return GenerateMathResponse(
        item=item,
        template_id=request.template_id,
        raw_model_output=question_data.get("raw_model_output"),
    )


# ------------------------------------------------------------
# 2) Standart English cloze soru üretimi
# ------------------------------------------------------------

@router.post(
    "/en_cloze",
    response_model=GenerateEnglishResponse,
    responses={400: {"model": ErrorResponse}},
)
async def generate_english_cloze(
    request: GenerateEnglishRequest,
    token: str = Depends(security),
):
    """
    Standart (adaptif olmayan) İngilizce cloze / boşluk doldurma soru üretimi.
    """

    try:
        # Buradaki fonksiyon ismi projendeki english_generation_service'e göre
        # değişebilir. generate_question / generate_cloze_item vs.
        question_data = english_generation_service.generate_cloze_item(
            template_id=request.template_id,
            difficulty_hint=request.target_difficulty,
            language=request.language or "en",
            tags=request.tags,  # varsa
        )
    except Exception as e:
        logger.error(f"English cloze generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"English cloze generation failed: {str(e)}",
        )

    # GenerateEnglishResponse içindeki alanlar projenin gerçek modeline göre
    # değişebilir; ben burada generic bir gövde bıraktım:
    return GenerateEnglishResponse(
        item=question_data["item"],
        template_id=request.template_id,
        raw_model_output=question_data.get("raw_model_output"),
    )


# ------------------------------------------------------------
# 3) ADAPTİF Math – zayıf skill'lere göre soru üretimi
# ------------------------------------------------------------

class AdaptiveMathRequest(BaseModel):
    """
    Zayıf skill'lere göre otomatik math soru üretme isteği.
    """
    user_id: str
    n_questions: int = 1
    days: int = 30                      # Kaç gün geriye bakılacak
    min_attempts_per_skill: int = 3     # Bir skill'e zayıf demek için min. deneme
    template_ids: Optional[List[str]] = None  # İstersen sadece belirli template'ler
    target_difficulty: Optional[float] = None # Verirsen otomatik hesap yerine bunu kullan
    persist: bool = False               # Üretilen soruları MathItem tablosuna kaydet


class AdaptiveMathQuestion(BaseModel):
    item: MathItem
    target_skill: Optional[str]
    template_id: str
    difficulty_used: Optional[float]
    saved_item_id: Optional[str] = None   # DB'ye kaydedildiyse MathItem.id


class AdaptiveMathResponse(BaseModel):
    user_id: str
    questions: List[AdaptiveMathQuestion]
    weak_skills: List[Dict[str, Any]]   # {"skill": "..", "count": .., "accuracy": ..}


@router.post("/math/adaptive", response_model=AdaptiveMathResponse)
async def generate_adaptive_math_questions(
    request: AdaptiveMathRequest,
    session: AsyncSession = Depends(get_db),
    token: str = Depends(security),
):
    """
    Öğrencinin geçmiş performansına (attempt'lerine) bakarak
    zayıf skill'leri bulur ve o skill'leri hedefleyen yeni math soruları üretir.
    İsteğe bağlı olarak soruları DB'ye kaydeder ve RAG indeksine gönderebilir.
    """

    try:
        result = await adaptive_generation_service.generate_adaptive_math(
            session=session,
            user_id=request.user_id,
            n_questions=request.n_questions,
            days=request.days,
            min_attempts_per_skill=request.min_attempts_per_skill,
            template_ids=request.template_ids,
            target_difficulty=request.target_difficulty,
            persist=request.persist,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Adaptive math generation failed for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during adaptive math generation: {str(e)}",
        )

    return result


# ------------------------------------------------------------
# 4) ADAPTİF English cloze – zayıf error_tag/skill'lere göre
# ------------------------------------------------------------

class AdaptiveEnglishRequest(BaseModel):
    """
    Zayıf error_tags (skill gibi) için adaptif cloze soru üretme isteği.
    """
    user_id: str
    n_questions: int = 1
    days: int = 30
    min_attempts_per_skill: int = 3
    template_ids: Optional[List[str]] = None
    target_difficulty: Optional[float] = None
    persist: bool = False


class AdaptiveEnglishQuestion(BaseModel):
    item: Dict[str, Any]                # Cloze item'ı generic dict olarak döndürüyoruz
    target_skill: Optional[str]
    template_id: Optional[str]
    difficulty_used: Optional[float]
    saved_item_id: Optional[str] = None


class AdaptiveEnglishResponse(BaseModel):
    user_id: str
    questions: List[AdaptiveEnglishQuestion]
    weak_skills: List[Dict[str, Any]]


@router.post("/en_cloze/adaptive", response_model=AdaptiveEnglishResponse)
async def generate_adaptive_english_cloze(
    request: AdaptiveEnglishRequest,
    session: AsyncSession = Depends(get_db),
    token: str = Depends(security),
):
    """
    Öğrencinin English attempt'lerine bakarak zayıf error_tags (skill'ler) için
    adaptif cloze soruları üretir. İsteğe bağlı olarak soruları DB'ye kaydeder
    ve RAG indeksine gönderir.
    """

    try:
        result = await adaptive_generation_service.generate_adaptive_english(
            session=session,
            user_id=request.user_id,
            n_questions=request.n_questions,
            days=request.days,
            min_attempts_per_skill=request.min_attempts_per_skill,
            template_ids=request.template_ids,
            target_difficulty=request.target_difficulty,
            persist=request.persist,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Adaptive English generation failed for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during adaptive English generation: {str(e)}",
        )

    return result
