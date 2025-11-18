"""
Question generation endpoints.

Bu dosya:
- Standart matematik soru üretimi
- Standart İngilizce cloze soru üretimi
- Öğrencinin zayıf olduğu skill'lere göre ADAPTİF math soru üretimi
- Öğrencinin zayıf olduğu error_tag/skill'lere göre ADAPTİF English cloze üretimi
- İsteğe bağlı: üretilen soruları DB'ye kaydetme ve RAG indeksine gönderme
işlevlerini içerir.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

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
from app.services.search_service import search_service  # RAG için
# Eğer ayrı vector_service varsa:
# from app.services.vector_service import vector_service

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

    # 1) Kullanıcının son N gün içindeki math attempt'lerini çek
    now = datetime.utcnow()
    date_from = now - timedelta(days=request.days)

    result = await session.execute(
        select(Attempt).where(
            Attempt.user_id == request.user_id,
            Attempt.item_type == "math",
            Attempt.created_at >= date_from,
        )
    )
    attempts = result.scalars().all()

    if not attempts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bu kullanıcı için belirtilen süre içinde math attempt bulunamadı.",
        )

    # 2) Attempt'lere göre skill bazlı istatistik çıkarmak için ilgili MathItem'ları çek
    math_ids = list({a.item_id for a in attempts if a.item_id})
    if not math_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Attempt'lerde geçerli math item_id bulunamadı.",
        )

    result_items = await session.execute(
        select(MathItemDB).where(MathItemDB.id.in_(math_ids))
    )
    math_items_by_id: Dict[str, MathItemDB] = {
        m.id: m for m in result_items.scalars().all()
    }

    # Skill istatistiklerini hesapla
    skill_stats: Dict[str, Dict[str, float]] = {}
    for a in attempts:
        m = math_items_by_id.get(a.item_id)
        if not m or not m.skills:
            continue

        for s in m.skills:
            if s not in skill_stats:
                skill_stats[s] = {"count": 0, "correct": 0}
            skill_stats[s]["count"] += 1
            if a.correct:
                skill_stats[s]["correct"] += 1

    weak_skills_info: List[Dict[str, Any]] = []
    for skill, stats in skill_stats.items():
        cnt = stats["count"]
        if cnt < request.min_attempts_per_skill:
            continue
        acc = stats["correct"] / cnt if cnt > 0 else 0.0
        weak_skills_info.append(
            {
                "skill": skill,
                "count": cnt,
                "accuracy": acc,
            }
        )

    if not weak_skills_info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Yeterli deneme sayısına sahip zayıf skill bulunamadı.",
        )

    # En zayıftan en güçlüye sırala
    weak_skills_info.sort(key=lambda x: x["accuracy"])

    # 3) Kullanılabilir math template'lerini al
    available_templates = math_generation_service.get_available_templates()
    if request.template_ids:
        allowed = set(request.template_ids)
        available_templates = [
            t for t in available_templates
            if t["template_id"] in allowed
        ]

    if not available_templates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Kullanılabilir math template bulunamadı.",
        )

    questions: List[AdaptiveMathQuestion] = []

    for i in range(request.n_questions):
        target_info = weak_skills_info[i % len(weak_skills_info)]
        target_skill = target_info["skill"]
        skill_acc = target_info["accuracy"]

        # Bu skill'i içeren template'leri bul
        skill_templates = [
            t for t in available_templates
            if target_skill in (t.get("skills") or [])
        ]
        if not skill_templates:
            skill_templates = available_templates

        chosen_template = skill_templates[0]  # Şimdilik deterministic
        template_id = chosen_template["template_id"]

        # Difficulty seçimi
        if request.target_difficulty is not None:
            difficulty_used = request.target_difficulty
        else:
            if skill_acc < 0.4:
                difficulty_used = -0.5   # kolay
            elif skill_acc < 0.7:
                difficulty_used = 0.0    # orta
            else:
                difficulty_used = 0.5    # biraz zor

        # 4) Math generation service ile soru üret
        try:
            question_data = math_generation_service.generate_question(
                template_id=template_id,
                params_hint=None,
                target_difficulty=difficulty_used,
                language="tr",
                rationale_required=False,
            )
        except Exception as e:
            logger.error(
                f"Adaptive math generation failed for user {request.user_id}, "
                f"skill={target_skill}, template={template_id}: {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Adaptive math generation failed: {str(e)}",
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

        saved_item_id: Optional[str] = None

        # 5) İsteğe bağlı: DB'ye kaydet + RAG indeksleme
        if request.persist:
            try:
                db_item = MathItemDB(
                    stem=item.stem,
                    choices=item.choices,
                    answer_key=item.answer_key,
                    solution=item.solution,
                    skills=item.skills,
                    bloom_level=item.bloom_level,
                    difficulty_a=difficulty_used,
                    difficulty_b=item.difficulty_estimate,
                    source="adaptive",
                )
                session.add(db_item)
                await session.commit()
                await session.refresh(db_item)
                saved_item_id = str(db_item.id)

                # 🔹 RAG indeksleme – projendeki fonksiyon ismine göre uyarlayabilirsin
                await search_service.index_math_item(db_item)
                # veya: await vector_service.index_math_item(db_item)

            except Exception as e:
                logger.error(f"Failed to persist/index adaptive math item: {e}")

        questions.append(
            AdaptiveMathQuestion(
                item=item,
                target_skill=target_skill,
                template_id=template_id,
                difficulty_used=difficulty_used,
                saved_item_id=saved_item_id,
            )
        )

    return AdaptiveMathResponse(
        user_id=request.user_id,
        questions=questions,
        weak_skills=weak_skills_info,
    )


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

    now = datetime.utcnow()
    date_from = now - timedelta(days=request.days)

    # 1) Bu kullanıcının son N gündeki English attempt'leri
    result = await session.execute(
        select(Attempt).where(
            Attempt.user_id == request.user_id,
            Attempt.item_type == "en",
            Attempt.created_at >= date_from,
        )
    )
    attempts = result.scalars().all()

    if not attempts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bu kullanıcı için belirtilen süre içinde English attempt bulunamadı.",
        )

    # 2) Attempt'lerde geçen EnglishItem'ları çek
    en_ids = list({a.item_id for a in attempts if a.item_id})
    if not en_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Attempt'lerde geçerli English item_id bulunamadı.",
        )

    result_items = await session.execute(
        select(EnglishItemDB).where(EnglishItemDB.id.in_(en_ids))
    )
    en_items_by_id: Dict[str, EnglishItemDB] = {
        e.id: e for e in result_items.scalars().all()
    }

    # 3) error_tags'i skill gibi kullanarak istatistik çıkar
    skill_stats: Dict[str, Dict[str, float]] = {}

    for a in attempts:
        e = en_items_by_id.get(a.item_id)
        if not e or not e.error_tags:
            continue

        for s in e.error_tags:
            if s not in skill_stats:
                skill_stats[s] = {"count": 0, "correct": 0}
            skill_stats[s]["count"] += 1
            if a.correct:
                skill_stats[s]["correct"] += 1

    weak_skills_info: List[Dict[str, Any]] = []
    for skill, stats in skill_stats.items():
        cnt = stats["count"]
        if cnt < request.min_attempts_per_skill:
            continue
        acc = stats["correct"] / cnt if cnt > 0 else 0.0
        weak_skills_info.append(
            {
                "skill": skill,
                "count": cnt,
                "accuracy": acc,
            }
        )

    if not weak_skills_info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Yeterli deneme sayısına sahip zayıf English skill (error_tag) bulunamadı.",
        )

    weak_skills_info.sort(key=lambda x: x["accuracy"])

    # 4) Kullanılabilir cloze template'leri al
    try:
        available_templates = english_generation_service.get_available_templates()
    except AttributeError:
        # Eğer böyle bir fonksiyon yoksa, templates'i boş bırakıyoruz -> template_id None olur
        available_templates = []

    questions: List[AdaptiveEnglishQuestion] = []

    for i in range(request.n_questions):
        target_info = weak_skills_info[i % len(weak_skills_info)]
        target_skill = target_info["skill"]
        skill_acc = target_info["accuracy"]

        skill_templates = (
            [
                t for t in available_templates
                if target_skill in (
                    t.get("error_tags") or t.get("skills") or []
                )
            ]
            if available_templates
            else []
        )

        chosen_template_id: Optional[str] = None
        if skill_templates:
            chosen_template_id = skill_templates[0]["template_id"]

        # Difficulty seçimi
        if request.target_difficulty is not None:
            difficulty_used = request.target_difficulty
        else:
            if skill_acc < 0.4:
                difficulty_used = -0.5
            elif skill_acc < 0.7:
                difficulty_used = 0.0
            else:
                difficulty_used = 0.5

        # 5) English generation service ile cloze soru üret
        try:
            question_data = english_generation_service.generate_cloze_item(
                template_id=chosen_template_id,
                target_error_tags=[target_skill],
                difficulty_hint=difficulty_used,
                language="en",
            )
        except Exception as e:
            logger.error(
                f"Adaptive English generation failed for user {request.user_id}, "
                f"skill={target_skill}, template={chosen_template_id}: {e}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Adaptive English generation failed: {str(e)}",
            )

        item_dict = question_data["item"]

        saved_item_id: Optional[str] = None

        # 6) İsteğe bağlı: DB'ye kaydet + RAG indeksleme
        if request.persist:
            try:
                db_item = EnglishItemDB(
                    text=item_dict.get("text"),
                    options=item_dict.get("options"),
                    answer_key=item_dict.get("answer_key"),
                    error_tags=item_dict.get("error_tags") or [target_skill],
                    source="adaptive",
                    difficulty_estimate=difficulty_used,
                )
                session.add(db_item)
                await session.commit()
                await session.refresh(db_item)
                saved_item_id = str(db_item.id)

                # RAG indeksleme
                await search_service.index_english_item(db_item)
                # veya: await vector_service.index_english_item(db_item)

            except Exception as e:
                logger.error(f"Failed to persist/index adaptive English item: {e}")

        questions.append(
            AdaptiveEnglishQuestion(
                item=item_dict,
                target_skill=target_skill,
                template_id=chosen_template_id,
                difficulty_used=difficulty_used,
                saved_item_id=saved_item_id,
            )
        )

    return AdaptiveEnglishResponse(
        user_id=request.user_id,
        questions=questions,
        weak_skills=weak_skills_info,
    )
