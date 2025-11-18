"""
User profile endpoints.
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Path
from fastapi.security import HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.db.models import Feedback, Attempt, MathItem, EnglishItem
from app.db.repositories.item import math_item_repository, english_item_repository
from app.db.repositories.attempt import attempt_repository
from app.services.profile_service import profile_service
from app.services.irt_service import irt_service, error_profile_service
from app.services.bandit_service import bandit_service

from app.models.request import ProfileUpdateRequest, AttemptRequest, FeedbackRequest
from app.models.response import ProfileResponse, AttemptResponse, FeedbackResponse

router = APIRouter()
security = HTTPBearer()


@router.get("/{user_id}", response_model=ProfileResponse)
async def get_user_profile(
    user_id: str = Path(..., description="User ID"),
    token: str = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user profile including theta values and error profiles.

    Returns:
    - Current theta values for math and English
    - Error profiles by skill/topic
    - Learning preferences and segments
    - Recent performance statistics
    """
    # TODO: RBAC vs. ekleyebilirsin, şimdilik sadece profil getiriyoruz.

    profile = await profile_service.get_user_profile(
        session=db,
        user_id=user_id,
        use_cache=True,
    )

    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"User profile not found for user_id={user_id}",
        )

    # error_profile_math + error_profile_en -> tek dict
    error_profiles = {
        "math": profile.get("error_profile_math", {}) or {},
        "english": profile.get("error_profile_en", {}) or {},
    }

    return ProfileResponse(
        user_id=profile["user_id"],
        grade=profile.get("grade"),
        lang=profile.get("lang", "tr"),
        theta_math=profile.get("theta_math"),
        theta_en=profile.get("theta_en"),
        error_profiles=error_profiles,
        segments=profile.get("segments", []),
        preferences=profile.get("preferences", {}),
    )


@router.post("/update", response_model=dict)
async def update_user_profile(
    request: ProfileUpdateRequest,
    token: str = Depends(security),
):
    """
    Update user profile data.

    Note: Students can only update preferences, not theta values.
    Only teachers/services can update model parameters.
    """
    # TODO: Implement profile update
    # - Validate permissions (RBAC)
    # - Check which fields can be updated by role
    # - Update database
    # - Invalidate cache
    # - Return updated fields

    raise HTTPException(
        status_code=501,
        detail="Profile update not implemented yet",
    )


@router.post("/attempt", response_model=AttemptResponse)
async def record_attempt(
    request: AttemptRequest,
    token: str = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """
    Record a student's attempt at answering a question.

    Bu endpoint:
    1. Attempt'i veritabanına kaydeder
    2. IRT ile theta değerlerini günceller
    3. Yanlış ise error profile'ı günceller
    4. Reward bileşenlerini hesaplar
    5. Bandit'i bu reward ile günceller
    """

    # 1) Soru tipini ve item'ın meta bilgisini bul (math / english)
    item_type = None
    item_skills = []
    item_a = 1.0   # discrimination
    item_b = 0.0   # difficulty

    math_item = await math_item_repository.get(db, request.item_id)
    if math_item:
        item_type = "math"
        item_skills = math_item.skills or []
        item_a = math_item.difficulty_a
        item_b = math_item.difficulty_b
    else:
        english_item = await english_item_repository.get(db, request.item_id)
        if english_item:
            item_type = "en"
            # İngilizce tarafında error_tags skill olarak kabul ediliyor
            item_skills = english_item.error_tags or []
            # Şimdilik difficulty parametreleri için default değerler
            item_a = 1.0
            item_b = 0.0

    if item_type is None:
        raise HTTPException(
            status_code=404,
            detail=f"Item not found for id={request.item_id}",
        )

    # 2) Profil servisi ile attempt + IRT + error_profile güncelle
    update_result = await profile_service.update_profile_after_attempt(
        session=db,
        user_id=request.user_id,
        item_id=request.item_id,
        item_type=item_type,
        answer=request.answer,
        correct=request.correct,
        time_ms=request.time_ms,
        hints_used=request.hints_used,
        context=request.context,
        item_skills=item_skills,
        item_a=item_a,
        item_b=item_b,
    )

    if not update_result:
        raise HTTPException(
            status_code=500,
            detail="Failed to update profile after attempt",
        )

    updated_theta = update_result.get("updated_theta")
    reward_components = update_result.get("reward_components") or {}

    # 3) Bandit'i rewardlarla güncelle
    try:
        await bandit_service.update_bandit(
            session=db,
            user_id=request.user_id,
            arm_id=request.item_id,
            reward_components=reward_components,
            policy_id="linucb",
            item_skills=item_skills,
        )
    except Exception:
        # Bandit hatası sistemi bozmasın; loglama ekleyebilirsin
        pass

    # 4) API response
    return AttemptResponse(
        stored=True,
        updated_theta=updated_theta,
        reward_components=reward_components,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(
    request: FeedbackRequest,
    token: str = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """
    Record user feedback about questions or system.

    Feedback is used to:
    - Improve question quality
    - Adjust difficulty estimates
    - Identify problematic content
    - Enhance recommendation algorithms
    """

    # 1) Item tipini belirle (math / en)
    item_type: Optional[str] = None

    math_item = await math_item_repository.get(db, request.item_id)
    if math_item:
        item_type = "math"
    else:
        english_item = await english_item_repository.get(db, request.item_id)
        if english_item:
            item_type = "en"

    if item_type is None:
        raise HTTPException(
            status_code=404,
            detail=f"Item not found for id={request.item_id}",
        )

    # 2) Feedback kaydını oluştur
    feedback = Feedback(
        user_id=request.user_id,
        item_id=request.item_id,
        rating=request.rating,
        flags=request.flags or {},
        comment=request.comment,
        item_type=item_type,
    )

    db.add(feedback)
    await db.commit()

    return FeedbackResponse(stored=True)


@router.get("/{user_id}/history")
async def get_attempt_history(
    user_id: str = Path(..., description="User ID"),
    limit: int = 50,
    offset: int = 0,
    token: str = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user's attempt history with pagination + basic stats.

    Dönen JSON:
    {
        "user_id": "...",
        "total_attempts": 10,
        "accuracy_overall": 0.7,
        "by_item_type": {
            "math": {"count": 6, "accuracy": 0.66},
            "en": {"count": 4, "accuracy": 0.75}
        },
        "items": [...]
    }
    """

    result = await db.execute(
        select(Attempt)
        .where(Attempt.user_id == user_id)
        .order_by(getattr(Attempt, "created_at", Attempt.id).desc())
        .offset(offset)
        .limit(limit)
    )
    attempts = result.scalars().all()

    total = len(attempts)
    if total == 0:
        return {
            "user_id": user_id,
            "total_attempts": 0,
            "accuracy_overall": None,
            "by_item_type": {},
            "items": [],
        }

    correct_total = sum(1 for a in attempts if getattr(a, "correct", False))
    accuracy_overall = correct_total / total if total > 0 else None

    stats: Dict[str, Dict[str, float]] = {}
    for a in attempts:
        item_type = getattr(a, "item_type", "unknown") or "unknown"
        if item_type not in stats:
            stats[item_type] = {"count": 0, "correct": 0}
        stats[item_type]["count"] += 1
        if getattr(a, "correct", False):
            stats[item_type]["correct"] += 1

    by_item_type: Dict[str, Dict[str, float]] = {}
    for item_type, s in stats.items():
        cnt = s["count"]
        by_item_type[item_type] = {
            "count": cnt,
            "accuracy": s["correct"] / cnt if cnt > 0 else None,
        }

    items = []
    for a in attempts:
        created_at = getattr(a, "created_at", None)
        items.append(
            {
                "attempt_id": getattr(a, "id", None),
                "item_id": getattr(a, "item_id", None),
                "item_type": getattr(a, "item_type", None),
                "correct": getattr(a, "correct", None),
                "time_ms": getattr(a, "time_ms", None),
                "hints_used": getattr(a, "hints_used", None),
                "created_at": created_at.isoformat() if created_at else None,
            }
        )

    return {
        "user_id": user_id,
        "total_attempts": total,
        "accuracy_overall": accuracy_overall,
        "by_item_type": by_item_type,
        "items": items,
    }


@router.get("/{user_id}/analytics")
async def get_user_analytics(
    user_id: str = Path(..., description="User ID"),
    days: int = 30,
    token: str = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user learning analytics and progress over the last N days.
    """

    now = datetime.utcnow()
    date_from = now - timedelta(days=days)

    has_created_at = hasattr(Attempt, "created_at")

    if has_created_at:
        query = (
            select(Attempt)
            .where(
                Attempt.user_id == user_id,
                Attempt.created_at >= date_from,
            )
            .order_by(Attempt.created_at.asc())
        )
    else:
        query = select(Attempt).where(Attempt.user_id == user_id)

    result = await db.execute(query)
    attempts = result.scalars().all()

    total = len(attempts)
    if total == 0:
        return {
            "user_id": user_id,
            "from": date_from.isoformat(),
            "to": now.isoformat(),
            "total_attempts": 0,
            "accuracy_overall": None,
            "daily": [],
            "by_item_type": {},
            "by_skill": [],
        }

    correct_total = sum(1 for a in attempts if getattr(a, "correct", False))
    accuracy_overall = correct_total / total if total > 0 else None

    # Günlük istatistikler
    daily_raw: Dict[str, Dict[str, float]] = {}
    if has_created_at:
        for a in attempts:
            created_at = getattr(a, "created_at", None)
            if not created_at:
                continue
            date_str = created_at.date().isoformat()
            if date_str not in daily_raw:
                daily_raw[date_str] = {"count": 0, "correct": 0}
            daily_raw[date_str]["count"] += 1
            if getattr(a, "correct", False):
                daily_raw[date_str]["correct"] += 1

    daily = []
    for date_str, s in sorted(daily_raw.items()):
        cnt = s["count"]
        acc = s["correct"] / cnt if cnt > 0 else None
        daily.append({"date": date_str, "count": cnt, "accuracy": acc})

    # item_type bazında istatistikler
    by_item_type_stats: Dict[str, Dict[str, float]] = {}
    for a in attempts:
        item_type = getattr(a, "item_type", "unknown") or "unknown"
        if item_type not in by_item_type_stats:
            by_item_type_stats[item_type] = {"count": 0, "correct": 0}
        by_item_type_stats[item_type]["count"] += 1
        if getattr(a, "correct", False):
            by_item_type_stats[item_type]["correct"] += 1

    by_item_type: Dict[str, Dict[str, float]] = {}
    for item_type, s in by_item_type_stats.items():
        cnt = s["count"]
        by_item_type[item_type] = {
            "count": cnt,
            "accuracy": s["correct"] / cnt if cnt > 0 else None,
        }

    # Skill bazlı istatistikler
    math_ids: List[int] = []
    en_ids: List[int] = []
    for a in attempts:
        if getattr(a, "item_type", None) == "math":
            math_ids.append(getattr(a, "item_id", None))
        elif getattr(a, "item_type", None) == "en":
            en_ids.append(getattr(a, "item_id", None))

    math_ids = [i for i in set(math_ids) if i is not None]
    en_ids = [i for i in set(en_ids) if i is not None]

    math_items_by_id: Dict[int, MathItem] = {}
    if math_ids:
        math_result = await db.execute(
            select(MathItem).where(MathItem.id.in_(math_ids))
        )
        for mi in math_result.scalars().all():
            math_items_by_id[mi.id] = mi

    en_items_by_id: Dict[int, EnglishItem] = {}
    if en_ids:
        en_result = await db.execute(
            select(EnglishItem).where(EnglishItem.id.in_(en_ids))
        )
        for ei in en_result.scalars().all():
            en_items_by_id[ei.id] = ei

    skill_stats: Dict[str, Dict[str, float]] = {}
    for a in attempts:
        item_type = getattr(a, "item_type", None)
        item_id = getattr(a, "item_id", None)
        is_correct = getattr(a, "correct", False)

        if item_type == "math" and item_id in math_items_by_id:
            skills = math_items_by_id[item_id].skills or []
        elif item_type == "en" and item_id in en_items_by_id:
            skills = en_items_by_id[item_id].error_tags or []
        else:
            skills = []

        for s in skills:
            if s not in skill_stats:
                skill_stats[s] = {"count": 0, "correct": 0}
            skill_stats[s]["count"] += 1
            if is_correct:
                skill_stats[s]["correct"] += 1

    by_skill = []
    for skill, s in sorted(skill_stats.items(), key=lambda kv: kv[0]):
        cnt = s["count"]
        acc = s["correct"] / cnt if cnt > 0 else None
        by_skill.append(
            {
                "skill": skill,
                "count": cnt,
                "accuracy": acc,
            }
        )

    return {
        "user_id": user_id,
        "from": date_from.isoformat(),
        "to": now.isoformat(),
        "total_attempts": total,
        "accuracy_overall": accuracy_overall,
        "daily": daily,
        "by_item_type": by_item_type,
        "by_skill": by_skill,
    }
