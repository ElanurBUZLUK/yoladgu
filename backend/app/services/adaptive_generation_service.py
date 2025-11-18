"""
Service for handling adaptive question generation logic.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.generate import (
    AdaptiveMathResponse,
    AdaptiveMathQuestion,
    AdaptiveEnglishResponse,
    AdaptiveEnglishQuestion,
)
from app.db.models import Attempt, MathItem as MathItemDB, EnglishItem as EnglishItemDB
from app.models.response import MathItem
from app.services.math_generation_service import math_generation_service
from app.services.english_generation_service import english_generation_service
from app.services.search_service import search_service

logger = logging.getLogger(__name__)


class AdaptiveGenerationService:
    async def generate_adaptive_math(
        self,
        session: AsyncSession,
        user_id: str,
        n_questions: int,
        days: int,
        min_attempts_per_skill: int,
        template_ids: Optional[List[str]],
        target_difficulty: Optional[float],
        persist: bool,
    ) -> AdaptiveMathResponse:
        """
        Orchestrates the adaptive math question generation process.
        """
        # 1. Analyze user performance to find weak skills
        weak_skills_info = await self._analyze_user_performance(
            session, user_id, days, min_attempts_per_skill, "math"
        )

        # 2. Get available templates
        available_templates = math_generation_service.get_available_templates()
        if template_ids:
            allowed = set(template_ids)
            available_templates = [
                t for t in available_templates if t["template_id"] in allowed
            ]

        if not available_templates:
            raise ValueError("Kullanılabilir math template bulunamadı.")

        # 3. Generate questions for weak skills
        questions: List[AdaptiveMathQuestion] = []
        for i in range(n_questions):
            target_info = weak_skills_info[i % len(weak_skills_info)]
            target_skill = target_info["skill"]

            # Find a suitable template
            skill_templates = [
                t for t in available_templates if target_skill in (t.get("skills") or [])
            ]
            if not skill_templates:
                skill_templates = available_templates  # Fallback

            chosen_template = skill_templates[0]
            template_id = chosen_template["template_id"]

            # Determine difficulty
            difficulty_used = self._determine_difficulty(
                target_difficulty, target_info["accuracy"]
            )

            # Generate question
            question_data = math_generation_service.generate_question(
                template_id=template_id,
                params_hint=None,
                target_difficulty=difficulty_used,
                language="tr",
                rationale_required=False,
            )

            item = MathItem(**question_data["item"])

            # Persist if requested
            saved_item_id = None
            if persist:
                saved_item_id = await self._persist_math_item(
                    session, item, difficulty_used
                )

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
            user_id=user_id, questions=questions, weak_skills=weak_skills_info
        )

    async def generate_adaptive_english(
        self,
        session: AsyncSession,
        user_id: str,
        n_questions: int,
        days: int,
        min_attempts_per_skill: int,
        template_ids: Optional[List[str]],
        target_difficulty: Optional[float],
        persist: bool,
    ) -> AdaptiveEnglishResponse:
        """
        Orchestrates the adaptive English question generation process.
        """
        # 1. Analyze user performance to find weak error_tags
        weak_skills_info = await self._analyze_user_performance(
            session, user_id, days, min_attempts_per_skill, "en"
        )

        # 2. Get available templates (if any)
        try:
            available_templates = english_generation_service.get_available_templates()
        except AttributeError:
            available_templates = []

        # 3. Generate questions for weak skills
        questions: List[AdaptiveEnglishQuestion] = []
        for i in range(n_questions):
            target_info = weak_skills_info[i % len(weak_skills_info)]
            target_skill = target_info["skill"]

            # Find a suitable template
            chosen_template_id = None
            if available_templates:
                skill_templates = [
                    t for t in available_templates
                    if target_skill in (t.get("error_tags") or t.get("skills") or [])
                ]
                if skill_templates:
                    chosen_template_id = skill_templates[0]["template_id"]

            # Determine difficulty
            difficulty_used = self._determine_difficulty(
                target_difficulty, target_info["accuracy"]
            )

            # Generate question
            question_data = english_generation_service.generate_cloze_item(
                template_id=chosen_template_id,
                target_error_tags=[target_skill],
                difficulty_hint=difficulty_used,
                language="en",
            )
            item_dict = question_data["item"]

            # Persist if requested
            saved_item_id = None
            if persist:
                saved_item_id = await self._persist_english_item(
                    session, item_dict, target_skill, difficulty_used
                )

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
            user_id=user_id, questions=questions, weak_skills=weak_skills_info
        )

    async def _analyze_user_performance(
        self, session: AsyncSession, user_id: str, days: int, min_attempts: int, item_type: str
    ) -> List[Dict[str, Any]]:
        """
        Fetches user attempts and calculates skill/error_tag based statistics.
        """
        date_from = datetime.utcnow() - timedelta(days=days)
        result = await session.execute(
            select(Attempt).where(
                Attempt.user_id == user_id,
                Attempt.item_type == item_type,
                Attempt.created_at >= date_from,
            )
        )
        attempts = result.scalars().all()
        if not attempts:
            raise ValueError(f"Bu kullanıcı için belirtilen süre içinde {item_type} attempt bulunamadı.")

        item_ids = list({a.item_id for a in attempts if a.item_id})
        if not item_ids:
            raise ValueError("Attempt'lerde geçerli item_id bulunamadı.")

        if item_type == "math":
            result_items = await session.execute(select(MathItemDB).where(MathItemDB.id.in_(item_ids)))
            items_by_id = {m.id: m for m in result_items.scalars().all()}
            skill_key = "skills"
        else:  # 'en'
            result_items = await session.execute(select(EnglishItemDB).where(EnglishItemDB.id.in_(item_ids)))
            items_by_id = {e.id: e for e in result_items.scalars().all()}
            skill_key = "error_tags"

        skill_stats: Dict[str, Dict[str, float]] = {}
        for a in attempts:
            item = items_by_id.get(a.item_id)
            if not item or not getattr(item, skill_key):
                continue
            for s in getattr(item, skill_key):
                if s not in skill_stats:
                    skill_stats[s] = {"count": 0, "correct": 0}
                skill_stats[s]["count"] += 1
                if a.correct:
                    skill_stats[s]["correct"] += 1

        weak_skills_info = [
            {"skill": s, "count": st["count"], "accuracy": st["correct"] / st["count"] if st["count"] > 0 else 0.0}
            for s, st in skill_stats.items()
            if st["count"] >= min_attempts
        ]

        if not weak_skills_info:
            raise ValueError(f"Yeterli deneme sayısına sahip zayıf {item_type} skill bulunamadı.")

        weak_skills_info.sort(key=lambda x: x["accuracy"])
        return weak_skills_info

    def _determine_difficulty(self, target_difficulty: Optional[float], skill_accuracy: float) -> float:
        if target_difficulty is not None:
            return target_difficulty
        if skill_accuracy < 0.4:
            return -0.5  # easy
        elif skill_accuracy < 0.7:
            return 0.0  # medium
        else:
            return 0.5  # hard

    async def _persist_math_item(self, session: AsyncSession, item: MathItem, difficulty: float) -> str:
        try:
            db_item = MathItemDB(
                stem=item.stem, choices=item.choices, answer_key=item.answer_key,
                solution=item.solution, skills=item.skills, bloom_level=item.bloom_level,
                difficulty_a=difficulty, difficulty_b=item.difficulty_estimate, source="adaptive",
            )
            session.add(db_item)
            await session.commit()
            await session.refresh(db_item)
            await search_service.index_math_item(db_item)
            return str(db_item.id)
        except Exception as e:
            logger.error(f"Failed to persist/index adaptive math item: {e}")
            await session.rollback()
            return ""

    async def _persist_english_item(self, session: AsyncSession, item_dict: dict, target_skill: str, difficulty: float) -> str:
        try:
            db_item = EnglishItemDB(
                text=item_dict.get("text"), options=item_dict.get("options"),
                answer_key=item_dict.get("answer_key"),
                error_tags=item_dict.get("error_tags") or [target_skill],
                source="adaptive", difficulty_estimate=difficulty,
            )
            session.add(db_item)
            await session.commit()
            await session.refresh(db_item)
            await search_service.index_english_item(db_item)
            return str(db_item.id)
        except Exception as e:
            logger.error(f"Failed to persist/index adaptive English item: {e}")
            await session.rollback()
            return ""

adaptive_generation_service = AdaptiveGenerationService()