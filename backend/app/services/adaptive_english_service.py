"""
Adaptive English question service that generates personalized questions based on user profiles.
"""

from typing import List, Optional, Dict, Any
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.profile_service import profile_service
from app.services.english_generation_service import english_generation_service
from app.services.retrieval_service import retrieval_service
from app.db.repositories.item import english_item_repository
from app.db.repositories.user import user_repository
from app.models.response import EnglishQuestionResponse

logger = logging.getLogger(__name__)


class AdaptiveEnglishService:
    """
    Öğrencinin İngilizce profilini (theta_en + error_profile_en) kullanarak
    zayıf olduğu hata türlerinden / skill'lerden yeni soru üreten servis.
    """

    async def _select_weak_skills(
        self,
        error_profile_en: Dict[str, Any],
        skill_focus: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[str]:
        """Select weakest skills either from focus list or error profile."""
        if skill_focus:
            return skill_focus

        if not error_profile_en:
            return ["general_grammar"]

        def _error_rate(stat_value: Any) -> float:
            if isinstance(stat_value, dict):
                if "error_rate" in stat_value:
                    return float(stat_value["error_rate"])
                errors = float(stat_value.get("errors", 0))
                attempts = float(stat_value.get("attempts", 1))
                return errors / attempts if attempts else 0.0
            if isinstance(stat_value, (int, float)):
                return float(stat_value)
            return 0.0

        sorted_skills = sorted(
            error_profile_en.items(),
            key=lambda kv: _error_rate(kv[1]),
            reverse=True,
        )
        return [name for name, _stats in sorted_skills[:top_k]] or ["general_grammar"]

    async def _gather_rag_context(
        self,
        session: AsyncSession,
        user_id: str,
        skills: List[str],
        difficulty_range: Optional[tuple] = None,
        top_k: int = 5,
    ) -> str:
        """Collect optional RAG context for English generation."""
        query = " ".join(skills) if skills else "english practice"
        rag_context_chunks: List[str] = []

        try:
            if hasattr(retrieval_service, "retrieve_english_context"):
                docs = await retrieval_service.retrieve_english_context(
                    session=session,
                    skills=skills,
                    difficulty_range=difficulty_range,
                    k=top_k,
                )
                for doc in docs or []:
                    chunk = getattr(doc, "content", None)
                    if chunk is None and isinstance(doc, dict):
                        chunk = doc.get("content") or doc.get("metadata", {}).get("content")
                    if chunk:
                        rag_context_chunks.append(chunk)
            elif hasattr(retrieval_service, "hybrid_search"):
                results = await retrieval_service.hybrid_search(
                    query=query,
                    goals={"skills": skills, "difficulty_range": difficulty_range},
                    lang="en",
                    k=top_k,
                )
                for res in results or []:
                    metadata = res.get("metadata", {})
                    chunk = metadata.get("content") or metadata.get("passage")
                    if chunk:
                        rag_context_chunks.append(chunk)
            elif hasattr(retrieval_service, "search_with_logging"):
                results, _log = await retrieval_service.search_with_logging(
                    user_id=user_id,
                    query=query,
                    top_k=top_k,
                    filters={"skills": skills} if skills else None,
                    db=session,
                )
                for res in results or []:
                    metadata = res.get("metadata", {})
                    chunk = metadata.get("content") or metadata.get("passage")
                    if chunk:
                        rag_context_chunks.append(chunk)
        except Exception as exc:
            logger.warning(
                "Adaptive English context retrieval failed",
                extra={"user_id": user_id, "error": str(exc)},
            )

        return "\n\n".join(rag_context_chunks)

    async def generate_adaptive_english_question(
        self,
        session: AsyncSession,
        user_id: str,
        skill_focus: Optional[List[str]] = None,
        max_difficulty_shift: float = 0.5,
    ) -> EnglishQuestionResponse:
        """
        Generate and persist an adaptive English question for the user.
        """
        user = await user_repository.get(session, user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")

        profile = await profile_service.get_user_profile(
            session=session,
            user_id=user_id,
            use_cache=True,
        )

        theta_en = (profile or {}).get("theta_en") or 0.0
        error_profile_en = (profile or {}).get("error_profile_en") or {}

        weak_skills = await self._select_weak_skills(
            error_profile_en=error_profile_en,
            skill_focus=skill_focus,
            top_k=3,
        )

        max_shift = max(max_difficulty_shift, 0.1)
        lower = theta_en - max_shift
        upper = theta_en + max_shift
        target_difficulty = theta_en

        rag_context = await self._gather_rag_context(
            session=session,
            user_id=user_id,
            skills=weak_skills,
            difficulty_range=(lower, upper),
        )

        question_data = english_generation_service.generate_cloze_item(
            target_error_tags=weak_skills,
            difficulty_hint=target_difficulty,
            language=user.lang or "en",
        )
        item_dict = question_data["item"]

        db_item = await english_item_repository.create(
            session=session,
            obj_in={
                "tenant_id": user.tenant_id,
                "passage": item_dict.get("passage"),
                "blanks": item_dict.get("blanks") or [],
                "level_cefr": item_dict.get("level_cefr") or "B1",
                "topic": item_dict.get("topic"),
                "error_tags": item_dict.get("error_tags") or weak_skills,
                "lang": user.lang or "en",
                "source": "adaptive_english_service",
                "generator": "adaptive_cloze",
                "status": "draft",
            },
        )

        prompt = item_dict.get("passage") or ""
        choices = []
        correct_answer = None
        explanation = None

        if db_item.blanks:
            first_blank = db_item.blanks[0]
            choices = first_blank.get("distractors", []) + [first_blank.get("answer")]
            correct_answer = first_blank.get("answer")
            explanation = first_blank.get("rationale")

        return EnglishQuestionResponse(
            item_id=db_item.id,
            prompt=prompt,
            choices=choices or None,
            correct_answer=correct_answer,
            explanation=explanation,
            error_tags=db_item.error_tags or weak_skills,
            source_context=rag_context or None,
        )


adaptive_english_service = AdaptiveEnglishService()
