"""
Adaptive math service that generates personalized math questions based on user profiles.
"""

from typing import List, Optional, Dict, Any
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.profile_service import profile_service
from app.services.math_generation_service import math_generation_service
from app.services.retrieval_service import retrieval_service
from app.db.repositories.item import math_item_repository
from app.db.repositories.user import user_repository
from app.models.response import MathQuestionResponse

logger = logging.getLogger(__name__)


class AdaptiveMathService:
    """
    Öğrencinin profilini + geçmiş hatalarını kullanarak
    zayıf olduğu konulardan yeni math sorusu üreten servis.
    """

    async def _select_weak_skills(
        self,
        error_profile_math: Dict[str, Any],
        skill_focus: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[str]:
        """
        Eğer skill_focus verilmişse onu kullan,
        verilmemişse error_profile_math içinden en zayıf skill'leri seç.
        """
        if skill_focus:
            return skill_focus

        if not error_profile_math:
            return ["general_arithmetic"]

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
            error_profile_math.items(),
            key=lambda kv: _error_rate(kv[1]),
            reverse=True,
        )
        return [name for name, _stats in sorted_skills[:top_k]] or ["general_arithmetic"]

    async def _gather_rag_context(
        self,
        session: AsyncSession,
        user_id: str,
        skills: List[str],
        difficulty_range: Optional[tuple] = None,
        top_k: int = 5,
    ) -> str:
        """Collect optional RAG context chunks for the requested skills."""
        query = " ".join(skills) if skills else "math practice"
        rag_context_chunks: List[str] = []

        try:
            if hasattr(retrieval_service, "retrieve_math_context"):
                docs = await retrieval_service.retrieve_math_context(
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
                    lang="tr",
                    k=top_k,
                )
                for res in results or []:
                    metadata = res.get("metadata", {})
                    chunk = metadata.get("content") or metadata.get("stem")
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
                    chunk = metadata.get("content") or metadata.get("stem")
                    if chunk:
                        rag_context_chunks.append(chunk)
        except Exception as exc:
            logger.warning(
                "Adaptive math context retrieval failed",
                extra={"user_id": user_id, "error": str(exc)},
            )

        return "\n\n".join(rag_context_chunks)

    def _choose_template(self, weak_skills: List[str]) -> str:
        """Pick a generation template that covers the target skills."""
        templates = math_generation_service.get_available_templates()
        if not templates:
            raise ValueError("No math generation templates available")

        for template in templates:
            template_skills = template.get("skills") or []
            if any(skill in template_skills for skill in weak_skills):
                return template["template_id"]

        return templates[0]["template_id"]

    async def generate_adaptive_math_question(
        self,
        session: AsyncSession,
        user_id: str,
        skill_focus: Optional[List[str]] = None,
        max_difficulty_shift: float = 0.5,
    ) -> MathQuestionResponse:
        """
        1. Kullanıcının profilini al (theta_math + error_profile_math)
        2. Zayıf skill'leri seç
        3. Theta'ya göre hedef zorluk belirle
        4. Opsiyonel RAG bağlamı topla
        5. Soru üret ve MathItem tablosuna kaydet
        6. Kaydedilen soruyu MathQuestionResponse olarak döndür
        """
        user = await user_repository.get(session, user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")

        profile = await profile_service.get_user_profile(
            session=session,
            user_id=user_id,
            use_cache=True,
        )

        theta_math = (profile or {}).get("theta_math") or 0.0
        error_profile_math = (profile or {}).get("error_profile_math") or {}

        weak_skills = await self._select_weak_skills(
            error_profile_math=error_profile_math,
            skill_focus=skill_focus,
            top_k=3,
        )

        max_shift = max(max_difficulty_shift, 0.1)
        lower = theta_math - max_shift
        upper = theta_math + max_shift
        target_difficulty = theta_math

        rag_context = await self._gather_rag_context(
            session=session,
            user_id=user_id,
            skills=weak_skills,
            difficulty_range=(lower, upper),
        )

        template_id = self._choose_template(weak_skills)
        generated = math_generation_service.generate_question(
            template_id=template_id,
            target_difficulty=target_difficulty,
            language=user.lang or "tr",
            rationale_required=False,
        )

        generated_item = generated["item"]
        generator_meta = generated.get("generator", {})
        params = generator_meta.get("params") or {}
        if rag_context:
            params = dict(params)
            params["source_context"] = rag_context

        difficulty_estimate = generated_item.get("difficulty_estimate") or {}
        difficulty_a = difficulty_estimate.get("a", 1.0)
        difficulty_b = difficulty_estimate.get("b", target_difficulty)

        db_item = await math_item_repository.create(
            session=session,
            obj_in={
                "tenant_id": user.tenant_id,
                "stem": generated_item["stem"],
                "params": params,
                "solution": generated_item.get("solution"),
                "answer_key": generated_item["answer_key"],
                "choices": generated_item.get("choices") or [],
                "skills": generated_item.get("skills") or weak_skills,
                "bloom_level": generated_item.get("bloom_level"),
                "topic": generator_meta.get("template_id"),
                "difficulty_a": difficulty_a,
                "difficulty_b": difficulty_b,
                "lang": user.lang or "tr",
                "source": "adaptive_math_service",
                "generator": generator_meta.get("template_id"),
            },
        )

        return MathQuestionResponse(
            item_id=db_item.id,
            question_text=db_item.stem,
            choices=db_item.choices,
            correct_answer=db_item.answer_key,
            solution_steps=db_item.solution,
            skills=db_item.skills or weak_skills,
            source_context=rag_context or None,
        )


adaptive_math_service = AdaptiveMathService()
