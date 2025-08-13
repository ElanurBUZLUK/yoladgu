from typing import Optional, Dict, Any
import structlog

# Re-export existing services
from app.core.deps import (
    get_linucb_service,  # LinUCBService
    get_ftrl_service,    # FTRLService
)
from functools import lru_cache
from app.services.cf import CFModel
from app.core.config import settings
from app.services.feature_store import FeatureStore
from app.services.advanced_rag import AdvancedRAGService

log = structlog.get_logger()


class _FeatureStore:
    def __init__(self, db_session_factory=None):
        self._db_session_factory = db_session_factory

    def get_user_features(self, user_id: int) -> Dict[str, float]:
        # Basit örnek: Attempt tablosundan son 30 gün başarı oranı ve ortalama süre
        try:
            from sqlalchemy import select, func, Integer
            from datetime import datetime, timedelta
            from app.core.db import async_session_maker
            from app.models import Attempt
            sess_maker = self._db_session_factory or async_session_maker
            now = datetime.utcnow()
            since = now - timedelta(days=30)
            async def _run():
                async with sess_maker() as db:
                    q = await db.execute(
                        select(
                            func.avg(func.cast(Attempt.is_correct, Integer)),
                            func.avg(Attempt.time_ms),
                            func.count()
                        ).where(Attempt.student_id == user_id, Attempt.created_at >= since)
                    )
                    avg_correct, avg_time, cnt = q.one_or_none() or (None, None, 0)
                    return {
                        "u_avg_correct_30d": float(avg_correct or 0.0),
                        "u_avg_time_ms_30d": float(avg_time or 0.0),
                        "u_attempts_30d": float(cnt or 0),
                    }
            import asyncio
            return asyncio.get_event_loop().run_until_complete(_run())
        except Exception:
            return {}

    def get_question_features(self, question_id: int) -> Dict[str, float]:
        # Basit örnek: Question tablosundan zorluk ve referans süre; Attempt’lerden son 30 gün doğruluk
        try:
            from sqlalchemy import select, func, Integer
            from datetime import datetime, timedelta
            from app.core.db import async_session_maker
            from app.models import Attempt, Question
            sess_maker = self._db_session_factory or async_session_maker
            now = datetime.utcnow()
            since = now - timedelta(days=30)
            async def _run():
                async with sess_maker() as db:
                    q1 = await db.execute(select(Question.difficulty_rating, Question.t_ref_ms).where(Question.id == question_id))
                    row = q1.one_or_none()
                    diff, tref = (row[0], row[1]) if row else (0.0, 10000)
                    q2 = await db.execute(
                        select(func.avg(func.cast(Attempt.is_correct, Integer)), func.count())
                        .where(Attempt.question_id == question_id, Attempt.created_at >= since)
                    )
                    avg_correct, cnt = q2.one_or_none() or (None, 0)
                    return {
                        "q_difficulty": float(diff or 0.0),
                        "q_t_ref_ms": float(tref or 0),
                        "q_avg_correct_30d": float(avg_correct or 0.0),
                        "q_attempts_30d": float(cnt or 0),
                    }
            import asyncio
            return asyncio.get_event_loop().run_until_complete(_run())
        except Exception:
            return {}


class _EventLogger:
    def log_exposure(self, user_id: int, variant: str, payload: Dict[str, Any]):
        log.info("exposure", user_id=user_id, variant=variant, payload=payload)

    def log_outcome(self, user_id: int, variant: str, payload: Dict[str, Any]):
        log.info("outcome", user_id=user_id, variant=variant, payload=payload)


def get_feature_store() -> _FeatureStore:
    # Replace placeholder with DB-backed feature store facade
    # We keep the same interface but return a thin wrapper to the new FeatureStore
    fs = FeatureStore(window_days=settings.FEATURE_WINDOW_DAYS)

    class _FSFacade(_FeatureStore):  # type: ignore
        def get_user_features(self, user_id: int) -> Dict[str, float]:
            return fs.get_user_features(user_id)

        def get_question_features(self, question_id: int) -> Dict[str, float]:
            return fs.get_question_features(question_id)

    return _FSFacade()


def get_event_logger() -> _EventLogger:
    return _EventLogger()


def get_retriever_service():
    # Provide a concrete retriever implementation (AdvancedRAGService)
    # Defaults: hybrid search enabled; reranking toggled by settings.RERANK_ENABLED
    enable_rerank = bool(getattr(settings, "RERANK_ENABLED", False))
    return AdvancedRAGService(enable_hybrid_search=True, enable_reranking=enable_rerank)


@lru_cache(maxsize=1)
def get_cf_model() -> CFModel:
    model = CFModel()
    path = getattr(settings, "CF_MODEL_PATH", None) or "backend/app/ml/models/cf.npz"
    try:
        model.load(path)
    except Exception:
        pass
    return model


