from __future__ import annotations

from typing import Dict
from datetime import datetime, timedelta
from functools import lru_cache
from sqlalchemy import select, func, Integer
from app.core.db import SessionLocal
from app.core.config import settings
from app.models import Attempt, Question


class FeatureStore:
    def __init__(self, window_days: int = 30):
        self.window_days = int(window_days)

    async def _user_features_async(self, user_id: int) -> Dict[str, float]:
        since = datetime.utcnow() - timedelta(days=self.window_days)
        async with SessionLocal() as db:  # type: ignore
            res = await db.execute(
                select(
                    func.avg(func.cast(Attempt.is_correct, Integer)).label("avg_correct"),
                    func.avg(Attempt.time_ms).label("avg_time"),
                    func.count().label("cnt"),
                ).where(Attempt.student_id == user_id, Attempt.created_at >= since)
            )
            row = res.one_or_none()
            avg_correct = float(row[0] or 0.0) if row else 0.0
            avg_time = float(row[1] or 0.0) if row else 0.0
            cnt = float(row[2] or 0) if row else 0.0
            return {
                "u_avg_correct_30d": avg_correct,
                "u_avg_time_ms_30d": avg_time,
                "u_attempts_30d": cnt,
            }

    async def _question_features_async(self, question_id: int) -> Dict[str, float]:
        since = datetime.utcnow() - timedelta(days=self.window_days)
        async with SessionLocal() as db:  # type: ignore
            q_static = await db.execute(
                select(Question.difficulty_rating, Question.t_ref_ms).where(Question.id == question_id)
            )
            row = q_static.one_or_none()
            difficulty = float(row[0] or 0.0) if row else 0.0
            t_ref = float(row[1] or 0) if row else 10000.0

            q_dyn = await db.execute(
                select(
                    func.avg(func.cast(Attempt.is_correct, Integer)).label("avg_correct"),
                    func.count().label("cnt"),
                ).where(Attempt.question_id == question_id, Attempt.created_at >= since)
            )
            row2 = q_dyn.one_or_none()
            avg_correct = float(row2[0] or 0.0) if row2 else 0.0
            cnt = float(row2[1] or 0) if row2 else 0.0
            return {
                "q_difficulty": difficulty,
                "q_t_ref_ms": t_ref,
                "q_avg_correct_30d": avg_correct,
                "q_attempts_30d": cnt,
            }

    def get_user_features(self, user_id: int) -> Dict[str, float]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._user_features_async(user_id))

    def get_question_features(self, question_id: int) -> Dict[str, float]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._question_features_async(question_id))

from __future__ import annotations

from typing import Dict
from datetime import datetime, timedelta

from sqlalchemy import select, func, Integer, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import SessionLocal
from app.models import Attempt, Question


class FeatureStore:
    """Database-backed feature store for user and question features.

    This minimal implementation computes windowed statistics on demand.
    Production systems should consider materialized views or background jobs.
    """

    def __init__(self, window_days: int = 30) -> None:
        self.window_days = int(window_days)

    async def get_user_features_async(self, db: AsyncSession, user_id: int) -> Dict[str, float]:
        now = datetime.utcnow()
        since = now - timedelta(days=self.window_days)
        q = await db.execute(
            select(
                func.avg(func.cast(Attempt.is_correct, Integer)),
                func.avg(Attempt.time_ms),
                func.count(),
            ).where(Attempt.student_id == user_id, Attempt.created_at >= since)
        )
        avg_correct, avg_time, cnt = q.one_or_none() or (None, None, 0)
        return {
            "u_avg_correct_window": float(avg_correct or 0.0),
            "u_avg_time_ms_window": float(avg_time or 0.0),
            "u_attempts_window": float(cnt or 0),
        }

    async def get_question_features_async(self, db: AsyncSession, question_id: int) -> Dict[str, float]:
        now = datetime.utcnow()
        since = now - timedelta(days=self.window_days)
        q1 = await db.execute(
            select(Question.difficulty_rating, Question.t_ref_ms).where(Question.id == question_id)
        )
        row = q1.one_or_none()
        diff, tref = (row[0], row[1]) if row else (0.0, 10000)
        q2 = await db.execute(
            select(func.avg(func.cast(Attempt.is_correct, Integer)), func.count()).where(
                Attempt.question_id == question_id, Attempt.created_at >= since
            )
        )
        avg_correct, cnt = q2.one_or_none() or (None, 0)
        return {
            "q_difficulty": float(diff or 0.0),
            "q_t_ref_ms": float(tref or 0),
            "q_avg_correct_window": float(avg_correct or 0.0),
            "q_attempts_window": float(cnt or 0),
        }

    # Convenience sync wrappers for contexts where we do not have an async session
    def get_user_features(self, user_id: int) -> Dict[str, float]:
        async def _run() -> Dict[str, float]:
            async with SessionLocal() as db:
                return await self.get_user_features_async(db, user_id)

        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # If already in an event loop, create a task and wait
            return loop.run_until_complete(_run())  # type: ignore
        return asyncio.run(_run())

    def get_question_features(self, question_id: int) -> Dict[str, float]:
        async def _run() -> Dict[str, float]:
            async with SessionLocal() as db:
                return await self.get_question_features_async(db, question_id)

        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            return loop.run_until_complete(_run())  # type: ignore
        return asyncio.run(_run())


