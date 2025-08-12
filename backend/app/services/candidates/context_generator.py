from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import SessionLocal
from app.models import Attempt
from app.services.vector_index_manager import VectorIndexManager


class ContextAwareCandidateGenerator:
    def __init__(self, k_recent: int = 5) -> None:
        self.k_recent = int(k_recent)
        self.idx = VectorIndexManager(settings.REDIS_URL)

    async def _recent_attempt_ids(self, db: AsyncSession, student_id: int) -> List[int]:
        q = await db.execute(
            select(Attempt.question_id)
            .where(Attempt.student_id == int(student_id))
            .order_by(Attempt.created_at.desc())
            .limit(self.k_recent)
        )
        return [int(x) for x in q.scalars().all()]

    async def _load_embeddings(self, db: AsyncSession, qids: List[int]) -> List[List[float]]:
        if not qids:
            return []
        sql = text("SELECT id, embedding FROM embeddings WHERE id = ANY(:ids)")
        res = await db.execute(sql.bindparams(ids=qids))
        out: List[List[float]] = []
        for row in res.fetchall():
            try:
                vec = row[1]
                # Depending on driver, vec may already be a list; ensure list[float]
                if isinstance(vec, list):
                    out.append([float(x) for x in vec])
                else:
                    # Fallback: try to parse text like "[0.1,0.2]"
                    s = str(vec).strip().strip("[]")
                    if s:
                        out.append([float(x) for x in s.split(",")])
            except Exception:
                continue
        return out

    async def _build_context(self, db: AsyncSession, student_id: int) -> Optional[np.ndarray]:
        recent_qids = await self._recent_attempt_ids(db, student_id)
        vecs = await self._load_embeddings(db, recent_qids)
        if not vecs:
            return None
        X = np.array(vecs, dtype=np.float32)
        v = X.mean(axis=0)
        # Normalize to unit length to align with cosine distance usage
        norm = np.linalg.norm(v) + 1e-9
        return (v / norm).astype(np.float32)

    def generate_candidates(self, student_id: int, topic_id: Optional[int], k: int = 50) -> List[int]:
        async def _run() -> List[int]:
            async with SessionLocal() as db:  # type: ignore
                ctx = await self._build_context(db, student_id)
                if ctx is None:
                    return []
                ids, _ = self.idx.search(ctx, k=max(k, 50))
                # Optional topic filter if metadata has topic_id, handled upstream if needed
                return [int(i) for i in ids[:k]]

        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            return loop.run_until_complete(_run())  # type: ignore
        return asyncio.run(_run())


