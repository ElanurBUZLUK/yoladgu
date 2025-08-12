from __future__ import annotations

from typing import Dict, Iterable
from datetime import datetime, timedelta, timezone
import math
import redis
from sqlalchemy import select
from sqlalchemy.sql import func

from app.services.peer_hardness import PeerStore
from app.core.config import settings
from app.core.db import SessionLocal
from app.models import Attempt


class RedisPeerStore(PeerStore):
    def __init__(
        self,
        redis_url: str,
        prefix: str = "peer",
        half_life_days: float = float(settings.PEER_HALF_LIFE_DAYS),
        lookback_days: int = int(settings.PEER_LOOKBACK_DAYS),
        cache_ttl_s: int = int(getattr(settings, "FEATURE_CACHE_TTL_S", 600)),
    ) -> None:
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix
        self.half_life_days = half_life_days
        self.lookback_days = lookback_days
        self.cache_ttl_s = cache_ttl_s

    def _k_wrong(self, sid: int) -> str:
        return f"{self.prefix}:wrong:{int(sid)}"

    def _k_right(self, sid: int) -> str:
        return f"{self.prefix}:right:{int(sid)}"

    def _k_ts(self, sid: int) -> str:
        return f"{self.prefix}:ts:{int(sid)}"

    def _k_students(self) -> str:
        return f"{self.prefix}:students"

    def _decay(self, ts: datetime) -> float:
        # exponential decay by days since attempt
        now = datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        days = max(0.0, (now - ts).total_seconds() / 86400.0)
        return math.exp(-days / max(1e-6, self.half_life_days))

    async def _rebuild_student_sets(self, sid: int) -> None:
        since = datetime.utcnow() - timedelta(days=self.lookback_days)
        async with SessionLocal() as db:  # type: ignore
            rows = (
                await db.execute(
                    select(Attempt.question_id, Attempt.is_correct, Attempt.created_at)
                    .where(Attempt.student_id == sid, Attempt.created_at >= since)
                )
            ).all()
        wrong: Dict[int, float] = {}
        right: Dict[int, float] = {}
        for qid, is_correct, created_at in rows:
            w = self._decay(created_at or since)
            if bool(is_correct):
                right[int(qid)] = right.get(int(qid), 0.0) + w
            else:
                wrong[int(qid)] = wrong.get(int(qid), 0.0) + w
        pipe = self.r.pipeline()
        pipe.delete(self._k_wrong(sid))
        pipe.delete(self._k_right(sid))
        if wrong:
            pipe.zadd(self._k_wrong(sid), {str(q): float(w) for q, w in wrong.items()})
        if right:
            pipe.zadd(self._k_right(sid), {str(q): float(w) for q, w in right.items()})
        pipe.setex(self._k_ts(sid), self.cache_ttl_s, str(int(datetime.utcnow().timestamp())))
        pipe.sadd(self._k_students(), int(sid))
        pipe.execute()

    def _ensure_cached(self, sid: int) -> None:
        try:
            ts_raw = self.r.get(self._k_ts(sid))
            if not ts_raw:
                import asyncio
                asyncio.get_event_loop().run_until_complete(self._rebuild_student_sets(sid))
        except Exception:
            # best-effort
            pass

    def wrong_set(self, student_id: int) -> Dict[int, float]:
        self._ensure_cached(student_id)
        items = self.r.zrange(self._k_wrong(student_id), 0, -1, withscores=True) or []
        return {int(q): float(w) for q, w in items}

    def right_set(self, student_id: int) -> Dict[int, float]:
        self._ensure_cached(student_id)
        items = self.r.zrange(self._k_right(student_id), 0, -1, withscores=True) or []
        return {int(q): float(w) for q, w in items}

    def has_attempt(self, student_id: int, question_id: int) -> bool:
        if self.r.zscore(self._k_wrong(student_id), str(int(question_id))) is not None:
            return True
        if self.r.zscore(self._k_right(student_id), str(int(question_id))) is not None:
            return True
        return False

    def sim_index_neighbors(self, student_id: int) -> Iterable[int]:
        # Iterate known students set
        try:
            sids = [int(x) for x in self.r.smembers(self._k_students())]
            for sid in sids:
                if sid != int(student_id):
                    yield sid
        except Exception:
            return []

from __future__ import annotations

from typing import Dict, Iterable
import redis


class RedisPeerStore:
    """Redis-backed PeerStore compatible with PeerHardnessService.

    Keys:
      - peer:wrong:{student_id} -> hash of question_id -> weight
      - peer:right:{student_id} -> hash of question_id -> weight
      - peer:neighbors:{student_id} -> set/list of neighbor ids (optional)
    """

    def __init__(self, redis_url: str, prefix: str = "peer") -> None:
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix

    def _k_wrong(self, student_id: int) -> str:
        return f"{self.prefix}:wrong:{int(student_id)}"

    def _k_right(self, student_id: int) -> str:
        return f"{self.prefix}:right:{int(student_id)}"

    def _k_neighbors(self, student_id: int) -> str:
        return f"{self.prefix}:neighbors:{int(student_id)}"

    def wrong_set(self, student_id: int) -> Dict[int, float]:
        m = self.r.hgetall(self._k_wrong(student_id))
        return {int(k): float(v) for k, v in m.items()}

    def right_set(self, student_id: int) -> Dict[int, float]:
        m = self.r.hgetall(self._k_right(student_id))
        return {int(k): float(v) for k, v in m.items()}

    def has_attempt(self, student_id: int, question_id: int) -> bool:
        sid = int(student_id)
        qid = str(int(question_id))
        return self.r.hexists(self._k_wrong(sid), qid) or self.r.hexists(self._k_right(sid), qid)

    def sim_index_neighbors(self, student_id: int) -> Iterable[int]:
        # Optional neighbor index; fallback to empty
        key = self._k_neighbors(student_id)
        try:
            arr = self.r.smembers(key)
            if not arr:
                return []
            return [int(x) for x in arr]
        except Exception:
            return []


