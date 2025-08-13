from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import time, json, redis
from app.core.config import settings


class MistakeRepo:
    # Redis buckets
    #  enqg:vocab:{student_id}   -> hash: lemma -> json
    #  enqg:grammar:{student_id} -> hash: rule_code -> json
    def __init__(self, redis_url: Optional[str] = None):
        self.r = redis.Redis.from_url(redis_url or settings.REDIS_URL, decode_responses=True)

    def inc_vocab(self, student_id: int, lemma: str, correct: bool) -> None:
        key = f"enqg:vocab:{student_id}"
        now = int(time.time())
        data = self._get_hash_json(key, lemma)
        data["seen"] = int(data.get("seen", 0)) + 1
        data["correct"] = int(data.get("correct", 0)) + (1 if correct else 0)
        data["wrong"] = int(data.get("wrong", 0)) + (0 if correct else 1)
        data["p_hat"] = float(0.8 * float(data.get("p_hat", 0.5)) + 0.2 * (1.0 if correct else 0.0))
        data["last_seen"] = now
        data["next_due"] = self._next_due(correct, data)
        self.r.hset(key, lemma, json.dumps(data, ensure_ascii=False))

    def inc_rule(self, student_id: int, rule_code: str, correct: bool) -> None:
        key = f"enqg:grammar:{student_id}"
        now = int(time.time())
        data = self._get_hash_json(key, rule_code)
        data["seen"] = int(data.get("seen", 0)) + 1
        data["correct"] = int(data.get("correct", 0)) + (1 if correct else 0)
        data["wrong"] = int(data.get("wrong", 0)) + (0 if correct else 1)
        data["p_hat"] = float(0.8 * float(data.get("p_hat", 0.5)) + 0.2 * (1.0 if correct else 0.0))
        data["last_seen"] = now
        data["next_due"] = self._next_due(correct, data)
        self.r.hset(key, rule_code, json.dumps(data, ensure_ascii=False))

    def pick_candidates(self, student_id: int, mode: str, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        now = int(time.time())
        out: Dict[str, List[Tuple[str, float]]] = {"vocab": [], "grammar": []}
        if mode in ("vocab", "mixed"):
            out["vocab"] = self._rank_bucket(f"enqg:vocab:{student_id}", now, top_n)
        if mode in ("grammar", "mixed"):
            out["grammar"] = self._rank_bucket(f"enqg:grammar:{student_id}", now, top_n)
        return out

    def _rank_bucket(self, key: str, now: int, top_n: int) -> List[Tuple[str, float]]:
        items: List[Tuple[str, float]] = []
        for k, raw in (self.r.hgetall(key) or {}).items():
            try:
                d = json.loads(raw) if raw else {}
            except Exception:
                d = {}
            is_due = 1.0 if int(d.get("next_due", 0)) <= now else 0.0
            p_hat = float(d.get("p_hat", 0.5))
            score = 0.6 * is_due + 0.4 * (1.0 - p_hat)
            items.append((k, score))
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_n]

    def _get_hash_json(self, key: str, field: str) -> Dict[str, Any]:
        raw = self.r.hget(key, field)
        try:
            return json.loads(raw) if raw else {}
        except Exception:
            return {}

    def _next_due(self, correct: bool, d: Dict[str, Any]) -> int:
        now = int(time.time())
        seen = int(d.get("seen", 0))
        if correct:
            days = [1, 3, 7, 14, 28]
            idx = min(max(seen - 1, 0), len(days) - 1)
            return now + days[idx] * 86400
        return now + 86400


