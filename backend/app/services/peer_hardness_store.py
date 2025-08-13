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

 


