import json, time, uuid, redis
from typing import Dict, Any, List, Optional
from enum import Enum
from app.core.config import settings

DEFAULT_TTL_SEC = 6 * 60 * 60


class SessionState(str, Enum):
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    FINISHED = "FINISHED"


class QuizSessionService:
    def __init__(self, redis_url: str):
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)

    def _key(self, sid: str) -> str:
        return f"quiz:session:{sid}"

    def create(self, user_id: int, topic_id: Optional[int]) -> str:
        sid = str(uuid.uuid4())
        payload = {
            "user_id": user_id,
            "topic_id": topic_id,
            "start_ts": int(time.time() * 1000),
            "asked_ids": [],
            "current_qid": None,
            "history": [],
            "state": SessionState.CREATED.value,
        }
        self.r.set(self._key(sid), json.dumps(payload), ex=DEFAULT_TTL_SEC)
        return sid

    def get(self, sid: str) -> Optional[Dict[str, Any]]:
        raw = self.r.get(self._key(sid))
        return json.loads(raw) if raw else None

    def set_current(self, sid: str, qid: int) -> None:
        s = self.get(sid)
        if not s:
            return
        # transition to ACTIVE on first question
        if s.get("state") == SessionState.CREATED.value:
            s["state"] = SessionState.ACTIVE.value
        if s.get("state") != SessionState.ACTIVE.value:
            return
        s["current_qid"] = qid
        if qid not in s["asked_ids"]:
            s["asked_ids"].append(qid)
        self.r.set(self._key(sid), json.dumps(s), ex=DEFAULT_TTL_SEC)

    def add_answer(self, sid: str, entry: Dict[str, Any]) -> None:
        s = self.get(sid)
        if not s:
            return
        if s.get("state") != SessionState.ACTIVE.value:
            return
        s["history"].append(entry)
        self.r.set(self._key(sid), json.dumps(s), ex=DEFAULT_TTL_SEC)

    def finish(self, sid: str) -> Dict[str, Any]:
        s = self.get(sid) or {}
        s["end_ts"] = int(time.time() * 1000)
        s["state"] = SessionState.FINISHED.value
        self.r.set(self._key(sid), json.dumps(s), ex=DEFAULT_TTL_SEC)
        return s


