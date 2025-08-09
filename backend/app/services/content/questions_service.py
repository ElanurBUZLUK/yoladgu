import json, redis, random
from typing import Dict, Any, List, Optional
from app.core.config import settings


class QuestionsService:
    def __init__(self, redis_url: str):
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)

    def _key(self, qid: int) -> str:
        return f"question:{qid}"

    def get(self, qid: int) -> Optional[Dict[str, Any]]:
        raw = self.r.get(self._key(qid))
        return json.loads(raw) if raw else None

    def exists(self, qid: int) -> bool:
        return self.r.exists(self._key(qid)) == 1

    def random_by_topic(self, topic_id: Optional[int], k: int = 50) -> List[int]:
        keys = self.r.keys("question:*")
        ids = []
        for kkey in keys:
            try:
                qid = int(kkey.split(":")[-1])
                data = self.get(qid)
                if data and (topic_id is None or data.get("topic_id") == topic_id):
                    ids.append(qid)
            except Exception:
                continue
        random.shuffle(ids)
        return ids[:k]


