import hashlib
import json
import logging
from redis import Redis
from typing import Optional

logger = logging.getLogger(__name__)

class SemanticCache:
    def __init__(self, redis_client: Redis, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
        self.hit_count = 0
        self.miss_count = 0

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

    def get(self, prompt: str) -> Optional[dict]:
        key = f"semcache:{self._hash_prompt(prompt)}"
        cached = self.redis.get(key)
        if cached:
            self.hit_count += 1
            return json.loads(cached)
        self.miss_count += 1
        return None

    def set(self, prompt: str, response: dict):
        key = f"semcache:{self._hash_prompt(prompt)}"
        self.redis.setex(key, self.ttl, json.dumps(response))
    
    def metrics(self) -> dict:
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0.0
        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": hit_rate,
            "size": self.redis.dbsize()
        }