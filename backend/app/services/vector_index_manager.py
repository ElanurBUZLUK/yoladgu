import os, threading
from typing import Tuple, List, Optional, Dict
import numpy as np, redis
from app.core.config import settings
from app.services.index_backends.hnsw_backend import HNSWBackend
from app.services.index_backends.faiss_backend import FaissIVFPQBackend

class VectorIndexManager:
    def __init__(self, redis_url: str):
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)
        os.makedirs(settings.VECTOR_INDEX_DIR, exist_ok=True)
        self._lock = threading.RLock()
        self._active = self._get_active_from_redis()
        self._backends: Dict[str, Optional[object]] = {"blue": None, "green": None}
        self._tombstone_key = settings.VECTOR_STORE_TOMBSTONE_KEY
        for name in ("blue","green"):
            self._try_load_slot(name)

    def _make_backend(self):
        if settings.VECTOR_BACKEND == "faiss":
            return FaissIVFPQBackend(settings.EMBED_DIM, settings.FAISS_NLIST, settings.FAISS_PQ_M, settings.FAISS_PQ_BITS, settings.FAISS_NPROBE)
        else:
            return HNSWBackend(settings.EMBED_DIM, settings.HNSW_SPACE, settings.HNSW_M, settings.HNSW_EF_CONSTRUCT, settings.HNSW_EF_SEARCH)

    def _slot_path(self, slot: str) -> str:
        ext = "faiss" if settings.VECTOR_BACKEND == "faiss" else "hnsw"
        return os.path.join(settings.VECTOR_INDEX_DIR, f"index_{slot}.{ext}")

    def _get_active_from_redis(self) -> str:
        val = self.r.get(settings.VECTOR_ACTIVE_KEY)
        return val or settings.VECTOR_BLUE_NAME

    def _set_active_to_redis(self, val: str) -> None:
        self.r.set(settings.VECTOR_ACTIVE_KEY, val)

    def _try_load_slot(self, slot: str) -> None:
        path = self._slot_path(slot)
        if os.path.exists(path):
            be = self._make_backend()
            be.load(path)
            self._backends[slot] = be

    def active_slot(self) -> str:
        with self._lock:
            return self._active

    def inactive_slot(self) -> str:
        return settings.VECTOR_GREEN_NAME if self.active_slot() == settings.VECTOR_BLUE_NAME else settings.VECTOR_BLUE_NAME

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[List[int], List[float]]:
        with self._lock:
            be = self._backends.get(self._active)
        if be is None:
            return [], []
        ids, dists = be.search(query, k + 50)
        deleted = {int(x) for x in self.r.smembers(self._tombstone_key)}
        filtered = [(i,d) for i,d in zip(ids,dists) if i != -1 and i not in deleted][:k]
        if not filtered: return [], []
        ids_out, dist_out = zip(*filtered)
        return list(ids_out), list(dist_out)

    def stats(self) -> dict:
        with self._lock:
            a = self._active
            be_a = self._backends.get(a)
            be_i = self._backends.get(self.inactive_slot())
        return {
            "backend": settings.VECTOR_BACKEND,
            "active": a,
            "active_loaded": be_a is not None,
            "inactive_loaded": be_i is not None,
            "active_stats": be_a.stats() if be_a else {},
            "inactive_stats": be_i.stats() if be_i else {},
        }

    def build_on_inactive(self, embeddings: np.ndarray, ids: Optional[np.ndarray]) -> dict:
        slot = self.inactive_slot()
        be = self._make_backend()
        be.build(embeddings, ids)
        path = self._slot_path(slot)
        be.save(path)
        with self._lock:
            self._backends[slot] = be
        return {"built_slot": slot, "path": path}

    def swap(self) -> dict:
        with self._lock:
            new_active = self.inactive_slot()
            if self._backends.get(new_active) is None:
                self._try_load_slot(new_active)
                if self._backends.get(new_active) is None:
                    return {"swapped": False, "reason": "inactive slot not built"}
            self._active = new_active
            self._set_active_to_redis(new_active)
        return {"swapped": True, "active": new_active}

    # Tombstone management
    def add_tombstone(self, ids: List[int]) -> dict:
        if not ids:
            return {"ok": True, "added": 0}
        pipe = self.r.pipeline()
        for i in ids:
            pipe.sadd(self._tombstone_key, int(i))
        added = pipe.execute()
        return {"ok": True, "added": sum(1 for x in added if x)}

    def remove_tombstone(self, ids: List[int]) -> dict:
        if not ids:
            return {"ok": True, "removed": 0}
        pipe = self.r.pipeline()
        for i in ids:
            pipe.srem(self._tombstone_key, int(i))
        removed = pipe.execute()
        return {"ok": True, "removed": sum(1 for x in removed if x)}

    def list_tombstones(self) -> dict:
        vals = sorted(int(x) for x in self.r.smembers(self._tombstone_key))
        return {"count": len(vals), "ids": vals}
