import os, threading
from typing import Tuple, List, Optional, Dict
import numpy as np, redis
from app.core.config import settings
from app.services.index_backends.hnsw_backend import HNSWBackend
from app.services.index_backends.faiss_backend import FaissIVFPQBackend
try:
    from app.services.index_backends.qdrant_backend import QdrantBackend
except Exception:
    QdrantBackend = None  # type: ignore
try:
    from app.services.index_backends.pgvector_backend import PgVectorBackend
except Exception:
    PgVectorBackend = None  # type: ignore

class VectorIndexManager:
    def __init__(self, redis_url: str):
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)
        os.makedirs(settings.VECTOR_INDEX_DIR, exist_ok=True)
        self._lock = threading.RLock()
        self._active = self._get_active_from_redis()
        self._backends: Dict[str, Optional[object]] = {"blue": None, "green": None}
        # Sharded pgvector backends
        self._pg_shards: int = int(getattr(settings, "VECTOR_SHARDS", 1) or 1)
        self._pg_backends_shards: Optional[Dict[int, object]] = None
        self._tombstone_key = settings.VECTOR_STORE_TOMBSTONE_KEY
        for name in ("blue","green"):
            self._try_load_slot(name)

    def _make_backend(self, slot: Optional[str] = None):
        # Guard: ensure dimension metadata aligns with index files to avoid corrupt writes
        dim = settings.EMBED_DIM
        if settings.VECTOR_BACKEND == "faiss":
            return FaissIVFPQBackend(dim, settings.FAISS_NLIST, settings.FAISS_PQ_M, settings.FAISS_PQ_BITS, settings.FAISS_NPROBE)
        if settings.VECTOR_BACKEND == "qdrant":
            if QdrantBackend is None:
                raise RuntimeError("qdrant backend selected but qdrant-client is not installed")
            # If slot not specified, default to inactive for build flows
            if slot is None:
                slot = self.inactive_slot()
            collection = settings.QDRANT_COLLECTION_BLUE if slot == settings.VECTOR_BLUE_NAME else settings.QDRANT_COLLECTION_GREEN
            return QdrantBackend(dim=dim, url=settings.QDRANT_URL, collection_name=collection, active_alias=settings.QDRANT_ALIAS_ACTIVE)
        if settings.VECTOR_BACKEND == "pgvector":
            if PgVectorBackend is None:
                raise RuntimeError("pgvector backend selected but module not available")
            # For sharded mode, we instantiate per-shard elsewhere
            return PgVectorBackend(dim=dim)
        # default
        return HNSWBackend(dim, settings.HNSW_SPACE, settings.HNSW_M, settings.HNSW_EF_CONSTRUCT, settings.HNSW_EF_SEARCH)

    def _slot_path(self, slot: str) -> str:
        if settings.VECTOR_BACKEND == "faiss":
            ext = "faiss"
        elif settings.VECTOR_BACKEND == "hnsw":
            ext = "hnsw"
        else:
            # qdrant is remote; keep a lightweight meta file extension
            ext = "qdrant"
        return os.path.join(settings.VECTOR_INDEX_DIR, f"index_{slot}.{ext}")

    def _get_active_from_redis(self) -> str:
        val = self.r.get(settings.VECTOR_ACTIVE_KEY)
        return val or settings.VECTOR_BLUE_NAME

    def _set_active_to_redis(self, val: str) -> None:
        self.r.set(settings.VECTOR_ACTIVE_KEY, val)

    def _try_load_slot(self, slot: str) -> None:
        path = self._slot_path(slot)
        # For qdrant, presence of the meta file is enough to attempt load
        exists = os.path.exists(path) if settings.VECTOR_BACKEND != "qdrant" else os.path.exists(f"{path}.meta.json")
        if exists:
            be = self._make_backend(slot)
            # Optional: check persisted dim meta if available alongside the index
            meta_path = f"{path}.meta.json"
            try:
                if os.path.exists(meta_path):
                    import json
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    file_dim = int(meta.get("dim", settings.EMBED_DIM))
                    if file_dim != settings.EMBED_DIM:
                        # Don't load mismatched index
                        return
            except Exception:
                pass
            be.load(path)
            self._backends[slot] = be

    def active_slot(self) -> str:
        with self._lock:
            return self._active

    def inactive_slot(self) -> str:
        return settings.VECTOR_GREEN_NAME if self.active_slot() == settings.VECTOR_BLUE_NAME else settings.VECTOR_BLUE_NAME

    def search(self, query: np.ndarray, k: int = 10) -> Tuple[List[int], List[float]]:
        # Sharded pgvector scatter-gather path
        if settings.VECTOR_BACKEND == "pgvector" and self._pg_shards > 1:
            with self._lock:
                if self._pg_backends_shards is None:
                    # lazy init shard backends
                    self._pg_backends_shards = {}
                    for i in range(self._pg_shards):
                        self._pg_backends_shards[i] = PgVectorBackend(dim=settings.EMBED_DIM, table_name=f"embeddings_{i}")  # type: ignore[arg-type]
            all_pairs: List[Tuple[int, float]] = []
            for i, be_shard in self._pg_backends_shards.items():  # type: ignore[union-attr]
                try:
                    s_ids, s_d = be_shard.search(query, k)
                    all_pairs.extend(list(zip(s_ids, s_d)))
                except Exception:
                    continue
            if not all_pairs:
                return [], []
            # Deduplicate by id keeping best (lowest distance)
            best: Dict[int, float] = {}
            for i, d in all_pairs:
                di = float(d)
                if i not in best or di < best[i]:
                    best[i] = di
            merged = sorted(best.items(), key=lambda x: x[1])[:k]
            ids = [i for i, _ in merged]
            dists = [d for _, d in merged]
        else:
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
        # Sharded pgvector build: route by id % shards
        if settings.VECTOR_BACKEND == "pgvector" and self._pg_shards > 1 and ids is not None:
            if embeddings.shape[1] != settings.EMBED_DIM:
                raise ValueError(f"embedding dim mismatch: got {embeddings.shape[1]}, expected {settings.EMBED_DIM}")
            # Prepare per-shard batches
            shard_to_idx: Dict[int, List[int]] = {i: [] for i in range(self._pg_shards)}
            for idx, qid in enumerate(ids.tolist()):
                shard_to_idx[int(qid) % self._pg_shards].append(idx)
            # Build per shard
            for shard, idxs in shard_to_idx.items():
                if not idxs:
                    continue
                be = PgVectorBackend(dim=settings.EMBED_DIM, table_name=f"embeddings_{shard}")  # type: ignore[arg-type]
                X = embeddings[idxs, :]
                I = ids[idxs]
                be.build(X, I)
            return {"built_shards": self._pg_shards}
        # Default single-backend path
        slot = self.inactive_slot()
        if embeddings.shape[1] != settings.EMBED_DIM:
            raise ValueError(f"embedding dim mismatch: got {embeddings.shape[1]}, expected {settings.EMBED_DIM}")
        be = self._make_backend(slot)
        be.build(embeddings, ids)
        path = self._slot_path(slot)
        try:
            be.save(path)
        except Exception:
            pass
        try:
            import json, time
            meta = {
                "provider": os.getenv("EMBEDDING_PROVIDER", "unknown"),
                "model_id": os.getenv("EMBEDDING_MODEL_ID", "unknown"),
                "dim": int(settings.EMBED_DIM),
                "created_at": int(time.time()),
            }
            with open(f"{path}.meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)
        except Exception:
            pass
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
            # For qdrant, point alias to the new active collection if available
            if settings.VECTOR_BACKEND == "qdrant":
                be = self._backends.get(new_active)
                try:
                    if hasattr(be, "set_alias_active"):
                        be.set_alias_active()
                except Exception:
                    pass
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
