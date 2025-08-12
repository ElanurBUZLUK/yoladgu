from typing import List, Tuple, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, CreateAlias, CreateAliasOperation, DeleteAlias, DeleteAliasOperation, AliasOperations
from .base import IndexBackend


_DISTANCE_MAP = {
    "l2": Distance.EUCLID,
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
}


class QdrantBackend(IndexBackend):
    def __init__(self, dim: int, url: str, collection_name: str, active_alias: Optional[str] = None, space: str = "l2"):
        self.dim = dim
        self.url = url
        self.collection = collection_name
        self.alias = active_alias
        self.space = space
        self.client = QdrantClient(url=self.url)

    # For remote vector DB, build means (re)create collection and upsert all points
    def build(self, embeddings: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        dist = _DISTANCE_MAP.get(self.space, Distance.EUCLID)
        # Recreate collection with correct vector params
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=int(self.dim), distance=dist),
        )
        # Upsert points
        if ids is None:
            ids = np.arange(embeddings.shape[0], dtype=np.int64)
        points = [
            PointStruct(id=int(i), vector=np.asarray(v, dtype=np.float32).tolist())
            for i, v in zip(ids.tolist(), embeddings.astype("float32"))
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    # No-op for remote backend
    def save(self, path: str) -> None:
        return None

    # No-op for remote backend
    def load(self, path: str) -> None:
        return None

    def search(self, query: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        q = query.reshape(1, -1).astype("float32")[0].tolist()
        res = self.client.search(collection_name=self.collection, query_vector=q, limit=int(k))
        if not res:
            return [], []
        ids = [int(p.id) for p in res]
        dists = [float(p.score) for p in res]
        return ids, dists

    def set_param(self, **kwargs) -> None:
        # Placeholder: could adjust HNSW/M optimizer params if needed via Qdrant API
        return None

    def set_alias_active(self) -> None:
        if not self.alias:
            return
        # Make alias point to this collection atomically (delete then create)
        ops = AliasOperations(
            operations=[
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=self.alias)),
                CreateAliasOperation(create_alias=CreateAlias(collection_name=self.collection, alias_name=self.alias)),
            ]
        )
        try:
            self.client.update_collection_aliases(ops)
        except Exception:
            # Try creating without delete if delete fails
            self.client.update_collection_aliases(
                AliasOperations(operations=[CreateAliasOperation(create_alias=CreateAlias(collection_name=self.collection, alias_name=self.alias))])
            )

    def stats(self) -> dict:
        try:
            count = self.client.count(self.collection, exact=True)
            total = int(getattr(count, "count", 0))
        except Exception:
            total = 0
        return {
            "type": "qdrant",
            "collection": self.collection,
            "alias": self.alias,
            "dim": int(self.dim),
            "ntotal": total,
            "space": self.space,
        }

from typing import List, Tuple, Optional
import numpy as np

from .base import IndexBackend

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore
    rest = None  # type: ignore


class QdrantBackend(IndexBackend):
    def __init__(self, dim: int, url: str, collection_name: str, alias_active: Optional[str] = None):
        if QdrantClient is None:
            raise RuntimeError("qdrant-client is not installed. Add it to requirements and install.")
        self.dim = int(dim)
        self.url = url
        self.collection_name = collection_name
        self.alias_active = alias_active
        self.client = QdrantClient(url=self.url)
        self.search_params: dict = {}

    # In Qdrant, build means (re)creating the collection and uploading all points
    def build(self, embeddings: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        vectors = embeddings.astype("float32")
        if ids is None:
            point_ids: List[int] = list(range(vectors.shape[0]))
        else:
            point_ids = [int(x) for x in ids.tolist()]

        # Recreate collection to ensure a clean build
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=self.dim, distance=rest.Distance.COSINE),
            )
        except Exception:
            # Fallback if recreate is unavailable in current server version
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=self.dim, distance=rest.Distance.COSINE),
            )

        # Upload points in batches
        self.client.upload_collection(
            collection_name=self.collection_name,
            vectors=vectors.tolist(),
            ids=point_ids,
            batch_size=512,
            parallel=1,
        )

    # No-op for save/load in remote service backend
    def save(self, path: str) -> None:  # noqa: ARG002
        return

    def load(self, path: str) -> None:  # noqa: ARG002
        # Best-effort check: create collection if missing so object is usable
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=self.dim, distance=rest.Distance.COSINE),
            )

    def search(self, query: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        qv = query.reshape(1, -1).astype("float32")[0].tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=qv,
            limit=int(k),
            with_payload=False,
            search_params=self.search_params or None,
        )

        ids: List[int] = []
        dists: List[float] = []
        for pt in results:
            pid = pt.id
            try:
                pid_int = int(pid)  # ids may come as str or int
            except Exception:
                # Unsupported id type for our pipeline
                continue
            # Convert cosine score (similarity, higher is better) into a distance-like measure
            score = float(pt.score)
            distance_like = 1.0 - max(min(score, 1.0), -1.0)
            ids.append(pid_int)
            dists.append(distance_like)
        return ids, dists

    def set_param(self, **kwargs) -> None:
        # Allow passing arbitrary parameters to qdrant search
        # Common: {"hnsw_ef": 128, "exact": False}
        self.search_params.update({k: v for k, v in kwargs.items() if v is not None})

    def stats(self) -> dict:
        count = 0
        try:
            count = int(self.client.count(self.collection_name, exact=True).count)
        except Exception:
            pass
        return {
            "type": "qdrant",
            "collection": self.collection_name,
            "dim": self.dim,
            "count": count,
        }

    # Helper for blue/green swaps: point alias to this collection
    def promote_alias(self) -> None:
        if not self.alias_active:
            return
        try:
            # Ensure alias points to this collection
            try:
                self.client.delete_alias(alias_name=self.alias_active)
            except Exception:
                pass
            self.client.create_alias(collection_name=self.collection_name, alias_name=self.alias_active)
        except Exception:
            # Non-fatal; swap can still be tracked via Redis active key
            pass


