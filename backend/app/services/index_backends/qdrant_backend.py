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
    from qdrant_client.models import Distance, VectorParams, PointStruct, CreateAlias, CreateAliasOperation, DeleteAlias, DeleteAliasOperation, AliasOperations
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore
    Distance = None  # type: ignore
    VectorParams = None  # type: ignore
    PointStruct = None  # type: ignore
    CreateAlias = None  # type: ignore
    CreateAliasOperation = None  # type: ignore
    DeleteAlias = None  # type: ignore
    DeleteAliasOperation = None  # type: ignore
    AliasOperations = None  # type: ignore


_DISTANCE_MAP = {
    "l2": Distance.EUCLID if Distance else None,
    "cosine": Distance.COSINE if Distance else None,
    "dot": Distance.DOT if Distance else None,
}


class QdrantBackend(IndexBackend):
    def __init__(self, dim: int, url: str, collection_name: str, active_alias: Optional[str] = None, space: str = "l2"):
        if QdrantClient is None:
            raise RuntimeError("qdrant-client is not installed. Add it to requirements and install.")
        self.dim = int(dim)
        self.url = url
        self.collection = collection_name
        self.alias = active_alias
        self.space = space
        self.client = QdrantClient(url=self.url)

    # For remote vector DB, build means (re)create collection and upsert all points
    def build(self, embeddings: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        dist = _DISTANCE_MAP.get(self.space, Distance.EUCLID if Distance else None)
        if dist is None:
            raise RuntimeError("qdrant-client distance mapping unavailable")
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
        if AliasOperations is None:
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


