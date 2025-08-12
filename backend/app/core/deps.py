from functools import lru_cache
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.services.embedding_service import EmbeddingService
from app.services.vector_index_manager import VectorIndexManager
from app.core.config import settings
from app.services.bandit.linucb import LinUCBService
from app.services.online.ftrl import FTRLService
from app.core.db import get_db
from app.models import User
from app.utils.auth import decode_token
from app.services.peer_hardness import PeerHardnessService, PeerStore, PeerParams, CosineWeightedStrategy
from app.services.peer_hardness_store import RedisPeerStore
from app.utils.vault import get_secret

@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    # Pull provider/model from Vault if present
    prov = get_secret("EMBEDDING_PROVIDER") or getattr(settings, "EMBEDDING_PROVIDER", None)
    model = get_secret("EMBEDDING_MODEL") or getattr(settings, "EMBEDDING_MODEL", None)
    if prov:
        import os
        os.environ["EMBEDDING_PROVIDER"] = prov
    if model:
        import os
        os.environ["EMBEDDING_MODEL_ID"] = model
    return EmbeddingService(settings.EMBED_DIM)

@lru_cache(maxsize=1)
def get_index_manager() -> VectorIndexManager:
    return VectorIndexManager(settings.REDIS_URL)

@lru_cache(maxsize=1)
def get_linucb_service() -> LinUCBService:
    return LinUCBService(feature_dim=16, alpha=0.2, redis_url=settings.REDIS_URL)

@lru_cache(maxsize=1)
def get_ftrl_service() -> FTRLService:
    return FTRLService(dim=32, alpha=0.1, beta=1.0, l1=0.0, l2=1.0, redis_url=settings.REDIS_URL)


class _InMemoryPeerStore(PeerStore):
    # Placeholder minimal store; replace with Redis/PG-backed implementation
    def __init__(self):
        self._wrong: dict[int, dict[int, float]] = {}
        self._right: dict[int, dict[int, float]] = {}

    def wrong_set(self, student_id: int) -> dict[int, float]:
        return self._wrong.get(student_id, {})

    def right_set(self, student_id: int) -> dict[int, float]:
        return self._right.get(student_id, {})

    def has_attempt(self, student_id: int, question_id: int) -> bool:
        return question_id in self._wrong.get(student_id, {}) or question_id in self._right.get(student_id, {})

    def sim_index_neighbors(self, student_id: int):
        # naive: everyone else
        for sid in set(self._wrong.keys()) | set(self._right.keys()):
            if sid != student_id:
                yield sid


_peer_store_singleton: _InMemoryPeerStore | None = None


@lru_cache(maxsize=1)
def get_peer_service() -> PeerHardnessService:
    backend = getattr(settings, "PEER_STORE_BACKEND", "memory")
    if backend == "redis":
        store = RedisPeerStore(settings.REDIS_URL, prefix=getattr(settings, "PEER_STORE_PREFIX", "peer"))
        return PeerHardnessService(store=store, params=PeerParams())
    # default to in-memory
    global _peer_store_singleton
    if _peer_store_singleton is None:
        _peer_store_singleton = _InMemoryPeerStore()
    # Example: swap strategy via setting in future
    return PeerHardnessService(store=_peer_store_singleton, params=PeerParams())


# Authentication & Authorization dependencies
async def get_current_user(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub"))
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    res = await db.execute(select(User).where(User.id == user_id))
    user = res.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_roles(*allowed_roles: str):
    async def _checker(user: User = Depends(get_current_user)) -> User:
        if allowed_roles and user.role not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return user
    return _checker
