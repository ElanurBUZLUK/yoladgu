# app/utils/distlock_idem.py
from __future__ import annotations
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, TypeVar, Coroutine

from redis.asyncio import Redis

T = TypeVar("T")

# ---------------------------
# Low-level: Distributed Lock
# ---------------------------

_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
else
  return 0
end
"""

@dataclass
class LockResult:
    acquired: bool
    token: Optional[str] = None

class RedisLock:
    """
    Async distributed lock using Redis SET NX PX + Lua-based safe release.
    """
    def __init__(
        self,
        client: Redis,
        key: str,
        ttl_ms: int = 10_000,
        token: Optional[str] = None,
    ):
        self.client = client
        self.key = key
        self.ttl_ms = ttl_ms
        self.token = token or f"{uuid.uuid4()}"

    async def acquire(self) -> bool:
        # SET key value NX PX ttl
        ok = await self.client.set(self.key, self.token, nx=True, px=self.ttl_ms)
        return bool(ok)

    async def release(self) -> bool:
        # Safe compare-and-delete
        script = self.client.register_script(_RELEASE_LUA)
        res = await script(keys=[self.key], args=[self.token])
        return res == 1

    async def extend(self, ttl_ms: Optional[int] = None) -> bool:
        # Optional lease extension
        ttl = ttl_ms or self.ttl_ms
        pipe = self.client.pipeline()
        pipe.watch(self.key)
        val = await self.client.get(self.key)
        if val is None or val.decode() != self.token:
            await pipe.reset()
            return False
        pipe.multi()
        pipe.pexpire(self.key, ttl)
        try:
            await pipe.execute()
            return True
        except Exception:
            return False
        finally:
            await pipe.reset()

    async def __aenter__(self):
        ok = await self.acquire()
        if not ok:
            raise TimeoutError(f"Could not acquire lock: {self.key}")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.release()


async def acquire_with_retry(
    client: Redis,
    key: str,
    ttl_ms: int = 10_000,
    wait_timeout_ms: int = 5_000,
    backoff_ms: int = 100,
) -> LockResult:
    """
    Spin with jittered backoff until acquired or timeout.
    """
    token = f"{uuid.uuid4()}"
    lock = RedisLock(client, key, ttl_ms, token)
    start = time.monotonic()
    sleep = backoff_ms / 1000.0

    while (time.monotonic() - start) * 1000 < wait_timeout_ms:
        if await lock.acquire():
            return LockResult(True, token)
        # jitter
        await asyncio.sleep(sleep)
        sleep = min(sleep * 1.5, 0.8)  # cap backoff
    return LockResult(False, None)


def lock_decorator(
    key_builder: Callable[..., str],
    ttl_ms: int = 10_000,
    wait_timeout_ms: int = 5_000,
    backoff_ms: int = 100,
    redis_client_factory: Callable[[], Redis] | None = None,
):
    """
    Decorator to guard a function with a Redis distributed lock.

    Example:
        @lock_decorator(lambda user_id: f"lock:reindex:{user_id}")
        async def rebuild_index(user_id: str): ...
    """
    def wrapper(func: Callable[..., Coroutine[Any, Any, T]]):
        async def inner(*args, **kwargs) -> T:
            key = key_builder(*args, **kwargs)
            client = redis_client_factory() if redis_client_factory else Redis.from_url("redis://redis:6379/0")
            lr = await acquire_with_retry(client, key, ttl_ms, wait_timeout_ms, backoff_ms)
            if not lr.acquired:
                raise TimeoutError(f"Lock busy: {key}")
            lock = RedisLock(client, key, ttl_ms, lr.token)
            try:
                return await func(*args, **kwargs)
            finally:
                try:
                    await lock.release()
                except Exception:
                    # swallow release error to not mask real exceptions
                    pass
        return inner
    return wrapper


# ---------------------------
# Idempotency Utilities
# ---------------------------

# States and key layout:
# idem:{scope}:{key} -> {
#   "status": "in_progress|done",
#   "result": <json-serializable>,
#   "updated_at": epoch_ms
# }

@dataclass
class IdempotencyConfig:
    scope: str = "default"        # e.g. "payments", "embeddings"
    ttl_seconds: int = 3600       # cache lifetime for completed results
    in_progress_ttl: int = 300    # how long to wait for ongoing work
    wait_timeout_ms: int = 5000   # how long caller should wait polling
    poll_interval_ms: int = 150   # polling cadence

def _idem_key(scope: str, key: str) -> str:
    return f"idem:{scope}:{key}"

async def _json_get(client: Redis, k: str) -> Optional[dict]:
    raw = await client.get(k)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None

async def _json_set(client: Redis, k: str, value: dict, ex: Optional[int] = None) -> None:
    await client.set(k, json.dumps(value, ensure_ascii=False), ex=ex)


async def idempotent_singleflight(
    client: Redis,
    key: str,
    config: IdempotencyConfig,
    worker: Callable[[], Awaitable[dict]],
) -> dict:
    """
    Ensures only one worker executes for a given (scope,key).
    Others will either wait for result or return last completed value.
    Result MUST be JSON-serializable dict.
    """
    k = _idem_key(config.scope, key)

    # fast path: completed
    snapshot = await _json_get(client, k)
    if snapshot and snapshot.get("status") == "done":
        return snapshot.get("result", {})

    # mark in-progress if not marked
    token = str(uuid.uuid4())
    # Use SET NX to avoid racing; we store minimal state first
    ok = await client.set(k, json.dumps({"status": "in_progress", "token": token, "updated_at": int(time.time()*1000)}), nx=True, ex=config.in_progress_ttl)
    if ok:
        # We are the leader; run worker
        try:
            result = await worker()
            payload = {
                "status": "done",
                "result": result,
                "updated_at": int(time.time()*1000)
            }
            await _json_set(client, k, payload, ex=config.ttl_seconds)
            return result
        except Exception:
            # reset in_progress quickly so others can retry
            await client.delete(k)
            raise

    # follower: someone else is doing the work → wait/poll or return last done
    start = time.monotonic()
    interval = config.poll_interval_ms / 1000.0
    while (time.monotonic() - start) * 1000 < config.wait_timeout_ms:
        await asyncio.sleep(interval)
        snap = await _json_get(client, k)
        if not snap:
            # leader failed & cleared → give caller chance to try again
            break
        if snap.get("status") == "done":
            return snap.get("result", {})
    # timeout: give up and let caller retry (or you can 409)
    raise TimeoutError("Idempotent operation still in progress, please retry later.")


def idempotency_decorator(
    key_builder: Callable[..., str],
    config: IdempotencyConfig = IdempotencyConfig(),
    redis_client_factory: Callable[[], Redis] | None = None,
):
    """
    Decorator that caches JSON-serializable results per Idempotency-Key.
    Suitable for POST endpoints where the client sends a stable key.
    """
    def wrapper(func: Callable[..., Awaitable[dict]]):
        async def inner(*args, **kwargs) -> dict:
            k = key_builder(*args, **kwargs)
            client = redis_client_factory() if redis_client_factory else Redis.from_url("redis://redis:6379/0")

            async def worker():
                return await func(*args, **kwargs)

            return await idempotent_singleflight(client, k, config, worker)
        return inner
    return wrapper
