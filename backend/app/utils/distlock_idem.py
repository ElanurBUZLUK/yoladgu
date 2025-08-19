# app/utils/distlock_idem.py
from __future__ import annotations
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, TypeVar, Coroutine
import logging # Added logging import

from redis.asyncio import Redis
from app.core.config import settings

T = TypeVar("T")

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
        ok = await self.client.set(self.key, self.token, nx=True, px=self.ttl_ms)
        return bool(ok)

    async def release(self) -> bool:
        script = self.client.register_script(_RELEASE_LUA)
        res = await script(keys=[self.key], args=[self.token])
        return res == 1

    async def extend(self, ttl_ms: Optional[int] = None) -> bool:
        ttl = ttl_ms or self.ttl_ms
        pipe = self.client.pipeline()
        await pipe.watch(self.key)
        val = await self.client.get(self.key)
        if val is None or (isinstance(val, bytes) and val.decode() != self.token) and val != self.token:
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
        try:
            await self.release()
        except Exception:
            pass


async def acquire_with_retry(
    client: Redis,
    key: str,
    ttl_ms: int = 10_000,
    wait_timeout_ms: int = 5_000,
    backoff_ms: int = 100,
) -> LockResult:
    token = f"{uuid.uuid4()}"
    lock = RedisLock(client, key, ttl_ms, token)
    start = time.monotonic()
    sleep = backoff_ms / 1000.0
    while (time.monotonic() - start) * 1000 < wait_timeout_ms:
        if await lock.acquire():
            return LockResult(True, token)
        await asyncio.sleep(sleep)
        sleep = min(sleep * 1.5, 0.8)
    return LockResult(False, None)


def lock_decorator(
    key_builder: Callable[..., str],
    ttl_ms: int = 10_000,
    wait_timeout_ms: int = 5_000,
    backoff_ms: int = 100,
    redis_client_factory: Optional[Callable[[], Redis]] = None,
):
    """
    Decorator to guard a function with a Redis distributed lock.
    """
    def wrapper(func: Callable[..., Coroutine[Any, Any, T]]):
        async def inner(*args, **kwargs) -> T:
            key = key_builder(*args, **kwargs)
            client = redis_client_factory() if redis_client_factory else Redis.from_url(settings.redis_url)
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
                    pass
        return inner
    return wrapper


# ---------------------------
# Idempotency Utilities
# ---------------------------

@dataclass
class IdempotencyConfig:
    scope: str = "default"
    ttl_seconds: int = 3600
    in_progress_ttl: int = 300
    wait_timeout_ms: int = 5000
    poll_interval_ms: int = 150

def _idem_key(scope: str, key: str) -> str:
    return f"idem:{scope}:{key}"

async def _json_get(client: Redis, k: str) -> Optional[dict]:
    raw = await client.get(k)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception as e:
        logging.error(f"Error decoding JSON from Redis key {k}: {e}", exc_info=True)
        return None

async def _json_set(client: Redis, k: str, value: dict, ex: Optional[int] = None) -> None:
    try:
        await client.set(k, json.dumps(value, ensure_ascii=False), ex=ex)
    except Exception as e:
        logging.error(f"Error encoding or setting JSON to Redis key {k}: {e}", exc_info=True)

async def idempotent_singleflight(
    client: Redis,
    key: str,
    config: IdempotencyConfig,
    worker: Callable[[], Awaitable[dict]],
) -> dict:
    k = _idem_key(config.scope, key)
    logger = logging.getLogger(__name__) # Initialize logger inside function

    logger.debug(f"Attempting idempotent single-flight for key: {k}")

    snapshot = await _json_get(client, k)
    if snapshot:
        status = snapshot.get("status")
        if status == "done":
            logger.debug(f"Found 'done' status for key {k}. Returning cached result.")
            return snapshot.get("result", {})
        elif status == "in_progress":
            logger.debug(f"Found 'in_progress' status for key {k}. Waiting for result.")
            # Continue to polling loop
        else:
            logger.warning(f"Unexpected status '{status}' for key {k}. Proceeding to acquire lock.")
    else:
        logger.debug(f"No existing state found for key {k}. Attempting to acquire lock.")

    token = str(uuid.uuid4())
    try:
        ok = await client.set(k, json.dumps({"status": "in_progress", "token": token, "updated_at": int(time.time()*1000)}),
                              nx=True, ex=config.in_progress_ttl)
    except Exception as e:
        logger.error(f"Error setting 'in_progress' state for key {k}: {e}", exc_info=True)
        raise

    if ok:
        logger.debug(f"Acquired 'in_progress' lock for key {k} with token {token}. Executing worker.")
        try:
            result = await worker()
            payload = {"status": "done", "result": result, "updated_at": int(time.time()*1000)}
            await _json_set(client, k, payload, ex=config.ttl_seconds)
            logger.debug(f"Worker completed for key {k}. State set to 'done'.")
            return result
        except Exception as e:
            logger.error(f"Worker failed for key {k}: {e}. Deleting 'in_progress' state.", exc_info=True)
            await client.delete(k)
            raise
    else:
        logger.debug(f"Failed to acquire 'in_progress' lock for key {k}. Another process is in progress. Polling for result.")
        start = time.monotonic()
        interval = config.poll_interval_ms / 1000.0
        while (time.monotonic() - start) * 1000 < config.wait_timeout_ms:
            await asyncio.sleep(interval)
            snap = await _json_get(client, k)
            if not snap:
                logger.debug(f"State for key {k} disappeared during polling. Assuming worker failed or expired.")
                break
            status = snap.get("status")
            if status == "done":
                logger.debug(f"Found 'done' status during polling for key {k}. Returning result.")
                return snap.get("result", {})
            elif status != "in_progress":
                logger.warning(f"Unexpected status '{status}' for key {k} during polling. Breaking loop.")
                break # Unexpected state, break and let it timeout or retry
            logger.debug(f"Still 'in_progress' for key {k}. Polling again.")
        
        logger.warning(f"Idempotent operation for key {k} still in progress after {config.wait_timeout_ms}ms. Timeout.")
        raise TimeoutError("Idempotent operation still in progress, please retry later.")

def idempotency_decorator(
    key_builder: Callable[..., str],
    config: IdempotencyConfig = IdempotencyConfig(),
    redis_client_factory: Optional[Callable[[], Redis]] = None,
):
    def wrapper(func: Callable[..., Awaitable[dict]]):
        async def inner(*args, **kwargs) -> dict:
            k = key_builder(*args, **kwargs)
            client = redis_client_factory() if redis_client_factory else Redis.from_url(settings.redis_url)
            async def worker():
                return await func(*args, **kwargs)
            return await idempotent_singleflight(client, k, config, worker)
        return inner
    return wrapper