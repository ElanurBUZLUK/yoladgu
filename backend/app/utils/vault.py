from __future__ import annotations

from typing import Optional
import hvac
from app.core.config import settings


def get_secret(key: str) -> Optional[str]:
    addr = settings.VAULT_ADDR
    token = settings.VAULT_TOKEN
    path = settings.VAULT_KV_PATH
    if not (addr and token and path):
        return None
    try:
        client = hvac.Client(url=addr, token=token)
        resp = client.secrets.kv.v2.read_secret_version(path=path)
        data = (resp.get("data") or {}).get("data") or {}
        return data.get(key)
    except Exception:
        return None


