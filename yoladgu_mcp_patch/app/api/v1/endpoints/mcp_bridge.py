from fastapi import APIRouter, HTTPException, Query, Body, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os, glob, json, re
import redis
import boto3
from botocore.config import Config as BotoConfig
import psycopg2
from psycopg2.extras import RealDictCursor
from app.core.rate_limit import SlidingWindowRateLimiter

router = APIRouter(prefix="/mcp", tags=["mcp"])

# Basit rate-limit (IP yerine proses-yerel anahtar kullanıyor; prod’da gateway önerilir)
rl_redis = SlidingWindowRateLimiter(max_calls=10, window_seconds=10)
rl_s3    = SlidingWindowRateLimiter(max_calls=6,  window_seconds=10)
rl_sql   = SlidingWindowRateLimiter(max_calls=5,  window_seconds=10)

def _require_env(keys: List[str]) -> List[str]:
    missing = []
    for k in keys:
        if not os.getenv(k):
            missing.append(k)
    return missing

@router.get("/configs")
def list_configs():
    cfg_dir = os.path.join(os.getcwd(), "mcp", "configs")
    files = sorted(glob.glob(os.path.join(cfg_dir, "*.json")))
    items = []
    missing_envs = set()
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            # env alanındaki ${KEY}’leri bul
            env_map = data.get("env", {})
            for v in env_map.values():
                m = re.findall(r"\${([^}]+)}", v) if isinstance(v, str) else []
                for k in m: 
                    if not os.getenv(k):
                        missing_envs.add(k)
            items.append({"name": data.get("name"), "file": os.path.basename(f)})
        except Exception as e:
            items.append({"file": os.path.basename(f), "error": str(e)})
    return {"configs": items, "missing_env": sorted(missing_envs)}

class RedisKeysIn(BaseModel):
    pattern: str = Field(default="*", description="SCAN pattern")
    limit: int = Field(default=100, le=1000)

def _redis():
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url, decode_responses=True)

@router.post("/redis/keys")
def redis_keys(body: RedisKeysIn):
    if not rl_redis.allow("keys"):
        raise HTTPException(status_code=429, detail="rate limit")
    r = _redis()
    out = []
    cursor = 0
    scanned = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=body.pattern, count=min(body.limit, 1000))
        out.extend(keys)
        scanned += len(keys)
        if cursor == 0 or len(out) >= body.limit:
            break
    return {"pattern": body.pattern, "keys": out[:body.limit], "scanned": scanned}

@router.get("/redis/queue-stats")
def redis_queue_stats(keys: str = Query(..., description="Comma separated redis keys")):
    if not rl_redis.allow("queue-stats"):
        raise HTTPException(status_code=429, detail="rate limit")
    r = _redis()
    res = {}
    for k in [x.strip() for x in keys.split(",") if x.strip()]:
        length = None
        try:
            length = r.llen(k)
        except Exception:
            length = None
        if not length:
            try:
                length = r.xlen(k)
            except Exception:
                pass
        res[k] = int(length or 0)
    return {"stats": res}

class S3ListIn(BaseModel):
    bucket: Optional[str] = Field(default=None, description="Varsa .env S3_BUCKET_NAME override")
    prefix: str = ""
    limit: int = 200

def _s3_client():
    cfg = BotoConfig(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"})
    return boto3.client("s3", config=cfg)

@router.post("/s3/list")
def s3_list(body: S3ListIn):
    if not rl_s3.allow("s3-list"):
        raise HTTPException(status_code=429, detail="rate limit")
    bucket = body.bucket or os.getenv("S3_BUCKET_NAME")
    if not bucket:
        raise HTTPException(400, "bucket gerekli (ya body.bucket ya S3_BUCKET_NAME)")
    s3 = _s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(Bucket=bucket, Prefix=body.prefix, PaginationConfig={"PageSize": min(body.limit, 1000)}):
        for obj in page.get("Contents", []):
            out.append({"key": obj["Key"], "size": obj["Size"]})
            if len(out) >= body.limit:
                return {"bucket": bucket, "prefix": body.prefix, "objects": out}
    return {"bucket": bucket, "prefix": body.prefix, "objects": out}

class FSImportIn(BaseModel):
    base_path: str = "./imports"
    patterns: List[str] = ["*.pdf", "*.md"]
    dry_run: bool = True

@router.post("/filesystem/import")
def fs_import(body: FSImportIn):
    if not os.path.isdir(body.base_path):
        raise HTTPException(400, f"Klasör yok: {body.base_path}")
    matched = []
    for pat in body.patterns:
        glob_pat = os.path.join(body.base_path, "**", pat)
        for f in glob.glob(glob_pat, recursive=True):
            matched.append(os.path.abspath(f))
    matched = matched[:1000]
    # Not: burada ingest kuyruğuna ekleme noktası (Celery/RQ) boş.
    return {"found": len(matched), "files": matched, "dry_run": body.dry_run}

class SQLRunIn(BaseModel):
    sql: str
    limit: int = Field(ge=1, le=1000, default=100)

def _pg_conn():
    dsn = os.getenv("PG_DSN")
    if not dsn:
        raise RuntimeError("PG_DSN env yok")
    return psycopg2.connect(dsn, cursor_factory=RealDictCursor)

@router.post("/sql/run")
def sql_run(body: SQLRunIn):
    if not rl_sql.allow("sql-run"):
        raise HTTPException(status_code=429, detail="rate limit")
    sql = body.sql.strip().rstrip(";")
    if not re.match(r"(?is)^select\s", sql):
        raise HTTPException(status_code=400, detail="Sadece SELECT sorgularına izin var")
    # güvenlik: LIMIT zorunlu; yoksa ekleyelim
    if not re.search(r"(?is)\blimit\b", sql):
        sql = f"{sql} LIMIT {body.limit}"
    try:
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                return {"rows": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
