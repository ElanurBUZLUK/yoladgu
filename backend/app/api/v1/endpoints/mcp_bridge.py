from fastapi import APIRouter, HTTPException, Query, Body, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os, glob, json, re
import redis
import boto3
from botocore.config import Config as BotoConfig
import psycopg2
from psycopg2.extras import RealDictCursor
from app.core.deps import require_roles
import numpy as np
from app.core.config import settings
from app.services.vector_index_manager import VectorIndexManager
from app.services.embedding_api import get_embedding_provider
from app.services.content.questions_service import QuestionsService

router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.get("/configs")
def list_configs(user=Depends(require_roles("admin"))):
    cfg_dir = os.path.join(os.getcwd(), "mcp", "configs")
    files = sorted(glob.glob(os.path.join(cfg_dir, "*.json")))
    items = []
    missing_envs = set()
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
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
def redis_keys(body: RedisKeysIn, user=Depends(require_roles("admin"))):
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


class S3ListIn(BaseModel):
    bucket: Optional[str] = Field(default=None, description="Varsa .env S3_BUCKET_NAME override")
    prefix: str = ""
    limit: int = 200


def _s3_client():
    cfg = BotoConfig(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"})
    return boto3.client("s3", config=cfg)


@router.post("/s3/list")
def s3_list(body: S3ListIn, user=Depends(require_roles("admin"))):
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


class SQLRunIn(BaseModel):
    sql: str
    limit: int = Field(ge=1, le=1000, default=100)


def _pg_conn():
    dsn = os.getenv("PG_DSN")
    if not dsn:
        raise RuntimeError("PG_DSN env yok")
    return psycopg2.connect(dsn, cursor_factory=RealDictCursor)


@router.post("/sql/run")
def sql_run(body: SQLRunIn, user=Depends(require_roles("admin"))):
    sql = body.sql.strip().rstrip(";")
    if not re.match(r"(?is)^select\s", sql):
        raise HTTPException(status_code=400, detail="Sadece SELECT sorgularına izin var")
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


class RetrieveIn(BaseModel):
    text: str
    language: str = Field(default="tr")
    k: int = Field(default=5, ge=1, le=50)


@router.post("/retrieve")
def mcp_retrieve(body: RetrieveIn):
    # Minimal MCP-like HTTP endpoint to serve retrieval contexts
    try:
        idx = VectorIndexManager(settings.REDIS_URL)
        embed = get_embedding_provider()
        qsvc = QuestionsService(settings.REDIS_URL)
        vec = np.array([embed.embed_one(body.text)], dtype=np.float32)
        ids, _ = idx.search(vec[0], k=max(5, body.k))
        results: List[Dict[str, Any]] = []
        for qid in ids[: body.k]:
            meta = qsvc.get(int(qid)) or {}
            results.append({"id": int(qid), "text": meta.get("text", ""), "meta": meta})
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

