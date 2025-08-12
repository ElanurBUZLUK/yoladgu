from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os
import boto3
from sqlalchemy import text
from app.core.config import settings
from app.core.db import SessionLocal


def export_table(name: str, lookback_days: int, batch_size: int) -> dict:
    cutoff = datetime.utcnow() - timedelta(days=int(lookback_days))
    key_prefix = f"archive/{name}/dt={cutoff.strftime('%Y-%m')}"
    total = 0

    async def _run() -> int:
        async with SessionLocal() as db:  # type: ignore
            last_id = 0
            count = 0
            while True:
                q = await db.execute(
                    text(f"SELECT * FROM {name} WHERE created_at < :cut AND id > :lid ORDER BY id ASC LIMIT :lim"),
                    {"cut": cutoff, "lid": last_id, "lim": int(batch_size)},
                )
                rows = q.fetchall()
                if not rows:
                    break
                # Serialize rows as NDJSON for simplicity
                payload = "\n".join(json.dumps(dict(r._mapping), default=str) for r in rows)
                last_id = int(rows[-1][0])
                _upload_s3(f"{key_prefix}/{name}_{last_id}.ndjson", payload)
                count += len(rows)
            return count

    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    total = loop.run_until_complete(_run()) if loop and loop.is_running() else asyncio.run(_run())
    return {"table": name, "archived": total, "prefix": key_prefix}


def _upload_s3(key: str, data: str) -> None:
    bucket = settings.S3_BUCKET or os.getenv("S3_BUCKET")
    region = settings.AWS_REGION or os.getenv("AWS_REGION")
    if not bucket:
        return
    s3 = boto3.client("s3", region_name=region)
    s3.put_object(Bucket=bucket, Key=key, Body=data.encode("utf-8"), ContentType="application/x-ndjson")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="events,attempts")
    ap.add_argument("--lookback_days", type=int, default=settings.ARCHIVE_LOOKBACK_DAYS)
    ap.add_argument("--batch_size", type=int, default=settings.ARCHIVE_BATCH_SIZE)
    args = ap.parse_args()
    res = []
    for t in [x.strip() for x in args.tables.split(",") if x.strip()]:
        res.append(export_table(t, args.lookback_days, args.batch_size))
    print(json.dumps({"results": res}))


if __name__ == "__main__":
    main()


