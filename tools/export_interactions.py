"""
Export interactions from the database into JSONL for CF training.

Each line: {"user_id": int, "question_id": int, "label": int}

Usage:
  python tools/export_interactions.py --out examples/interactions.jsonl --lookback_days 180
"""

import argparse
import asyncio
import json
from datetime import datetime, timedelta
from typing import AsyncIterator

from sqlalchemy import select

from app.core.db import SessionLocal
from app.models import Attempt


async def _iter_attempts_since(since: datetime, chunk_size: int = 1000) -> AsyncIterator[list[Attempt]]:
    async with SessionLocal() as db:  # type: ignore
        # naive chunking by attempt_id ranges
        last_id = 0
        while True:
            rows = (
                await db.execute(
                    select(Attempt)
                    .where(Attempt.created_at >= since, Attempt.attempt_id > last_id)
                    .order_by(Attempt.attempt_id.asc())
                    .limit(chunk_size)
                )
            ).scalars().all()
            if not rows:
                break
            last_id = rows[-1].attempt_id
            yield rows


async def export_interactions(out_path: str, lookback_days: int = 180) -> int:
    since = datetime.utcnow() - timedelta(days=int(lookback_days))
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        async for rows in _iter_attempts_since(since):
            for a in rows:
                rec = {
                    "user_id": int(a.student_id),
                    "question_id": int(a.question_id),
                    "label": int(1 if a.is_correct else 0),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/interactions.jsonl")
    ap.add_argument("--lookback_days", type=int, default=180)
    args = ap.parse_args()
    n = asyncio.get_event_loop().run_until_complete(export_interactions(args.out, args.lookback_days))
    print(f"Exported {n} interactions to {args.out}")


if __name__ == "__main__":
    main()


