from __future__ import annotations

import asyncio
from typing import List
import numpy as np
from sqlalchemy import select, text
from app.core.db import SessionLocal
from app.models import Question
from app.core.config import settings
from app.services.embedding_service import EmbeddingService


MODEL_NAME = settings.EMBEDDING_MODEL or "sentence-transformers/all-MiniLM-L6-v2"


async def seed_embeddings(batch_size: int = 128) -> None:
    embedder = EmbeddingService(dim=settings.EMBED_DIM)
    # Force SBERT mode if configured
    embedder.mode = "sbert"
    async with SessionLocal() as db:  # type: ignore
        # Ensure pgvector table exists
        await db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await db.execute(text(
            f"CREATE TABLE IF NOT EXISTS embeddings (id INT PRIMARY KEY, embedding vector({int(settings.EMBED_DIM)}) NOT NULL, content JSONB)"
        ))
        # Fetch all questions (assuming question text stored in Redis; fallback to metadata fields if any)
        res = await db.execute(select(Question.id))
        qids: List[int] = [int(x) for x in res.scalars().all()]
        # Best-effort text retrieval from Redis metadata via QuestionsService
        from app.services.content.questions_service import QuestionsService
        qsvc = QuestionsService(settings.REDIS_URL)

        def to_text(qid: int) -> str:
            meta = qsvc.get(qid) or {}
            title = meta.get("title") or ""
            text = meta.get("text") or ""
            topic = str(meta.get("topic") or meta.get("topic_id") or "")
            return f"{title}\n{text}\n[TOPIC:{topic}]".strip()

        for i in range(0, len(qids), batch_size):
            batch_ids = qids[i:i+batch_size]
            texts = [to_text(qid) for qid in batch_ids]
            # Compute embeddings (sync wrapper)
            X = embedder.embed_sync(texts)
            # Insert
            # Build VALUES for parameterized insert
            values = ",".join([
                "(" + ",".join([
                    str(int(qid)),
                    "'[" + ",".join(str(float(x)) for x in X[j].astype(float).tolist()) + "]'",
                    "NULL",
                ]) + ")"
                for j, qid in enumerate(batch_ids)
            ])
            sql = text(
                f"""
                INSERT INTO embeddings (id, embedding, content)
                VALUES {values}
                ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding
                """
            )
            await db.execute(sql)
            await db.commit()
            print(f"Processed {i + len(batch_ids)}/{len(qids)}")


def main() -> None:
    asyncio.run(seed_embeddings())


if __name__ == "__main__":
    main()


