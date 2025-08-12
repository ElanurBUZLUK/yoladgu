from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
from sqlalchemy import text

from app.core.db import SessionLocal  # AsyncSession factory


class PgVectorBackend:
    def __init__(self, dim: int, table_name: str = "embeddings") -> None:
        self.dim = int(dim)
        self.table = table_name

    # Build: (re)create or upsert rows
    def build(self, embeddings: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        if embeddings is None or embeddings.size == 0:
            return
        vecs = embeddings.astype("float32")
        ids_list: List[int]
        if ids is None:
            ids_list = list(range(vecs.shape[0]))
        else:
            ids_list = [int(x) for x in ids.tolist()]

        async def _run():
            async with SessionLocal() as db:  # type: ignore
                # Ensure extension/table exist (idempotent)
                await db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await db.execute(text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table} (
                      id INT PRIMARY KEY,
                      embedding vector({self.dim}) NOT NULL,
                      content JSONB
                    );
                    """
                ))
                # Upsert in batches
                batch = 512
                for i in range(0, len(ids_list), batch):
                    ids_chunk = ids_list[i:i+batch]
                    vecs_chunk = vecs[i:i+batch]
                    # Build VALUES list
                    values = ",".join(
                        [
                            f"({int(pid)}, '{np.array(v, dtype=np.float32).tolist()}')"  # content omitted
                            for pid, v in zip(ids_chunk, vecs_chunk)
                        ]
                    )
                    # Use explicit casting from json to vector via to_json and pgvector input, or use parameterized?
                    # Here we use vector literal via array syntax
                    # Transform Python list -> PostgreSQL array literal
                    def to_pg_vector(lst: list[float]) -> str:
                        return "[" + ",".join(f"{x}" for x in lst) + "]"
                    values = ",".join(
                        [
                            f"({int(pid)}, '{to_pg_vector(list(np.asarray(v, dtype=np.float32)))}')"
                            for pid, v in zip(ids_chunk, vecs_chunk)
                        ]
                    )
                    sql = text(
                        f"""
                        INSERT INTO {self.table} (id, embedding)
                        VALUES {values}
                        ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding
                        """
                    )
                    await db.execute(sql)
                await db.commit()

        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # In server context, run synchronously by creating a task and waiting
            loop.run_until_complete(_run())  # type: ignore
        else:
            asyncio.run(_run())

    def save(self, path: str) -> None:
        # Remote DB backend: no local save
        return None

    def load(self, path: str) -> None:
        # Ensure table exists
        async def _run():
            async with SessionLocal() as db:  # type: ignore
                await db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await db.execute(text(
                    f"CREATE TABLE IF NOT EXISTS {self.table} (id INT PRIMARY KEY, embedding vector({self.dim}) NOT NULL, content JSONB)"
                ))
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            loop.run_until_complete(_run())  # type: ignore
        else:
            asyncio.run(_run())

    def search(self, query: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        q = query.reshape(1, -1).astype("float32")[0]
        # Use parameterized query to avoid SQL injection; async call wrapped sync
        async def _run() -> Tuple[List[int], List[float]]:
            async with SessionLocal() as db:  # type: ignore
                # Build vector literal
                qvec = "[" + ",".join(str(float(x)) for x in q.tolist()) + "]"
                sql = text(
                    f"SELECT id, (embedding <-> :qvec) AS dist FROM {self.table} ORDER BY embedding <-> :qvec LIMIT :k"
                )
                res = await db.execute(sql.bindparams(qvec=qvec, k=int(k)))
                rows = res.fetchall()
                ids = [int(r[0]) for r in rows]
                dists = [float(r[1]) for r in rows]
                return ids, dists
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            return loop.run_until_complete(_run())  # type: ignore
        return asyncio.run(_run())

    def stats(self) -> dict:
        # Count rows best-effort
        async def _run() -> int:
            try:
                async with SessionLocal() as db:  # type: ignore
                    res = await db.execute(text(f"SELECT COUNT(1) FROM {self.table}"))
                    return int(res.scalar() or 0)
            except Exception:
                return 0
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        total = loop.run_until_complete(_run()) if loop and loop.is_running() else asyncio.run(_run())
        return {"type": "pgvector", "dim": self.dim, "count": total, "table": self.table}


