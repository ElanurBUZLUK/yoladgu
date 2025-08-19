from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class SystemInitializationService:
    """
    Açılış kontrolleri ve (opsiyonel) onarım:
    - pgvector extension
    - embeddings.embedding vector(…)
    - cosine uyumlu IVFFLAT index
    apply=True ise eksikleri oluşturmaya çalışır (yetki gerekebilir).
    """

    def __init__(
        self,
        table_name: str = "embeddings",
        dim: int = settings.embedding_dimension,
        index_name: str = "embeddings_embedding_cos_ivf_idx",
    ):
        self.table_name = table_name
        self.dim = dim
        self.index_name = index_name

    async def check_pgvector(self, db: AsyncSession) -> bool:
        q = text("SELECT 1 FROM pg_extension WHERE extname='vector'")
        res = await db.execute(q)
        return res.first() is not None

    async def ensure_extension(self, db: AsyncSession) -> bool:
        try:
            await db.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            return True
        except Exception as e:
            logger.warning("CREATE EXTENSION failed (permission issue?): %s", e)
            return False

    async def column_exists(self, db: AsyncSession, column: str) -> bool:
        q = text("""
            SELECT 1
            FROM information_schema.columns
            WHERE table_name=:t AND column_name=:c
        """)
        res = await db.execute(q.bindparams(t=self.table_name, c=column))
        return res.first() is not None

    async def index_exists(self, db: AsyncSession, index_name: Optional[str] = None) -> bool:
        idx = index_name or self.index_name
        q = text("SELECT 1 FROM pg_indexes WHERE indexname=:i")
        res = await db.execute(q.bindparams(i=idx))
        return res.first() is not None

    async def ensure_column(self, db: AsyncSession, column: str = "embedding") -> bool:
        if await self.column_exists(db, column):
            return True
        try:
            await db.execute(text(f"ALTER TABLE {self.table_name} ADD COLUMN {column} vector({self.dim});"))
            return True
        except Exception as e:
            logger.error("ALTER TABLE add vector column failed: %s", e)
            return False

    async def ensure_index(self, db: AsyncSession, concurrently: bool = False) -> bool:
        if await self.index_exists(db, self.index_name):
            return True
        try:
            conc = "CONCURRENTLY " if concurrently else ""
            await db.execute(text(f"""
                CREATE INDEX {conc}{self.index_name}
                ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """))
            await db.execute(text(f"ANALYZE {self.table_name};"))
            return True
        except Exception as e:
            logger.error("CREATE INDEX failed: %s", e)
            return False

    async def run_all(self, db: AsyncSession, apply: bool = False) -> Dict[str, Any]:
        report = {"extension": False, "vector_column": False, "index": False, "applied": apply, "errors": []}

        try:
            ok = await self.check_pgvector(db)
            if not ok and apply:
                ok = await self.ensure_extension(db)
            report["extension"] = ok
        except Exception as e:
            logger.exception("pgvector check/ensure failed: %s", e)
            report["errors"].append(str(e))

        try:
            col_ok = await self.column_exists(db, "embedding")
            if not col_ok and apply:
                col_ok = await self.ensure_column(db, "embedding")
            report["vector_column"] = col_ok
        except Exception as e:
            logger.exception("vector column check/ensure failed: %s", e)
            report["errors"].append(str(e))

        try:
            idx_ok = await self.index_exists(db, self.index_name)
            if not idx_ok and apply:
                idx_ok = await self.ensure_index(db, concurrently=False)
            report["index"] = idx_ok
        except Exception as e:
            logger.exception("index check/ensure failed: %s", e)
            report["errors"].append(str(e))

        if apply:
            try:
                await db.commit()
            except Exception as e:
                await db.rollback()
                report["errors"].append(f"commit -> {e}")

        return report

system_initialization_service = SystemInitializationService()
