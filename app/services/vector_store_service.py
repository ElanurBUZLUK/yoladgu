"""
Kalıcı Vektör Depolama Servisi - pgvector ile
K-En Yakın Komşu (KNN) aramaları için optimize edilmiş vektör veritabanı
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import asyncpg
import structlog
from app.core.config import settings

logger = structlog.get_logger()


@dataclass
class VectorSearchResult:
    """Vektör arama sonucu"""

    id: int
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    distance: float


class VectorStoreService:
    """
    pgvector kullanarak kalıcı vektör depolama ve KNN arama servisi
    """

    def __init__(self):
        self.embedding_dim = 384  # SBERT paraphrase-MiniLM-L6-v2 boyutu
        self.pool = None

    async def initialize(self):
        """Async connection pool başlat"""
        try:
            self.pool = await asyncpg.create_pool(
                settings.DATABASE_URL.replace(
                    "postgresql+psycopg2://", "postgresql://"
                ),
                min_size=settings.ASYNCPG_MIN_SIZE,
                max_size=settings.ASYNCPG_MAX_SIZE,
                timeout=settings.ASYNCPG_TIMEOUT,
                server_settings={
                    "application_name": "yoladgu_vector_store",
                    "jit": "off",  # Disable JIT for better performance on small queries
                },
            )
            await self._ensure_tables_exist()
            logger.info("vector_store_initialized", embedding_dim=self.embedding_dim)
        except Exception as e:
            logger.error("vector_store_init_error", error=str(e))
            raise

    async def _ensure_tables_exist(self):
        """pgvector extension ve tabloları oluştur"""
        async with self.pool.acquire() as conn:
            # pgvector extension'ı aktifleştir
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Embedding tablosu
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS question_embeddings (
                    id SERIAL PRIMARY KEY,
                    question_id INTEGER UNIQUE NOT NULL,
                    embedding vector(%s) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    subject_id INTEGER,
                    topic_id INTEGER,
                    difficulty_level INTEGER,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """,
                self.embedding_dim,
            )

            # Optimized HNSW index for KNN performance
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS question_embeddings_embedding_idx
                ON question_embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {settings.HNSW_M}, ef_construction = {settings.HNSW_EF_CONSTRUCTION});
            """
            )

            # Set HNSW search parameters for this session
            await conn.execute(f"SET hnsw.ef_search = {settings.HNSW_EF_SEARCH};")

            # Metadata filtreleme için index'ler
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS question_embeddings_subject_idx
                ON question_embeddings (subject_id);
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS question_embeddings_difficulty_idx
                ON question_embeddings (difficulty_level);
            """
            )

            logger.info("vector_tables_ensured")

    async def store_embedding(
        self,
        question_id: int,
        embedding: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        subject_id: Optional[int] = None,
        topic_id: Optional[int] = None,
        difficulty_level: Optional[int] = None,
    ) -> bool:
        """Tek bir embedding'i sakla"""
        try:
            if len(embedding) != self.embedding_dim:
                raise ValueError(
                    f"Embedding boyutu {len(embedding)}, beklenen {self.embedding_dim}"
                )

            metadata = metadata or {}

            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO question_embeddings
                    (question_id, embedding, content, metadata, subject_id, topic_id, difficulty_level)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (question_id)
                    DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        subject_id = EXCLUDED.subject_id,
                        topic_id = EXCLUDED.topic_id,
                        difficulty_level = EXCLUDED.difficulty_level,
                        updated_at = NOW()
                """,
                    question_id,
                    embedding,
                    content,
                    json.dumps(metadata),
                    subject_id,
                    topic_id,
                    difficulty_level,
                )

            logger.debug("embedding_stored", question_id=question_id)
            return True

        except Exception as e:
            logger.error("store_embedding_error", question_id=question_id, error=str(e))
            return False

    async def batch_store_embeddings(
        self, embeddings_data: List[Dict[str, Any]]
    ) -> int:
        """Toplu embedding saklama"""
        stored_count = 0

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for data in embeddings_data:
                        await conn.execute(
                            """
                            INSERT INTO question_embeddings
                            (question_id, embedding, content, metadata, subject_id, topic_id, difficulty_level)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (question_id)
                            DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                content = EXCLUDED.content,
                                metadata = EXCLUDED.metadata,
                                subject_id = EXCLUDED.subject_id,
                                topic_id = EXCLUDED.topic_id,
                                difficulty_level = EXCLUDED.difficulty_level,
                                updated_at = NOW()
                        """,
                            data["question_id"],
                            data["embedding"],
                            data["content"],
                            json.dumps(data.get("metadata", {})),
                            data.get("subject_id"),
                            data.get("topic_id"),
                            data.get("difficulty_level"),
                        )
                        stored_count += 1

            logger.info("batch_embeddings_stored", count=stored_count)
            return stored_count

        except Exception as e:
            logger.error("batch_store_error", error=str(e))
            return stored_count

    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        use_index_scan: bool = True,
    ) -> List[VectorSearchResult]:
        """
        KNN semantic similarity search

        Args:
            query_embedding: Sorgu embedding'i
            k: Kaç sonuç döneceği
            similarity_threshold: Minimum benzerlik skoru
            filters: subject_id, difficulty_level vb. filtreler
        """
        try:
            if len(query_embedding) != self.embedding_dim:
                raise ValueError(
                    f"Query embedding boyutu {len(query_embedding)}, beklenen {self.embedding_dim}"
                )

            # WHERE koşulları oluştur
            where_conditions = []
            params = [query_embedding, k]
            param_counter = 3

            if filters:
                if "subject_id" in filters:
                    where_conditions.append(f"subject_id = ${param_counter}")
                    params.append(filters["subject_id"])
                    param_counter += 1

                if "difficulty_level" in filters:
                    where_conditions.append(f"difficulty_level = ${param_counter}")
                    params.append(filters["difficulty_level"])
                    param_counter += 1

                if "exclude_question_ids" in filters:
                    excluded_ids = filters["exclude_question_ids"]
                    if excluded_ids:
                        placeholders = ",".join(
                            [f"${param_counter + i}" for i in range(len(excluded_ids))]
                        )
                        where_conditions.append(f"question_id NOT IN ({placeholders})")
                        params.extend(excluded_ids)
                        param_counter += len(excluded_ids)

            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

            # Performance hint for large datasets
            index_hint = (
                "/*+ IndexScan(question_embeddings question_embeddings_embedding_idx) */"
                if use_index_scan
                else ""
            )

            # Optimized query with early termination
            query = f"""
                {index_hint}
                SELECT
                    question_id,
                    content,
                    metadata,
                    subject_id,
                    topic_id,
                    difficulty_level,
                    1 - (embedding <=> $1) as similarity_score,
                    embedding <=> $1 as distance
                FROM question_embeddings
                WHERE {where_clause}
                    AND (embedding <=> $1) <= {1.0 - similarity_threshold}
                ORDER BY embedding <=> $1
                LIMIT $2
            """

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            results = []
            for row in rows:
                results.append(
                    VectorSearchResult(
                        id=row["question_id"],
                        content=row["content"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        similarity_score=float(row["similarity_score"]),
                        distance=float(row["distance"]),
                    )
                )

            logger.debug(
                "similarity_search_completed",
                query_dim=len(query_embedding),
                results_count=len(results),
                k=k,
                threshold=similarity_threshold,
            )

            return results

        except Exception as e:
            logger.error("similarity_search_error", error=str(e))
            return []

    async def get_embedding_by_question_id(
        self, question_id: int
    ) -> Optional[List[float]]:
        """Soru ID'sine göre embedding getir"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT embedding
                    FROM question_embeddings
                    WHERE question_id = $1
                """,
                    question_id,
                )

            return list(row["embedding"]) if row else None

        except Exception as e:
            logger.error("get_embedding_error", question_id=question_id, error=str(e))
            return None

    async def delete_embedding(self, question_id: int) -> bool:
        """Embedding'i sil"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM question_embeddings WHERE question_id = $1
                """,
                    question_id,
                )

            return result == "DELETE 1"

        except Exception as e:
            logger.error(
                "delete_embedding_error", question_id=question_id, error=str(e)
            )
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Vektör deposu istatistikleri"""
        try:
            async with self.pool.acquire() as conn:
                stats = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total_embeddings,
                        COUNT(DISTINCT subject_id) as subjects_count,
                        COUNT(DISTINCT difficulty_level) as difficulty_levels_count,
                        AVG(array_length(embedding, 1)) as avg_embedding_dim
                    FROM question_embeddings
                """
                )

            return dict(stats) if stats else {}

        except Exception as e:
            logger.error("get_stats_error", error=str(e))
            return {}

    async def close(self):
        """Connection pool'u kapat"""
        if self.pool:
            await self.pool.close()
            logger.info("vector_store_closed")


# Global instance
vector_store_service = VectorStoreService()
