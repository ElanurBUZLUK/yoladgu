"""
Enhanced SBERT Embedding Service
Gelişmiş semantic similarity ve embedding yönetimi
"""

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import redis
import structlog
from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import vector_store_service
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger()


def safe_int(val, default=0):
    try:
        if isinstance(val, int):
            return val
        # SQLAlchemy Column: get value if possible
        if hasattr(val, "__int__"):
            return int(val)
        # Sometimes val could be a descriptor, try to get value
        if hasattr(val, "default") and hasattr(val.default, "arg"):
            return int(val.default.arg)
        return int(val)
    except Exception:
        return default


class EnhancedEmbeddingService(EmbeddingService):
    """Gelişmiş SBERT Embedding Servisi"""

    def __init__(self):
        super().__init__()

        # Optimized Redis connection pool for caching
        self.redis_pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        self.redis = redis.Redis(connection_pool=self.redis_pool)
        self.cache_ttl = settings.CACHE_EMBEDDING_TTL

        # Vector Store için initialization flag
        self.vector_store_initialized = False

        # Available models
        self.available_models = {
            "small": {
                "name": "paraphrase-MiniLM-L6-v2",
                "dimensions": 384,
                "description": "Fast, lightweight model",
            },
            "medium": {
                "name": "paraphrase-mpnet-base-v2",
                "dimensions": 768,
                "description": "Balanced speed and quality",
            },
            "large": {
                "name": "paraphrase-multilingual-mpnet-base-v2",
                "dimensions": 768,
                "description": "Multilingual support",
            },
        }

        # Current model
        self.current_model_key = "small"
        self._model_name = self.available_models[self.current_model_key]["name"]
        self.embedding_dim = self.available_models[self.current_model_key]["dimensions"]

        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "embeddings_computed": 0,
            "similarities_computed": 0,
        }

    @property
    def model_name(self) -> str:
        """Get current model name (property to avoid protected namespace conflict)"""
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        """Set current model name"""
        self._model_name = value

    def switch_model(self, model_key: str) -> bool:
        """SBERT modelini değiştir"""
        if model_key not in self.available_models:
            logger.error(
                "invalid_model_key",
                available=list(self.available_models.keys()),
                requested=model_key,
            )
            return False

        try:
            with self.model_lock:
                old_model = self.current_model_key
                self.current_model_key = model_key
                self._model_name = self.available_models[model_key]["name"]
                self.embedding_dim = self.available_models[model_key]["dimensions"]
                self.model = None  # Force reload

                # Test new model
                self._load_model()

                logger.info(
                    "model_switched",
                    from_model=old_model,
                    to_model=model_key,
                    new_dimensions=self.embedding_dim,
                )
                return True

        except Exception as e:
            logger.error("model_switch_error", error=str(e))
            return False

    def get_embedding_cache_key(self, text: str) -> str:
        """Cache key oluştur"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{self.current_model_key}:{text_hash}"

    async def compute_embedding_cached(self, text: str) -> List[float]:
        """Cache'li embedding hesaplama"""
        if not text or not text.strip():
            return [0.0] * self.embedding_dim

        # Optimized cache check with pipeline
        cache_key = self.get_embedding_cache_key(text)

        try:
            cached_result = await self.redis.get(cache_key)
            if cached_result:
                self.stats["cache_hits"] += 1
                if isinstance(cached_result, bytes):
                    return json.loads(cached_result.decode("utf-8"))
                else:
                    return json.loads(cached_result)
        except Exception as e:
            logger.debug("cache_read_error", error=str(e))

        # Cache miss - hesapla
        self.stats["cache_misses"] += 1
        embedding = self.compute_embedding(text)

        # Optimized cache write with pipeline
        try:
            await self.redis.setex(cache_key, self.cache_ttl, json.dumps(embedding))
        except Exception as e:
            logger.debug("cache_write_error", error=str(e))

        self.stats["embeddings_computed"] += 1
        return embedding

    async def compute_embeddings_batch_cached_async(
        self, texts: List[str]
    ) -> List[List[float]]:
        """Cache'li toplu embedding hesaplama"""
        if not texts:
            return []

        results = []
        uncached_texts = []
        uncached_indices = []

        # Optimized batch cache lookup with Redis pipeline
        cache_keys = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append([0.0] * self.embedding_dim)
                cache_keys.append(None)
            else:
                cache_key = self.get_embedding_cache_key(text)
                cache_keys.append(cache_key)
                results.append(None)  # Placeholder

        # Batch cache lookup
        valid_keys = [key for key in cache_keys if key is not None]
        if valid_keys:
            try:
                pipe = self.redis.pipeline()
                for key in valid_keys:
                    pipe.get(key)
                cached_results = pipe.execute()

                key_index = 0
                for i, text in enumerate(texts):
                    if cache_keys[i] is not None:
                        cached_result = cached_results[key_index]
                        key_index += 1

                        if cached_result:
                            try:
                                if isinstance(cached_result, bytes):
                                    results[i] = json.loads(
                                        cached_result.decode("utf-8")
                                    )
                                else:
                                    results[i] = json.loads(cached_result)
                                self.stats["cache_hits"] += 1
                            except json.JSONDecodeError:
                                uncached_texts.append(text)
                                uncached_indices.append(i)
                                self.stats["cache_misses"] += 1
                        else:
                            uncached_texts.append(text)
                            uncached_indices.append(i)
                            self.stats["cache_misses"] += 1

            except Exception as e:
                logger.debug("batch_cache_read_error", error=str(e))
                # Fallback: treat all as cache miss
                for i, text in enumerate(texts):
                    if cache_keys[i] is not None and results[i] is None:
                        uncached_texts.append(text)
                        uncached_indices.append(i)

        # Cache'de olmayan metinleri hesapla
        if uncached_texts:
            uncached_embeddings = self.compute_embeddings_batch(uncached_texts)

            # Optimized batch cache write with pipeline
            try:
                pipe = self.redis.pipeline()
                for i, (text, embedding) in enumerate(
                    zip(uncached_texts, uncached_embeddings)
                ):
                    index = uncached_indices[i]
                    results[index] = embedding

                    # Batch cache write
                    cache_key = self.get_embedding_cache_key(text)
                    pipe.setex(cache_key, self.cache_ttl, json.dumps(embedding))

                pipe.execute()

            except Exception as e:
                logger.debug("batch_cache_write_error", error=str(e))

            self.stats["embeddings_computed"] += len(uncached_texts)

        return results

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """İki metin arasındaki semantic benzerlik"""
        try:
            embeddings = self.compute_embeddings_batch([text1, text2])
            if len(embeddings) != 2:
                return 0.0

            similarity = cosine_similarity(
                np.array(embeddings[0]).reshape(1, -1),
                np.array(embeddings[1]).reshape(1, -1),
            )[0][0]

            self.stats["similarities_computed"] += 1
            return float(similarity)

        except Exception as e:
            logger.error("semantic_similarity_error", error=str(e))
            return 0.0

    def batch_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Metinler arası similarity matrisi"""
        try:
            embeddings = self.compute_embeddings_batch(texts)
            if not embeddings:
                return np.array([])

            embedding_matrix = np.array(embeddings)
            similarity_matrix = cosine_similarity(embedding_matrix)

            self.stats["similarities_computed"] += len(texts) ** 2
            return similarity_matrix

        except Exception as e:
            logger.error("similarity_matrix_error", error=str(e))
            return np.array([])

    def semantic_clustering(self, texts: List[str], n_clusters: int = 5) -> Dict:
        """Metinleri semantic olarak kümelere ayır"""
        try:
            if len(texts) < n_clusters:
                logger.warning(
                    "insufficient_texts_for_clustering",
                    texts_count=len(texts),
                    clusters_requested=n_clusters,
                )
                return {"clusters": [], "labels": [], "centroids": []}

            # Embeddings'i hesapla
            embeddings = self.compute_embeddings_batch(texts)
            embedding_matrix = np.array(embeddings)

            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(embedding_matrix)

            # Sonuçları organize et
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({"text": texts[i], "index": i})

            # Cluster merkezlerini metin olarak tanımla
            cluster_descriptions = []
            for cluster_id in range(n_clusters):
                cluster_texts = [item["text"] for item in clusters.get(cluster_id, [])]
                if cluster_texts:
                    # En uzun metni cluster'ın temsilcisi olarak al
                    representative = max(cluster_texts, key=len)
                    cluster_descriptions.append(
                        {
                            "id": cluster_id,
                            "representative": representative,
                            "size": len(cluster_texts),
                        }
                    )

            logger.info(
                "semantic_clustering_completed",
                n_texts=len(texts),
                n_clusters=n_clusters,
                cluster_sizes=[desc["size"] for desc in cluster_descriptions],
            )

            return {
                "clusters": clusters,
                "labels": labels.tolist(),
                "cluster_descriptions": cluster_descriptions,
                "centroids": kmeans.cluster_centers_.tolist(),
            }

        except Exception as e:
            logger.error("semantic_clustering_error", error=str(e))
            return {"clusters": [], "labels": [], "centroids": []}

    def find_semantic_outliers(
        self, texts: List[str], threshold: float = 0.3
    ) -> List[Dict]:
        """Semantic olarak farklı metinleri bul"""
        try:
            if len(texts) < 3:
                return []

            # Similarity matrix hesapla
            similarity_matrix = self.batch_similarity_matrix(texts)

            outliers = []
            for i, text in enumerate(texts):
                # Bu metinin diğerleriyle ortalama benzerliği
                avg_similarity = np.mean(
                    [similarity_matrix[i][j] for j in range(len(texts)) if i != j]
                )

                if avg_similarity < threshold:
                    outliers.append(
                        {
                            "text": text,
                            "index": i,
                            "avg_similarity": float(avg_similarity),
                        }
                    )

            # En düşük benzerlik skoruna göre sırala
            outliers.sort(key=lambda x: x["avg_similarity"])

            logger.info(
                "semantic_outliers_found", count=len(outliers), threshold=threshold
            )

            return outliers

        except Exception as e:
            logger.error("semantic_outliers_error", error=str(e))
            return []

    async def semantic_search(
        self,
        query: str,
        question_pool: List[Dict],
        top_k: int = 10,
        similarity_threshold: float = 0.6,
    ) -> List[Dict]:
        """Gelişmiş semantic arama"""
        try:
            if not question_pool:
                return []

            # Query embedding'i hesapla
            query_embedding = await self.compute_embedding_cached(query)

            # Question embeddings'leri hesapla
            question_texts = [q.get("content", "") for q in question_pool]
            question_embeddings = self.compute_embeddings_batch(question_texts)

            # Similarity skorları hesapla
            query_vector = np.array(query_embedding).reshape(1, -1)
            results_with_scores = []

            for question, embedding in zip(question_pool, question_embeddings):
                if not embedding or sum(embedding) == 0:
                    continue

                question_vector = np.array(embedding).reshape(1, -1)
                similarity = cosine_similarity(query_vector, question_vector)[0][0]

                if similarity >= similarity_threshold:
                    result = question.copy()
                    result["semantic_similarity"] = float(similarity)
                    results_with_scores.append(result)

            # Similarity'ye göre sırala
            results_with_scores.sort(
                key=lambda x: x["semantic_similarity"], reverse=True
            )

            self.stats["similarities_computed"] += len(question_pool)

            logger.info(
                "semantic_search_completed",
                query_length=len(query),
                pool_size=len(question_pool),
                results_found=len(results_with_scores),
                top_similarity=results_with_scores[0]["semantic_similarity"]
                if results_with_scores
                else 0,
            )

            return results_with_scores[:top_k]

        except Exception as e:
            logger.error("semantic_search_error", error=str(e))
            return []

    def find_similar_questions_by_id(
        self, question_id: int, db_session, threshold: float = 0.8, limit: int = 10
    ) -> List[Dict]:
        """Helper method: Find similar questions by question ID (sync version)"""
        try:
            # Get the question content
            from app.db.models import Question

            question = (
                db_session.query(Question).filter(Question.id == question_id).first()
            )
            if not question:
                logger.warning("question_not_found", question_id=question_id)
                return []

            # Compute embedding for the question
            query_embedding = self.compute_embedding(str(question.content))
            if not query_embedding:
                logger.error("embedding_computation_failed", question_id=question_id)
                return []

            # For sync version, we'll use a simple similarity search
            # This is a fallback when async is not available
            return self._sync_find_similar_questions(
                query_embedding, threshold=threshold, limit=limit
            )

        except Exception as e:
            logger.error(
                "find_similar_questions_by_id_error",
                question_id=question_id,
                error=str(e),
            )
            return []

    async def find_similar_questions_by_id_async(
        self, question_id: int, db_session, threshold: float = 0.8, limit: int = 10
    ) -> List[Dict]:
        """Helper method: Find similar questions by question ID (async version)"""
        try:
            # Get the question content
            from app.db.models import Question

            question = (
                db_session.query(Question).filter(Question.id == question_id).first()
            )
            if not question:
                logger.warning("question_not_found", question_id=question_id)
                return []

            # Compute embedding for the question
            query_embedding = self.compute_embedding(str(question.content))
            if not query_embedding:
                logger.error("embedding_computation_failed", question_id=question_id)
                return []

            # Find similar questions using the embedding
            return await self.find_similar_questions(
                query_embedding, threshold=threshold, limit=limit
            )

        except Exception as e:
            logger.error(
                "find_similar_questions_by_id_error",
                question_id=question_id,
                error=str(e),
            )
            return []

    def _sync_find_similar_questions(
        self, query_embedding: List[float], threshold: float = 0.8, limit: int = 10
    ) -> List[Dict]:
        """Synchronous fallback for finding similar questions"""
        try:
            # Simple similarity search using database

            conn = self.get_postgres_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id, content, embedding
                FROM questions
                WHERE embedding IS NOT NULL AND is_active = true
                LIMIT 100
                """
            )

            results = []
            for row in cursor.fetchall():
                question_id, content, embedding_json = row

                if not embedding_json:
                    continue

                try:
                    stored_embedding = json.loads(embedding_json)
                    similarity = cosine_similarity(
                        np.array(query_embedding).reshape(1, -1),
                        np.array(stored_embedding).reshape(1, -1),
                    )[0][0]

                    if similarity >= threshold:
                        results.append(
                            {
                                "id": question_id,
                                "content": content,
                                "similarity": float(similarity),
                                "source": "sync_embedding",
                            }
                        )
                except (json.JSONDecodeError, ValueError):
                    continue

            # Sort by similarity and limit
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error("sync_similar_questions_error", error=str(e))
            return []

    def dimensionality_reduction(
        self, texts: List[str], target_dims: int = 2
    ) -> Tuple[np.ndarray, Optional[PCA]]:
        """Embedding'leri boyut azaltma ile görselleştirme için hazırla"""
        try:
            embeddings = self.compute_embeddings_batch(texts)
            if not embeddings:
                return np.array([]), None

            embedding_matrix = np.array(embeddings)

            # PCA ile boyut azaltma
            pca = PCA(n_components=target_dims, random_state=42)
            reduced_embeddings = pca.fit_transform(embedding_matrix)

            logger.info(
                "dimensionality_reduction_completed",
                original_dims=embedding_matrix.shape[1],
                target_dims=target_dims,
                explained_variance=float(pca.explained_variance_ratio_.sum()),
            )

            return reduced_embeddings, pca

        except Exception as e:
            logger.error("dimensionality_reduction_error", error=str(e))
            return np.array([]), None

    def get_enhanced_stats(self) -> Dict:
        """Gelişmiş istatistikler"""
        basic_stats = self.get_embedding_stats()

        # Cache statistics
        cache_total = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            (self.stats["cache_hits"] / cache_total * 100) if cache_total > 0 else 0
        )

        enhanced_stats = {
            **basic_stats,
            "model_info": {
                "current_model": self.current_model_key,
                "model_name": self.model_name,
                "embedding_dimensions": self.embedding_dim,
                "available_models": list(self.available_models.keys()),
            },
            "cache_stats": {
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "cache_hit_rate": round(cache_hit_rate, 2),
                "cache_ttl_seconds": self.cache_ttl,
            },
            "computation_stats": {
                "embeddings_computed": self.stats["embeddings_computed"],
                "similarities_computed": self.stats["similarities_computed"],
            },
        }

        return enhanced_stats

    async def initialize_vector_store(self):
        """Vector store'u başlat"""
        if not self.vector_store_initialized:
            try:
                await vector_store_service.initialize()
                self.vector_store_initialized = True
                logger.info("vector_store_initialized_in_enhanced_service")
            except Exception as e:
                logger.error("vector_store_init_error", error=str(e))
                raise

    async def store_question_embedding(
        self,
        question_id: int,
        question_text: str,
        metadata: Optional[Dict] = None,
        subject_id: Optional[int] = None,
        topic_id: Optional[int] = None,
        difficulty_level: Optional[int] = None,
    ) -> bool:
        """Soru embedding'ini hem cache hem vector store'a kaydet"""
        if not self.vector_store_initialized:
            await self.initialize_vector_store()

        try:
            # Embedding hesapla (cache'li)
            embedding = await self.compute_embedding_cached(question_text)

            # Vector store'a kaydet
            success = await vector_store_service.store_embedding(
                question_id=question_id,
                embedding=embedding,
                content=question_text,
                metadata=metadata,
                subject_id=subject_id,
                topic_id=topic_id,
                difficulty_level=difficulty_level,
            )

            logger.debug(
                "question_embedding_stored", question_id=question_id, success=success
            )

            return success

        except Exception as e:
            logger.error(
                "store_question_embedding_error", question_id=question_id, error=str(e)
            )
            return False

    async def semantic_search_vector_db(
        self,
        query_text: str,
        k: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Vector DB üzerinden semantic arama (O(log N) performance)"""
        if not self.vector_store_initialized:
            await self.initialize_vector_store()

        try:
            # Query embedding hesapla
            query_embedding = await self.compute_embedding_cached(query_text)

            # Vector DB'den ara
            results = await vector_store_service.similarity_search(
                query_embedding=query_embedding,
                k=k,
                similarity_threshold=similarity_threshold,
                filters=filters,
            )

            # Response formatına çevir
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "question_id": result.id,
                        "content": result.content,
                        "similarity_score": result.similarity_score,
                        "distance": result.distance,
                        "metadata": result.metadata,
                    }
                )

            logger.debug(
                "vector_semantic_search_completed",
                query_length=len(query_text),
                results_count=len(formatted_results),
                k=k,
                threshold=similarity_threshold,
            )

            return formatted_results

        except Exception as e:
            logger.error("vector_semantic_search_error", error=str(e))
            return []

    async def batch_update_embeddings(
        self, batch_size: int = 100, force_recompute: bool = False
    ) -> Dict[str, int]:
        """Veritabanındaki tüm soruları vector store'a batch update et"""
        if not self.vector_store_initialized:
            await self.initialize_vector_store()

        from app.db.database import SessionLocal
        from app.db.models import Question

        stats = {"processed": 0, "stored": 0, "errors": 0, "skipped": 0}

        try:
            db = SessionLocal()

            # Toplam soru sayısını al
            total_questions = db.query(Question).count()
            logger.info(
                "batch_update_starting",
                total_questions=total_questions,
                batch_size=batch_size,
            )

            # Batch işlemi
            offset = 0
            while offset < total_questions:
                batch_questions = (
                    db.query(Question).offset(offset).limit(batch_size).all()
                )

                if not batch_questions:
                    break

                # Batch embedding data hazırla
                embeddings_data = []

                for question in batch_questions:
                    stats["processed"] += 1

                    try:
                        # Force recompute yoksa mevcut embedding'i kontrol et
                        if not force_recompute:
                            existing = (
                                await vector_store_service.get_embedding_by_question_id(
                                    safe_int(question.id)
                                )
                            )
                            if existing:
                                stats["skipped"] += 1
                                continue

                        # Embedding hesapla
                        embedding = self.compute_embedding_cached(str(question.content))

                        embeddings_data.append(
                            {
                                "question_id": safe_int(question.id),
                                "embedding": embedding,
                                "content": str(question.content),
                                "metadata": {
                                    "created_at": question.created_at.isoformat()
                                    if question.created_at is not None
                                    else None
                                },
                                "subject_id": safe_int(question.subject_id)
                                if question.subject_id is not None
                                else None,
                                "topic_id": safe_int(question.topic_id)
                                if question.topic_id is not None
                                else None,
                                "difficulty_level": safe_int(question.difficulty_level)
                                if question.difficulty_level is not None
                                else 1,
                            }
                        )

                    except Exception as e:
                        logger.error(
                            "embedding_computation_error",
                            question_id=safe_int(question.id)
                            if question.id is not None
                            else None,
                            error=str(e),
                        )
                        stats["errors"] += 1

                # Batch store
                if embeddings_data:
                    stored_count = await vector_store_service.batch_store_embeddings(
                        embeddings_data
                    )
                    stats["stored"] += stored_count

                offset += batch_size

                # Progress log
                if offset % (batch_size * 10) == 0:
                    logger.info(
                        "batch_update_progress",
                        processed=stats["processed"],
                        stored=stats["stored"],
                        progress_pct=round((offset / total_questions) * 100, 2),
                    )

            db.close()

            logger.info("batch_update_completed", **stats)
            return stats

        except Exception as e:
            logger.error("batch_update_error", error=str(e))
            stats["errors"] += 1
            return stats

    async def get_vector_store_stats(self) -> Dict:
        """Vector store istatistikleri"""
        if not self.vector_store_initialized:
            await self.initialize_vector_store()

        try:
            return await vector_store_service.get_stats()
        except Exception as e:
            logger.error("get_vector_stats_error", error=str(e))
            return {}

    async def clear_cache(self) -> bool:
        """Embedding cache'ini temizle"""
        try:
            # Pattern ile embedding cache'ini temizle
            pattern = f"embedding:{self.current_model_key}:*"
            keys = await self.redis.keys(pattern)
            if keys:
                deleted_count = await self.redis.delete(*keys)
                logger.info("cache_cleared", deleted_keys=deleted_count)
            else:
                logger.info("cache_already_empty")

            # Stats'ı sıfırla
            self.stats["cache_hits"] = 0
            self.stats["cache_misses"] = 0

            return True

        except Exception as e:
            logger.error("cache_clear_error", error=str(e))
            return False


# Global instance with enhanced features
enhanced_embedding_service = EnhancedEmbeddingService()


# Enhanced convenience functions
async def compute_embedding_cached(text: str) -> List[float]:
    """Cache'li embedding hesaplama"""
    return await enhanced_embedding_service.compute_embedding_cached(text)


def semantic_similarity(text1: str, text2: str) -> float:
    """Semantic benzerlik"""
    return enhanced_embedding_service.semantic_similarity(text1, text2)


def semantic_clustering(texts: List[str], n_clusters: int = 5) -> Dict:
    """Semantic clustering"""
    return enhanced_embedding_service.semantic_clustering(texts, n_clusters)


async def semantic_search(
    query: str,
    question_pool: List[Dict],
    top_k: int = 10,
    similarity_threshold: float = 0.6,
) -> List[Dict]:
    """Gelişmiş semantic arama"""
    return await enhanced_embedding_service.semantic_search(
        query, question_pool, top_k, similarity_threshold
    )


def find_semantic_outliers(texts: List[str], threshold: float = 0.3) -> List[Dict]:
    """Semantic outlier'ları bul"""
    return enhanced_embedding_service.find_semantic_outliers(texts, threshold)
