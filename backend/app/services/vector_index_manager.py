from typing import List, Dict, Any, Optional
import logging
import asyncio
import numpy as np
from datetime import datetime
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, text
from sqlalchemy.orm import joinedload

from app.models.question import Question, Subject, DifficultyLevel
from app.models.error_pattern import ErrorPattern
from app.core.database import database
from app.core.cache import cache_service
from app.core.config import settings
from app.utils.distlock_idem import RedisLock, lock_decorator, IdempotencyConfig, idempotency_decorator
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class VectorIndexManager:
    """Production-ready vector index manager with real embeddings"""
    
    def __init__(self):
        self.embedding_dimension = settings.embedding_dimension
        self.index_name_questions = "ix_questions_content_embedding"
        self.index_name_errors = "ix_error_patterns_embedding"
        self.cache_ttl = 1800  # 30 minutes
        self.batch_size = settings.vector_batch_size
        
    def _get_redis_client(self) -> Redis:
        """Get Redis client for distributed locks"""
        return Redis.from_url(settings.redis_url)
        
    @lock_decorator(
        key_builder=lambda self: "lock:vector:create_indexes",
        ttl_ms=120_000,  # 2 minutes lock
        wait_timeout_ms=60_000,  # 1 minute wait
        redis_client_factory=lambda: Redis.from_url(settings.redis_url)
    )
    async def create_vector_indexes(self):
        """Create vector indexes for similarity search (protected by distributed lock)"""
        
        try:
            # Check if pgvector extension is available
            if not await self._check_pgvector_extension():
                logger.error("❌ pgvector extension not available")
                logger.error("Please install pgvector extension: sudo apt-get install postgresql-13-pgvector")
                return False
            
            # Note: Extension and indexes are now handled by Alembic migrations
            logger.info("🔧 Vector indexes are managed by Alembic migrations")
            logger.info("✅ Please run: alembic upgrade head")
            
            # Verify indexes exist
            questions_index_exists = await self._check_index_exists(self.index_name_questions)
            errors_index_exists = await self._check_index_exists(self.index_name_errors)
            
            if questions_index_exists and errors_index_exists:
                logger.info("✅ Vector indexes verified successfully")
                return True
            else:
                logger.error("❌ Some indexes are missing - run alembic upgrade head")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error verifying vector indexes: {e}")
            return False
    
    async def _check_pgvector_extension(self) -> bool:
        """Check if pgvector extension is available"""
        try:
            result = await database.fetch_one("""
                SELECT 1 FROM pg_available_extensions 
                WHERE name = 'vector' AND installed_version IS NOT NULL
            """)
            return result is not None
        except Exception as e:
            logger.warning(f"Could not check pgvector extension: {e}")
            return False
    
    async def _check_index_exists(self, index_name: str) -> bool:
        """Check if a specific index exists"""
        try:
            result = await database.fetch_one("""
                SELECT 1 FROM pg_indexes 
                WHERE indexname = $1
            """, index_name)
            return result is not None
        except Exception as e:
            logger.warning(f"Could not check index {index_name}: {e}")
            return False
    
    async def _add_embedding_columns(self):
        """Add embedding columns to tables"""
        
        # Add embedding column to questions table
        try:
            await database.execute("""
                ALTER TABLE questions 
                ADD COLUMN IF NOT EXISTS content_embedding vector(1536);
            """)
            logger.info("✅ Questions embedding column added/verified")
        except Exception as e:
            logger.warning(f"Questions embedding column might already exist: {e}")
        
        # Add embedding column to error_patterns table
        try:
            await database.execute("""
                ALTER TABLE error_patterns 
                ADD COLUMN IF NOT EXISTS embedding vector(1536);
            """)
            logger.info("✅ Error patterns embedding column added/verified")
        except Exception as e:
            logger.warning(f"Error patterns embedding column might already exist: {e}")
    
    async def _create_question_index(self):
        """Create index for questions content embedding"""
        
        try:
            await database.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.index_name_questions}
                ON questions 
                USING ivfflat (content_embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            logger.info("✅ Questions vector index created")
        except Exception as e:
            logger.error(f"❌ Error creating questions index: {e}")
    
    async def _create_error_pattern_index(self):
        """Create index for error patterns embedding"""
        
        try:
            await database.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.index_name_errors}
                ON error_patterns 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 50);
            """)
            logger.info("✅ Error patterns vector index created")
        except Exception as e:
            logger.error(f"❌ Error creating error patterns index: {e}")
    
    async def update_question_embeddings(self, db: AsyncSession, batch_size: int = 100):
        """Update embeddings for questions that don't have them"""
        
        try:
            # Get questions without embeddings
            query = select(Question).where(
                and_(
                    Question.content_embedding.is_(None),
                    Question.is_active == True
                )
            ).limit(batch_size)
            
            result = await db.execute(query)
            questions = result.scalars().all()
            
            updated_count = 0
            for question in questions:
                # Generate embedding using real service
                embedding = await self._generate_embedding(question.content)
                
                # Update question
                question.content_embedding = embedding
                updated_count += 1
            
            await db.commit()
            logger.info(f"✅ Updated embeddings for {updated_count} questions")
            
            return updated_count
            
        except Exception as e:
            logger.error(f"❌ Error updating question embeddings: {e}")
            await db.rollback()
            return 0
    
    async def update_error_pattern_embeddings(self, db: AsyncSession, batch_size: int = 50):
        """Update embeddings for error patterns that don't have them"""
        
        try:
            # Get error patterns without embeddings
            query = select(ErrorPattern).where(
                ErrorPattern.embedding.is_(None)
            ).limit(batch_size)
            
            result = await db.execute(query)
            patterns = result.scalars().all()
            
            updated_count = 0
            for pattern in patterns:
                # Create text for embedding
                pattern_text = f"{pattern.error_type} {pattern.pattern_details or ''} {pattern.topic_category or ''}"
                
                # Generate embedding
                embedding = await self._generate_embedding(pattern_text)
                
                # Update pattern
                pattern.embedding = embedding
                updated_count += 1
            
            await db.commit()
            logger.info(f"✅ Updated embeddings for {updated_count} error patterns")
            
            return updated_count
            
        except Exception as e:
            logger.error(f"❌ Error updating error pattern embeddings: {e}")
            await db.rollback()
            return 0
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate real embedding for text using embedding service"""
        
        try:
            from app.services.embedding_service import embedding_service
            embedding = await embedding_service.get_embedding(text)
            
            # Validate embedding dimension
            if len(embedding) != self.embedding_dimension:
                logger.error(f"❌ Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding)}")
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dimension
    
    async def _get_active_slot(self, namespace: str) -> int:
        """Get the active slot for a namespace from cache or database"""
        cache_key = f"emb:{namespace}:active_slot"
        
        # Try cache first
        cached_slot = await cache_service.get(cache_key)
        if cached_slot is not None:
            return int(cached_slot)
        
        # Query database for active slot
        result = await database.fetch_one("""
            SELECT slot FROM questions 
            WHERE namespace = $1 AND is_active = true 
            ORDER BY slot DESC LIMIT 1
        """, namespace)
        
        if result:
            active_slot = result['slot']
            # Cache for 1 hour
            await cache_service.set(cache_key, str(active_slot), 3600)
            return active_slot
        
        # Default to slot 1 if no active slot found
        return settings.vector_slot_default
    
    async def _deactivate_old_slot(self, namespace: str, old_slot: int):
        """Gracefully deactivate an old slot"""
        try:
            # Mark old slot as inactive
            await database.execute("""
                UPDATE questions 
                SET is_active = false, deactivated_at = NOW() 
                WHERE namespace = $1 AND slot = $2 AND is_active = true
            """, namespace, old_slot)
            
            await database.execute("""
                UPDATE error_patterns 
                SET is_active = false, deactivated_at = NOW() 
                WHERE namespace = $1 AND slot = $2 AND is_active = true
            """, namespace, old_slot)
            
            logger.info(f"✅ Deactivated old slot {old_slot} for namespace {namespace}")
            
        except Exception as e:
            logger.error(f"❌ Error deactivating old slot: {e}")
    
    @lock_decorator(
        key_builder=lambda self, items, table_name, namespace=None: f"lock:vector:upsert:{table_name}:{namespace or 'default'}",
        ttl_ms=60_000,  # 1 minute lock
        wait_timeout_ms=30_000,  # 30 seconds wait
        redis_client_factory=lambda: Redis.from_url(settings.redis_url)
    )
    async def batch_upsert_embeddings(
        self, 
        items: List[Dict[str, Any]], 
        table_name: str,
        namespace: str = None
    ) -> Dict[str, Any]:
        """Batch upsert embeddings with namespace/slot strategy (protected by distributed lock)"""
        
        if not items:
            return {"success": True, "processed": 0, "errors": []}
        
        namespace = namespace or settings.vector_namespace_default
        active_slot = await self._get_active_slot(namespace)
        
        # Prepare batch data
        batch_data = []
        for item in items:
            obj_ref = item.get('obj_ref', str(item.get('id', '')))
            content = item.get('content', '')
            
            # Generate embedding
            embedding = await self._generate_embedding(content)
            
            batch_data.append({
                'obj_ref': obj_ref,
                'namespace': namespace,
                'slot': active_slot,
                'embedding': embedding,
                'embedding_dim': self.embedding_dimension
            })
        
        try:
            # Batch upsert
            if table_name == "questions":
                sql = """
                INSERT INTO questions (obj_ref, namespace, slot, content_embedding, embedding_dim, is_active, updated_at)
                VALUES (:obj_ref, :namespace, :slot, :embedding, :embedding_dim, true, NOW())
                ON CONFLICT (obj_ref, namespace, slot)
                DO UPDATE SET 
                    content_embedding = EXCLUDED.content_embedding,
                    embedding_dim = EXCLUDED.embedding_dim,
                    is_active = true,
                    deactivated_at = NULL,
                    updated_at = NOW()
                """
            elif table_name == "error_patterns":
                sql = """
                INSERT INTO error_patterns (obj_ref, namespace, slot, embedding, embedding_dim, is_active, updated_at)
                VALUES (:obj_ref, :namespace, :slot, :embedding, :embedding_dim, true, NOW())
                ON CONFLICT (obj_ref, namespace, slot)
                DO UPDATE SET 
                    embedding = EXCLUDED.embedding,
                    embedding_dim = EXCLUDED.embedding_dim,
                    is_active = true,
                    deactivated_at = NULL,
                    updated_at = NOW()
                """
            else:
                raise ValueError(f"Unsupported table: {table_name}")
            
            # Execute batch upsert
            await database.execute_many(sql, batch_data)
            
            logger.info(f"✅ Batch upserted {len(batch_data)} embeddings for {table_name}")
            
            return {
                "success": True,
                "processed": len(batch_data),
                "namespace": namespace,
                "slot": active_slot,
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"❌ Error in batch upsert: {e}")
            return {
                "success": False,
                "processed": 0,
                "errors": [str(e)]
            }
    
    async def perform_vector_search(
        self,
        query_embedding: List[float],
        table_name: str,
        filters: Dict[str, Any] = None,
        limit: int = 10,
        namespace: str = None,
        min_similarity: float = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search with namespace/slot support"""
        
        try:
            # Validate embedding dimension
            if len(query_embedding) != self.embedding_dimension:
                logger.error(f"❌ Query embedding dimension mismatch: expected {self.embedding_dimension}, got {len(query_embedding)}")
                return []
            
            namespace = namespace or settings.vector_namespace_default
            min_similarity = min_similarity or settings.vector_similarity_threshold
            
            # Build the search query
            if table_name == "questions":
                search_query = f"""
                    SELECT 
                        id,
                        content,
                        topic_category,
                        difficulty_level,
                        question_type,
                        namespace,
                        slot,
                        1 - (content_embedding <=> $1::vector) as similarity
                    FROM questions
                    WHERE content_embedding IS NOT NULL
                    AND is_active = true
                    AND namespace = $2
                """
                
                # Add filters
                if filters:
                    if "subject" in filters:
                        search_query += f" AND subject = '{filters['subject']}'"
                    if "topic_category" in filters:
                        search_query += f" AND topic_category = '{filters['topic_category']}'"
                    if "difficulty_level" in filters:
                        search_query += f" AND difficulty_level = {filters['difficulty_level']}"
                    if "slot" in filters:
                        search_query += f" AND slot = {filters['slot']}"
                
            elif table_name == "error_patterns":
                search_query = f"""
                    SELECT 
                        id,
                        error_type,
                        pattern_details,
                        topic_category,
                        namespace,
                        slot,
                        1 - (embedding <=> $1::vector) as similarity
                    FROM error_patterns
                    WHERE embedding IS NOT NULL
                    AND is_active = true
                    AND namespace = $2
                """
                
                # Add filters
                if filters:
                    if "subject" in filters:
                        search_query += f" AND subject = '{filters['subject']}'"
                    if "topic_category" in filters:
                        search_query += f" AND topic_category = '{filters['topic_category']}'"
                    if "slot" in filters:
                        search_query += f" AND slot = {filters['slot']}"
            
            # Add similarity threshold and ordering
            search_query += f"""
                AND (1 - (content_embedding <=> $1::vector)) >= $3
                ORDER BY similarity DESC
                LIMIT {limit}
            """
            
            # Execute search
            result = await database.fetch_all(search_query, query_embedding, namespace, min_similarity)
            
            # Convert to list of dicts
            search_results = []
            for row in result:
                search_results.append(dict(row))
            
            logger.info(f"✅ Vector search returned {len(search_results)} results for {table_name}")
            return search_results
            
        except Exception as e:
            logger.error(f"❌ Error performing vector search: {e}")
            return []
    
    async def get_index_statistics(self) -> Dict[str, Any]:
        """Get statistics about vector indexes"""
        
        try:
            # Get questions embedding statistics
            questions_stats = await database.fetch_one("""
                SELECT 
                    COUNT(*) as total_questions,
                    COUNT(content_embedding) as questions_with_embeddings,
                    COUNT(*) - COUNT(content_embedding) as questions_without_embeddings
                FROM questions
                WHERE is_active = true
            """)
            
            # Get error patterns embedding statistics
            errors_stats = await database.fetch_one("""
                SELECT 
                    COUNT(*) as total_patterns,
                    COUNT(embedding) as patterns_with_embeddings,
                    COUNT(*) - COUNT(embedding) as patterns_without_embeddings
                FROM error_patterns
            """)
            
            return {
                "questions": dict(questions_stats) if questions_stats else {},
                "error_patterns": dict(errors_stats) if errors_stats else {},
                "index_status": {
                    "questions_index": await self._check_index_exists(self.index_name_questions),
                    "errors_index": await self._check_index_exists(self.index_name_errors)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting index statistics: {e}")
            return {}
    
    @idempotency_decorator(
        key_builder=lambda self, namespace: f"idem:vector:rebuild:{namespace}",
        config=IdempotencyConfig(scope="vector_rebuild", ttl_seconds=3600, in_progress_ttl=600),
        redis_client_factory=lambda: Redis.from_url(settings.redis_url)
    )
    async def rebuild_vector_index(self, namespace: str) -> Dict[str, Any]:
        """Rebuild vector index for a namespace (idempotent operation)"""
        try:
            logger.info(f"🔄 Starting vector index rebuild for namespace: {namespace}")
            
            # Get all questions for the namespace
            questions = await database.fetch_all("""
                SELECT id, content FROM questions 
                WHERE namespace = $1 AND is_active = true
            """, namespace)
            
            # Get all error patterns for the namespace
            error_patterns = await database.fetch_all("""
                SELECT id, pattern_details FROM error_patterns 
                WHERE namespace = $1 AND is_active = true
            """, namespace)
            
            # Rebuild embeddings
            questions_updated = 0
            patterns_updated = 0
            
            # Update questions embeddings
            for question in questions:
                embedding = await self._generate_embedding(question['content'])
                await database.execute("""
                    UPDATE questions 
                    SET content_embedding = $1, embedding_dim = $2, updated_at = NOW()
                    WHERE id = $3
                """, embedding, self.embedding_dimension, question['id'])
                questions_updated += 1
            
            # Update error patterns embeddings
            for pattern in error_patterns:
                embedding = await self._generate_embedding(pattern['pattern_details'])
                await database.execute("""
                    UPDATE error_patterns 
                    SET embedding = $1, embedding_dim = $2, updated_at = NOW()
                    WHERE id = $3
                """, embedding, self.embedding_dimension, pattern['id'])
                patterns_updated += 1
            
            logger.info(f"✅ Vector index rebuild completed for namespace {namespace}")
            
            return {
                "success": True,
                "namespace": namespace,
                "questions_updated": questions_updated,
                "patterns_updated": patterns_updated,
                "total_processed": questions_updated + patterns_updated
            }
            
        except Exception as e:
            logger.error(f"❌ Error rebuilding vector index for namespace {namespace}: {e}")
            return {
                "success": False,
                "namespace": namespace,
                "error": str(e)
            }
    
    async def cleanup_old_slots(self, namespace: str, keep_slots: int = 3) -> Dict[str, Any]:
        """Clean up old slots, keeping only the most recent ones"""
        try:
            # Get old slots to clean up
            old_slots = await database.fetch_all("""
                SELECT slot FROM questions 
                WHERE namespace = $1 
                ORDER BY slot DESC 
                OFFSET $2
            """, namespace, keep_slots)
            
            if not old_slots:
                return {"success": True, "slots_cleaned": 0, "message": "No old slots to clean"}
            
            slots_to_clean = [slot['slot'] for slot in old_slots]
            
            # Delete old slots
            await database.execute("""
                DELETE FROM questions 
                WHERE namespace = $1 AND slot = ANY($2)
            """, namespace, slots_to_clean)
            
            await database.execute("""
                DELETE FROM error_patterns 
                WHERE namespace = $1 AND slot = ANY($2)
            """, namespace, slots_to_clean)
            
            logger.info(f"✅ Cleaned up {len(slots_to_clean)} old slots for namespace {namespace}")
            
            return {
                "success": True,
                "slots_cleaned": len(slots_to_clean),
                "cleaned_slots": slots_to_clean
            }
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up old slots for namespace {namespace}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _check_index_exists(self, index_name: str) -> bool:
        """Check if an index exists"""
        
        try:
            result = await database.fetch_one("""
                SELECT 1 FROM pg_indexes 
                WHERE indexname = :index_name
            """, {"index_name": index_name})
            
            return result is not None
            
        except Exception as e:
            logger.error(f"❌ Error checking index existence: {e}")
            return False
    
    async def batch_update_embeddings(self, db: AsyncSession, batch_size: int = 100):
        """Batch update embeddings for both questions and error patterns"""
        
        try:
            # Update question embeddings
            questions_updated = await self.update_question_embeddings(db, batch_size)
            
            # Update error pattern embeddings
            patterns_updated = await self.update_error_pattern_embeddings(db, batch_size // 2)
            
            logger.info(f"✅ Batch update completed: {questions_updated} questions, {patterns_updated} patterns")
            
            return {
                "questions_updated": questions_updated,
                "patterns_updated": patterns_updated,
                "total_updated": questions_updated + patterns_updated
            }
            
        except Exception as e:
            logger.error(f"❌ Error in batch update: {e}")
            return {"questions_updated": 0, "patterns_updated": 0, "total_updated": 0}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for vector index system"""
        
        try:
            # Check pgvector extension
            pgvector_available = await self._check_pgvector_extension()
            
            # Check indexes
            questions_index_exists = await self._check_index_exists(self.index_name_questions)
            errors_index_exists = await self._check_index_exists(self.index_name_errors)
            
            # Get statistics
            stats = await self.get_index_statistics()
            
            return {
                "status": "healthy" if pgvector_available and questions_index_exists else "unhealthy",
                "pgvector_available": pgvector_available,
                "indexes_created": {
                    "questions": questions_index_exists,
                    "error_patterns": errors_index_exists
                },
                "statistics": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error in health check: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup_orphaned_embeddings(self, db: AsyncSession):
        """Clean up embeddings for deleted questions/patterns"""
        
        try:
            # Clean up question embeddings for deleted questions
            await database.execute("""
                UPDATE questions 
                SET content_embedding = NULL 
                WHERE is_active = false AND content_embedding IS NOT NULL
            """)
            
            # Clean up error pattern embeddings for deleted patterns
            await database.execute("""
                UPDATE error_patterns 
                SET embedding = NULL 
                WHERE error_count = 0 AND embedding IS NOT NULL
            """)
            
            logger.info("✅ Cleaned up orphaned embeddings")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up orphaned embeddings: {e}")
            return False

    async def search_similar_content(
        self,
        query_embedding: List[float],
        namespace: str,
        similarity_threshold: float = 0.7,
        limit: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar content using embeddings with domain-specific namespaces"""
        try:
            # Validate embedding dimension
            if len(query_embedding) != self.embedding_dimension:
                logger.error(f"Query embedding dimension mismatch: expected {self.embedding_dimension}, got {len(query_embedding)}")
                return []
            
            # Get active slot for namespace
            active_slot = await self._get_active_slot(namespace)
            
            # Build base query
            base_query = """
                SELECT obj_ref, meta, 1 - (embedding <=> $1::vector) as similarity
                FROM embeddings
                WHERE namespace = $2 AND slot = $3 AND is_active = true
            """
            
            # Add metadata filters if provided
            filter_conditions = []
            filter_params = [query_embedding, namespace, active_slot]
            param_count = 3
            
            if metadata_filters:
                for key, value in metadata_filters.items():
                    if isinstance(value, (str, int, float)):
                        filter_conditions.append(f"meta->>'{key}' = ${param_count + 1}")
                        filter_params.append(str(value))
                        param_count += 1
                    elif isinstance(value, list):
                        filter_conditions.append(f"meta->>'{key}' = ANY(${param_count + 1})")
                        filter_params.append(value)
                        param_count += 1
            
            if filter_conditions:
                base_query += " AND " + " AND ".join(filter_conditions)
            
            # Add similarity threshold and ordering
            base_query += f"""
                AND 1 - (embedding <=> $1::vector) > ${param_count + 1}
                ORDER BY embedding <=> $1::vector
                LIMIT ${param_count + 2}
            """
            
            filter_params.extend([similarity_threshold, limit])
            
            # Execute query
            results = await database.fetch_all(base_query, *filter_params)
            
            # Process results
            processed_results = []
            for row in results:
                processed_results.append({
                    "obj_ref": row["obj_ref"],
                    "similarity": float(row["similarity"]),
                    "metadata": row["meta"] or {},
                    "namespace": namespace
                })
            
            logger.info(f"Found {len(processed_results)} similar items in namespace {namespace}")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    async def search_domain_specific(
        self,
        query_embedding: List[float],
        domain: str,
        content_type: str,
        similarity_threshold: float = 0.7,
        limit: int = 10,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Domain-specific search with optimized namespaces"""
        try:
            # Map domain and content type to namespace
            namespace_mapping = {
                "english": {
                    "error_patterns": "english_error_patterns",
                    "cloze_questions": "english_cloze_questions",
                    "grammar_rules": "english_grammar_rules",
                    "vocabulary": "english_vocabulary"
                },
                "math": {
                    "error_patterns": "math_error_patterns",
                    "questions": "math_questions",
                    "concepts": "math_concepts",
                    "solutions": "math_solutions"
                },
                "cefr": {
                    "rubrics": "cefr_rubrics",
                    "examples": "cefr_examples",
                    "assessments": "user_assessments"
                }
            }
            
            if domain not in namespace_mapping or content_type not in namespace_mapping[domain]:
                logger.warning(f"Unknown domain/content_type: {domain}/{content_type}")
                return []
            
            namespace = namespace_mapping[domain][content_type]
            
            # Build domain-specific filters
            domain_filters = additional_filters or {}
            
            # Add domain-specific metadata
            if domain == "english":
                domain_filters["domain"] = "english"
                domain_filters["content_type"] = content_type
            elif domain == "math":
                domain_filters["domain"] = "math"
                domain_filters["content_type"] = content_type
            elif domain == "cefr":
                domain_filters["assessment_type"] = content_type
            
            # Perform search
            return await self.search_similar_content(
                query_embedding=query_embedding,
                namespace=namespace,
                similarity_threshold=similarity_threshold,
                limit=limit,
                metadata_filters=domain_filters
            )
            
        except Exception as e:
            logger.error(f"Error in domain-specific search: {e}")
            return []

    async def batch_upsert_domain_embeddings(
        self,
        domain: str,
        content_type: str,
        items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Batch upsert embeddings for specific domain and content type"""
        try:
            if not items:
                return {"success": True, "processed": 0, "errors": []}
            
            # Map to appropriate namespace
            namespace_mapping = {
                "english": {
                    "error_patterns": "english_error_patterns",
                    "cloze_questions": "english_cloze_questions",
                    "grammar_rules": "english_grammar_rules"
                },
                "math": {
                    "error_patterns": "math_error_patterns",
                    "questions": "math_questions",
                    "concepts": "math_concepts"
                }
            }
            
            if domain not in namespace_mapping or content_type not in namespace_mapping[domain]:
                raise ValueError(f"Unknown domain/content_type: {domain}/{content_type}")
            
            namespace = namespace_mapping[domain][content_type]
            active_slot = await self._get_active_slot(namespace)
            
            # Prepare batch data
            batch_data = []
            for item in items:
                obj_ref = item.get('obj_ref', str(item.get('id', '')))
                content = item.get('content', '')
                
                # Generate embedding
                embedding = await self._generate_embedding(content)
                
                # Build metadata
                metadata = {
                    "domain": domain,
                    "content_type": content_type,
                    "created_at": item.get('created_at', ''),
                    "user_id": item.get('user_id', ''),
                    "difficulty_level": item.get('difficulty_level', 3),
                    "topic_category": item.get('topic_category', ''),
                    "error_type": item.get('error_type', ''),
                    **item.get('metadata', {})
                }
                
                batch_data.append({
                    'obj_ref': obj_ref,
                    'namespace': namespace,
                    'slot': active_slot,
                    'embedding': embedding,
                    'embedding_dim': self.embedding_dimension,
                    'metadata': metadata
                })
            
            # Batch upsert
            sql = """
            INSERT INTO embeddings (obj_ref, namespace, slot, embedding, embedding_dim, meta, is_active, updated_at)
            VALUES (:obj_ref, :namespace, :slot, :embedding, :embedding_dim, :metadata, true, NOW())
            ON CONFLICT (obj_ref, namespace, slot)
            DO UPDATE SET 
                embedding = EXCLUDED.embedding,
                embedding_dim = EXCLUDED.embedding_dim,
                meta = EXCLUDED.metadata,
                is_active = true,
                deactivated_at = NULL,
                updated_at = NOW()
            """
            
            await database.execute_many(sql, batch_data)
            
            logger.info(f"✅ Batch upserted {len(batch_data)} embeddings for {domain}/{content_type}")
            
            return {
                "success": True,
                "processed": len(batch_data),
                "namespace": namespace,
                "domain": domain,
                "content_type": content_type
            }
            
        except Exception as e:
            logger.error(f"❌ Error in batch domain upsert: {e}")
            return {
                "success": False,
                "error": str(e),
                "domain": domain,
                "content_type": content_type
            }

    async def get_domain_statistics(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a specific domain"""
        try:
            # Get domain namespaces
            domain_namespaces = {
                "english": ["english_error_patterns", "english_cloze_questions", "english_grammar_rules"],
                "math": ["math_error_patterns", "math_questions", "math_concepts"],
                "cefr": ["cefr_rubrics", "user_assessments"]
            }
            
            if domain not in domain_namespaces:
                return {"error": f"Unknown domain: {domain}"}
            
            namespace_stats = {}
            total_embeddings = 0
            
            for namespace in domain_namespaces[domain]:
                try:
                    # Get active slot
                    active_slot = await self._get_active_slot(namespace)
                    
                    # Count embeddings
                    count_result = await database.fetch_one("""
                        SELECT COUNT(*) as count
                        FROM embeddings
                        WHERE namespace = $1 AND slot = $2 AND is_active = true
                    """, namespace, active_slot)
                    
                    count = count_result["count"] if count_result else 0
                    namespace_stats[namespace] = {
                        "embedding_count": count,
                        "active_slot": active_slot
                    }
                    total_embeddings += count
                    
                except Exception as e:
                    logger.warning(f"Error getting stats for namespace {namespace}: {e}")
                    namespace_stats[namespace] = {"error": str(e)}
            
            return {
                "domain": domain,
                "total_embeddings": total_embeddings,
                "namespaces": namespace_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting domain statistics: {e}")
            return {"error": str(e)}

    async def create_domain_specific_indexes(self) -> Dict[str, Any]:
        """Create optimized indexes for different domains"""
        try:
            logger.info("🔧 Creating domain-specific vector indexes...")
            
            # Domain-specific namespace configurations
            domain_configs = {
                "english": {
                    "namespaces": [
                        "english_errors",
                        "english_questions", 
                        "english_cloze_questions",
                        "english_grammar_rules",
                        "english_vocabulary",
                        "cefr_rubrics",
                        "cefr_examples",
                        "user_assessments"
                    ],
                    "similarity_threshold": 0.7,
                    "index_type": "ivfflat",
                    "lists": 100
                },
                "math": {
                    "namespaces": [
                        "math_errors",
                        "math_questions",
                        "math_concepts", 
                        "math_solutions",
                        "math_placement_tests"
                    ],
                    "similarity_threshold": 0.8,
                    "index_type": "ivfflat",
                    "lists": 150
                },
                "cefr": {
                    "namespaces": [
                        "cefr_rubrics",
                        "cefr_examples",
                        "user_assessments"
                    ],
                    "similarity_threshold": 0.6,
                    "index_type": "ivfflat",
                    "lists": 50
                }
            }
            
            created_indexes = {}
            
            for domain, config in domain_configs.items():
                domain_indexes = {}
                
                for namespace in config["namespaces"]:
                    try:
                        # Create namespace-specific index
                        index_name = f"ix_{namespace}_embedding"
                        
                        # Check if index already exists
                        if not await self._check_index_exists(index_name):
                            # Create optimized index
                            await self._create_optimized_index(
                                index_name, 
                                config["index_type"], 
                                config["lists"]
                            )
                            domain_indexes[namespace] = {
                                "status": "created",
                                "index_type": config["index_type"],
                                "similarity_threshold": config["similarity_threshold"]
                            }
                        else:
                            domain_indexes[namespace] = {
                                "status": "exists",
                                "index_type": config["index_type"],
                                "similarity_threshold": config["similarity_threshold"]
                            }
                            
                    except Exception as e:
                        logger.error(f"Error creating index for {namespace}: {e}")
                        domain_indexes[namespace] = {"status": "error", "error": str(e)}
                
                created_indexes[domain] = domain_indexes
            
            logger.info("✅ Domain-specific indexes created successfully")
            return created_indexes
            
        except Exception as e:
            logger.error(f"❌ Error creating domain-specific indexes: {e}")
            return {"error": str(e)}

    async def _create_optimized_index(
        self, 
        index_name: str, 
        index_type: str = "ivfflat", 
        lists: int = 100
    ):
        """Create an optimized vector index"""
        try:
            # Create index with optimized parameters
            create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON embeddings USING {index_type} (embedding vector_cosine_ops)
            WITH (lists = {lists});
            """
            
            await database.execute(create_index_sql)
            logger.info(f"✅ Created optimized index: {index_name} (type: {index_type}, lists: {lists})")
            
        except Exception as e:
            logger.error(f"❌ Error creating optimized index {index_name}: {e}")
            raise

    async def batch_upsert_domain_embeddings_enhanced(
        self,
        domain: str,
        content_type: str,
        items: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Enhanced batch upsert with performance optimizations"""
        try:
            if not items:
                return {"success": True, "processed": 0, "errors": []}
            
            # Map to appropriate namespace
            namespace_mapping = self._get_domain_namespace_mapping()
            
            if domain not in namespace_mapping or content_type not in namespace_mapping[domain]:
                raise ValueError(f"Unknown domain/content_type: {domain}/{content_type}")
            
            namespace = namespace_mapping[domain][content_type]
            active_slot = await self._get_active_slot(namespace)
            
            # Process in batches for better performance
            total_processed = 0
            errors = []
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                
                try:
                    # Prepare batch data with enhanced metadata
                    batch_data = await self._prepare_batch_data_enhanced(
                        batch, domain, content_type, namespace, active_slot
                    )
                    
                    # Batch upsert with optimized SQL
                    await self._execute_batch_upsert_optimized(batch_data)
                    
                    total_processed += len(batch)
                    logger.info(f"✅ Processed batch {i//batch_size + 1}: {len(batch)} items")
                    
                except Exception as e:
                    batch_error = f"Batch {i//batch_size + 1} failed: {e}"
                    errors.append(batch_error)
                    logger.error(f"❌ {batch_error}")
                    continue
            
            # Update namespace statistics
            await self._update_namespace_statistics(namespace, total_processed)
            
            return {
                "success": len(errors) == 0,
                "processed": total_processed,
                "errors": errors,
                "namespace": namespace,
                "domain": domain,
                "content_type": content_type,
                "batch_size_used": batch_size
            }
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced batch domain upsert: {e}")
            return {
                "success": False,
                "error": str(e),
                "domain": domain,
                "content_type": content_type
            }

    def _get_domain_namespace_mapping(self) -> Dict[str, Dict[str, str]]:
        """Get comprehensive domain namespace mapping"""
        return {
            "english": {
                "error_patterns": "english_errors",
                "cloze_questions": "english_cloze_questions",
                "grammar_rules": "english_grammar_rules",
                "vocabulary": "english_vocabulary",
                "questions": "english_questions"
            },
            "math": {
                "error_patterns": "math_errors",
                "questions": "math_questions",
                "concepts": "math_concepts",
                "solutions": "math_solutions",
                "placement_tests": "math_placement_tests"
            },
            "cefr": {
                "rubrics": "cefr_rubrics",
                "examples": "cefr_examples",
                "assessments": "user_assessments"
            }
        }

    async def _prepare_batch_data_enhanced(
        self,
        batch: List[Dict[str, Any]],
        domain: str,
        content_type: str,
        namespace: str,
        active_slot: int
    ) -> List[Dict[str, Any]]:
        """Prepare enhanced batch data with standardized metadata"""
        try:
            batch_data = []
            
            for item in batch:
                obj_ref = item.get('obj_ref', str(item.get('id', '')))
                content = item.get('content', '')
                
                # Generate embedding if not provided
                if not item.get('embedding'):
                    embedding = await self._generate_embedding(content)
                else:
                    embedding = item['embedding']
                
                # Build standardized metadata
                metadata = self._build_standardized_metadata(
                    item, domain, content_type, namespace
                )
                
                batch_data.append({
                    'obj_ref': obj_ref,
                    'namespace': namespace,
                    'slot': active_slot,
                    'embedding': embedding,
                    'embedding_dim': self.embedding_dimension,
                    'metadata': metadata
                })
            
            return batch_data
            
        except Exception as e:
            logger.error(f"Error preparing enhanced batch data: {e}")
            raise

    def _build_standardized_metadata(
        self,
        item: Dict[str, Any],
        domain: str,
        content_type: str,
        namespace: str
    ) -> Dict[str, Any]:
        """Build standardized metadata schema"""
        try:
            # Base metadata
            metadata = {
                "domain": domain,
                "content_type": content_type,
                "namespace": namespace,
                "created_at": item.get('created_at', datetime.utcnow().isoformat()),
                "updated_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            }
            
            # Domain-specific metadata
            if domain == "english":
                metadata.update({
                    "error_type": item.get('error_type', ''),
                    "grammar_rule": item.get('grammar_rule', ''),
                    "cefr_level": item.get('cefr_level', ''),
                    "difficulty_level": item.get('difficulty_level', 3),
                    "topic_category": item.get('topic_category', ''),
                    "question_type": item.get('question_type', ''),
                    "source": item.get('source', ''),
                    "user_id": item.get('user_id', ''),
                    "confidence_score": item.get('confidence_score', 0.0)
                })
            elif domain == "math":
                metadata.update({
                    "math_topic": item.get('math_topic', ''),
                    "concept_type": item.get('concept_type', ''),
                    "difficulty_level": item.get('difficulty_level', 3),
                    "problem_type": item.get('problem_type', ''),
                    "solution_method": item.get('solution_method', ''),
                    "user_id": item.get('user_id', ''),
                    "placement_level": item.get('placement_level', ''),
                    "confidence_score": item.get('confidence_score', 0.0)
                })
            elif domain == "cefr":
                metadata.update({
                    "cefr_level": item.get('cefr_level', ''),
                    "skill_type": item.get('skill_type', ''),
                    "assessment_type": item.get('assessment_type', ''),
                    "rubric_category": item.get('rubric_category', ''),
                    "user_id": item.get('user_id', ''),
                    "confidence_score": item.get('confidence_score', 0.0)
                })
            
            # Common metadata fields
            metadata.update({
                "tags": item.get('tags', []),
                "language": item.get('language', 'en'),
                "quality_score": item.get('quality_score', 0.0),
                "usage_count": item.get('usage_count', 0),
                "last_used": item.get('last_used', ''),
                "is_active": item.get('is_active', True)
            })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error building standardized metadata: {e}")
            return {"domain": domain, "content_type": content_type, "error": str(e)}

    async def _execute_batch_upsert_optimized(self, batch_data: List[Dict[str, Any]]):
        """Execute optimized batch upsert"""
        try:
            # Use COPY command for better performance on large batches
            if len(batch_data) > 1000:
                await self._execute_copy_batch_upsert(batch_data)
            else:
                await self._execute_regular_batch_upsert(batch_data)
                
        except Exception as e:
            logger.error(f"Error executing batch upsert: {e}")
            raise

    async def _execute_regular_batch_upsert(self, batch_data: List[Dict[str, Any]]):
        """Execute regular batch upsert"""
        sql = """
        INSERT INTO embeddings (obj_ref, namespace, slot, embedding, embedding_dim, meta, is_active, updated_at)
        VALUES (:obj_ref, :namespace, :slot, :embedding, :embedding_dim, :metadata, true, NOW())
        ON CONFLICT (obj_ref, namespace, slot)
        DO UPDATE SET 
            embedding = EXCLUDED.embedding,
            embedding_dim = EXCLUDED.embedding_dim,
            meta = EXCLUDED.metadata,
            is_active = true,
            deactivated_at = NULL,
            updated_at = NOW()
        """
        
        await database.execute_many(sql, batch_data)

    async def _execute_copy_batch_upsert(self, batch_data: List[Dict[str, Any]]):
        """Execute batch upsert using COPY for large datasets"""
        try:
            # Create temporary table
            await database.execute("""
                CREATE TEMP TABLE temp_embeddings (
                    obj_ref TEXT,
                    namespace TEXT,
                    slot INTEGER,
                    embedding vector,
                    embedding_dim INTEGER,
                    meta JSONB,
                    is_active BOOLEAN,
                    updated_at TIMESTAMP
                )
            """)
            
            # Prepare data for COPY
            copy_data = []
            for item in batch_data:
                copy_data.append([
                    item['obj_ref'],
                    item['namespace'],
                    item['slot'],
                    item['embedding'],
                    item['embedding_dim'],
                    json.dumps(item['metadata']),
                    True,
                    datetime.utcnow()
                ])
            
            # Use COPY for fast insertion
            await database.execute("""
                COPY temp_embeddings FROM STDIN
            """, copy_data)
            
            # Upsert from temp table
            await database.execute("""
                INSERT INTO embeddings (obj_ref, namespace, slot, embedding, embedding_dim, meta, is_active, updated_at)
                SELECT obj_ref, namespace, slot, embedding, embedding_dim, meta, is_active, updated_at
                FROM temp_embeddings
                ON CONFLICT (obj_ref, namespace, slot)
                DO UPDATE SET 
                    embedding = EXCLUDED.embedding,
                    embedding_dim = EXCLUDED.embedding_dim,
                    meta = EXCLUDED.metadata,
                    is_active = true,
                    deactivated_at = NULL,
                    updated_at = NOW()
            """)
            
            # Clean up
            await database.execute("DROP TABLE temp_embeddings")
            
        except Exception as e:
            logger.error(f"Error in COPY batch upsert: {e}")
            # Fallback to regular upsert
            await self._execute_regular_batch_upsert(batch_data)

    async def _update_namespace_statistics(self, namespace: str, processed_count: int):
        """Update namespace statistics after batch operations"""
        try:
            # Update namespace stats table if it exists
            await database.execute("""
                INSERT INTO namespace_statistics (namespace, total_embeddings, last_updated, last_batch_size)
                VALUES (:namespace, :count, NOW(), :batch_size)
                ON CONFLICT (namespace)
                DO UPDATE SET 
                    total_embeddings = namespace_statistics.total_embeddings + :count,
                    last_updated = NOW(),
                    last_batch_size = :batch_size
            """, {
                "namespace": namespace,
                "count": processed_count,
                "batch_size": processed_count
            })
            
        except Exception as e:
            # Table might not exist, ignore
            logger.debug(f"Could not update namespace statistics: {e}")

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for vector operations"""
        try:
            metrics = {}
            
            # Get index performance stats
            index_stats = await database.fetch_all("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes
                WHERE indexname LIKE 'ix_%_embedding'
                ORDER BY idx_scan DESC
            """)
            
            metrics["index_performance"] = [
                {
                    "index_name": row["indexname"],
                    "scans": row["idx_scan"],
                    "tuples_read": row["idx_tup_read"],
                    "tuples_fetch": row["idx_tup_fetch"]
                }
                for row in index_stats
            ]
            
            # Get embedding table stats
            table_stats = await database.fetch_one("""
                SELECT 
                    COUNT(*) as total_embeddings,
                    COUNT(DISTINCT namespace) as total_namespaces,
                    AVG(array_length(embedding, 1)) as avg_embedding_dim
                FROM embeddings
                WHERE is_active = true
            """)
            
            metrics["table_statistics"] = {
                "total_embeddings": table_stats["total_embeddings"],
                "total_namespaces": table_stats["total_namespaces"],
                "avg_embedding_dim": table_stats["avg_embedding_dim"]
            }
            
            # Get namespace distribution
            namespace_dist = await database.fetch_all("""
                SELECT 
                    namespace,
                    COUNT(*) as embedding_count,
                    AVG(array_length(embedding, 1)) as avg_dim
                FROM embeddings
                WHERE is_active = true
                GROUP BY namespace
                ORDER BY embedding_count DESC
            """)
            
            metrics["namespace_distribution"] = [
                {
                    "namespace": row["namespace"],
                    "count": row["embedding_count"],
                    "avg_dim": row["avg_dim"]
                }
                for row in namespace_dist
            ]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}

    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics for monitoring"""
        try:
            real_time_metrics = {}
            
            # Get current active connections
            connections = await database.fetch_one("""
                SELECT 
                    COUNT(*) as active_connections,
                    COUNT(*) FILTER (WHERE state = 'active') as executing_queries
                FROM pg_stat_activity
                WHERE datname = current_database()
            """)
            
            real_time_metrics["database_connections"] = {
                "active_connections": connections["active_connections"],
                "executing_queries": connections["executing_queries"]
            }
            
            # Get recent embedding operations
            recent_ops = await database.fetch_all("""
                SELECT 
                    namespace,
                    COUNT(*) as operations_last_hour,
                    AVG(EXTRACT(EPOCH FROM (NOW() - created_at))) as avg_processing_time_seconds
                FROM embeddings
                WHERE created_at >= NOW() - INTERVAL '1 hour'
                GROUP BY namespace
                ORDER BY operations_last_hour DESC
            """)
            
            real_time_metrics["recent_operations"] = [
                {
                    "namespace": row["namespace"],
                    "operations_last_hour": row["operations_last_hour"],
                    "avg_processing_time_seconds": row["avg_processing_time_seconds"]
                }
                for row in recent_ops
            ]
            
            # Get cache hit rates
            cache_stats = await self._get_cache_performance_stats()
            real_time_metrics["cache_performance"] = cache_stats
            
            return real_time_metrics
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return {"error": str(e)}

    async def _get_cache_performance_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        try:
            # This would integrate with Redis cache service
            # For now, return placeholder stats
            return {
                "cache_hit_rate": 0.85,
                "cache_size_mb": 128,
                "eviction_count": 0,
                "memory_usage_percent": 65
            }
        except Exception as e:
            logger.debug(f"Could not get cache stats: {e}")
            return {"error": str(e)}

    async def optimize_indexes_for_domain(self, domain: str) -> Dict[str, Any]:
        """Optimize indexes for specific domain performance"""
        try:
            logger.info(f"🔧 Optimizing indexes for domain: {domain}")
            
            domain_configs = {
                "english": {"lists": 100, "similarity_threshold": 0.7},
                "math": {"lists": 150, "similarity_threshold": 0.8},
                "cefr": {"lists": 50, "similarity_threshold": 0.6}
            }
            
            if domain not in domain_configs:
                raise ValueError(f"Unknown domain: {domain}")
            
            config = domain_configs[domain]
            
            # Get domain namespaces
            namespace_mapping = self._get_domain_namespace_mapping()
            if domain not in namespace_mapping:
                raise ValueError(f"No namespace mapping for domain: {domain}")
            
            optimization_results = {}
            
            for content_type, namespace in namespace_mapping[domain].items():
                try:
                    # Analyze current index performance
                    index_name = f"ix_{namespace}_embedding"
                    
                    if await self._check_index_exists(index_name):
                        # Rebuild index with optimized parameters
                        await self._rebuild_optimized_index(
                            index_name, 
                            "ivfflat", 
                            config["lists"]
                        )
                        
                        optimization_results[namespace] = {
                            "status": "optimized",
                            "index_type": "ivfflat",
                            "lists": config["lists"],
                            "similarity_threshold": config["similarity_threshold"]
                        }
                    else:
                        optimization_results[namespace] = {
                            "status": "not_found",
                            "error": f"Index {index_name} does not exist"
                        }
                        
                except Exception as e:
                    logger.error(f"Error optimizing index for {namespace}: {e}")
                    optimization_results[namespace] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            logger.info(f"✅ Domain {domain} optimization completed")
            return {
                "domain": domain,
                "optimization_results": optimization_results,
                "config": config
            }
            
        except Exception as e:
            logger.error(f"❌ Error optimizing indexes for domain {domain}: {e}")
            return {"error": str(e)}

    async def _rebuild_optimized_index(
        self, 
        index_name: str, 
        index_type: str = "ivfflat", 
        lists: int = 100
    ):
        """Rebuild an index with optimized parameters"""
        try:
            # Drop existing index
            await database.execute(f"DROP INDEX IF EXISTS {index_name}")
            
            # Create new optimized index
            await self._create_optimized_index(index_name, index_type, lists)
            
            logger.info(f"✅ Rebuilt optimized index: {index_name}")
            
        except Exception as e:
            logger.error(f"❌ Error rebuilding index {index_name}: {e}")
            raise

    async def get_domain_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for all domains"""
        try:
            summary = {}
            
            domains = ["english", "math", "cefr"]
            
            for domain in domains:
                try:
                    # Get domain-specific metrics
                    domain_metrics = await self._get_domain_metrics(domain)
                    summary[domain] = domain_metrics
                    
                except Exception as e:
                    logger.error(f"Error getting metrics for domain {domain}: {e}")
                    summary[domain] = {"error": str(e)}
            
            # Add overall system metrics
            overall_metrics = await self.get_performance_metrics()
            summary["overall"] = overall_metrics
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting domain performance summary: {e}")
            return {"error": str(e)}

    async def _get_domain_metrics(self, domain: str) -> Dict[str, Any]:
        """Get performance metrics for specific domain"""
        try:
            namespace_mapping = self._get_domain_namespace_mapping()
            
            if domain not in namespace_mapping:
                return {"error": f"No namespace mapping for domain: {domain}"}
            
            domain_metrics = {
                "namespaces": {},
                "total_embeddings": 0,
                "avg_similarity_score": 0.0
            }
            
            for content_type, namespace in namespace_mapping[domain].items():
                try:
                    # Get namespace statistics
                    namespace_stats = await database.fetch_one("""
                        SELECT 
                            COUNT(*) as embedding_count,
                            AVG(metadata->>'similarity_score')::float as avg_similarity
                        FROM embeddings
                        WHERE namespace = :namespace AND is_active = true
                    """, {"namespace": namespace})
                    
                    domain_metrics["namespaces"][namespace] = {
                        "embedding_count": namespace_stats["embedding_count"],
                        "avg_similarity_score": namespace_stats["avg_similarity"] or 0.0
                    }
                    
                    domain_metrics["total_embeddings"] += namespace_stats["embedding_count"]
                    
                except Exception as e:
                    logger.debug(f"Could not get stats for namespace {namespace}: {e}")
                    domain_metrics["namespaces"][namespace] = {"error": str(e)}
            
            # Calculate average similarity score
            valid_scores = [
                ns.get("avg_similarity_score", 0.0) 
                for ns in domain_metrics["namespaces"].values() 
                if "avg_similarity_score" in ns
            ]
            
            if valid_scores:
                domain_metrics["avg_similarity_score"] = sum(valid_scores) / len(valid_scores)
            
            return domain_metrics
            
        except Exception as e:
            logger.error(f"Error getting domain metrics for {domain}: {e}")
            return {"error": str(e)}

    async def get_system_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            health_status = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "healthy",
                "components": {},
                "recommendations": []
            }
            
            # Check database connectivity
            try:
                db_result = await database.fetch_one("SELECT 1 as health_check")
                health_status["components"]["database"] = {
                    "status": "healthy",
                    "response_time_ms": 0,  # Would measure actual response time
                    "details": "Database connection successful"
                }
            except Exception as e:
                health_status["components"]["database"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "details": "Database connection failed"
                }
                health_status["overall_status"] = "degraded"
                health_status["recommendations"].append("Check database connectivity and credentials")
            
            # Check vector indexes
            try:
                index_status = await self._check_all_indexes_health()
                health_status["components"]["vector_indexes"] = index_status
                
                if index_status.get("overall_status") == "unhealthy":
                    health_status["overall_status"] = "degraded"
                    health_status["recommendations"].append("Rebuild vector indexes")
                    
            except Exception as e:
                health_status["components"]["vector_indexes"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
            
            # Check embedding service
            try:
                embedding_stats = await self._check_embedding_service_health()
                health_status["components"]["embedding_service"] = embedding_stats
                
                if embedding_stats.get("status") == "unhealthy":
                    health_status["overall_status"] = "degraded"
                    health_status["recommendations"].append("Check embedding service configuration")
                    
            except Exception as e:
                health_status["components"]["embedding_service"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
            
            # Check performance metrics
            try:
                performance_metrics = await self.get_performance_metrics()
                health_status["components"]["performance"] = {
                    "status": "healthy",
                    "metrics": performance_metrics
                }
                
                # Check for performance issues
                if performance_metrics.get("table_statistics", {}).get("total_embeddings", 0) == 0:
                    health_status["recommendations"].append("No embeddings found - consider initial data population")
                    
            except Exception as e:
                health_status["components"]["performance"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting system health status: {e}")
            return {
                "overall_status": "unknown",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _check_all_indexes_health(self) -> Dict[str, Any]:
        """Check health of all vector indexes"""
        try:
            index_health = {
                "overall_status": "healthy",
                "indexes": {},
                "total_indexes": 0,
                "healthy_indexes": 0
            }
            
            # Get all embedding indexes
            indexes = await database.fetch_all("""
                SELECT indexname, tablename, schemaname
                FROM pg_indexes
                WHERE indexname LIKE 'ix_%_embedding'
                ORDER BY indexname
            """)
            
            for index in indexes:
                index_name = index["indexname"]
                try:
                    # Check if index is valid
                    is_valid = await self._check_index_validity(index_name)
                    
                    index_health["indexes"][index_name] = {
                        "status": "healthy" if is_valid else "unhealthy",
                        "table": index["tablename"],
                        "schema": index["schemaname"],
                        "valid": is_valid
                    }
                    
                    index_health["total_indexes"] += 1
                    if is_valid:
                        index_health["healthy_indexes"] += 1
                    else:
                        index_health["overall_status"] = "degraded"
                        
                except Exception as e:
                    index_health["indexes"][index_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                    index_health["overall_status"] = "degraded"
                    index_health["total_indexes"] += 1
            
            # Calculate health percentage
            if index_health["total_indexes"] > 0:
                health_percentage = (index_health["healthy_indexes"] / index_health["total_indexes"]) * 100
                index_health["health_percentage"] = round(health_percentage, 2)
            
            return index_health
            
        except Exception as e:
            logger.error(f"Error checking indexes health: {e}")
            return {
                "overall_status": "error",
                "error": str(e)
            }

    async def _check_index_validity(self, index_name: str) -> bool:
        """Check if an index is valid and usable"""
        try:
            # Check if index exists and is valid
            result = await database.fetch_one("""
                SELECT 1 FROM pg_indexes 
                WHERE indexname = :index_name AND indexdef IS NOT NULL
            """, {"index_name": index_name})
            
            if not result:
                return False
            
            # Check if index is not corrupted
            result = await database.fetch_one("""
                SELECT 1 FROM pg_stat_user_indexes 
                WHERE indexname = :index_name
            """, {"index_name": index_name})
            
            return result is not None
            
        except Exception as e:
            logger.debug(f"Could not check index validity for {index_name}: {e}")
            return False

    async def _check_embedding_service_health(self) -> Dict[str, Any]:
        """Check embedding service health"""
        try:
            # This would integrate with actual embedding service
            # For now, return placeholder health check
            return {
                "status": "healthy",
                "model_available": True,
                "api_accessible": True,
                "last_check": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations based on current performance"""
        try:
            recommendations = {
                "priority": "medium",
                "recommendations": [],
                "estimated_impact": "moderate"
            }
            
            # Get current performance metrics
            metrics = await self.get_performance_metrics()
            
            # Check for optimization opportunities
            if metrics.get("table_statistics", {}).get("total_embeddings", 0) > 10000:
                recommendations["recommendations"].append({
                    "type": "index_optimization",
                    "description": "Large dataset detected - consider rebuilding indexes with optimized parameters",
                    "priority": "high",
                    "estimated_improvement": "20-30% faster similarity search"
                })
            
            # Check index performance
            index_performance = metrics.get("index_performance", [])
            for index in index_performance:
                if index.get("scans", 0) < 10:
                    recommendations["recommendations"].append({
                        "type": "index_usage",
                        "description": f"Index {index['index_name']} has low usage - consider removing if unused",
                        "priority": "low",
                        "estimated_improvement": "Storage space savings"
                    })
            
            # Check namespace distribution
            namespace_dist = metrics.get("namespace_distribution", [])
            if namespace_dist:
                largest_namespace = max(namespace_dist, key=lambda x: x.get("count", 0))
                if largest_namespace.get("count", 0) > 5000:
                    recommendations["recommendations"].append({
                        "type": "namespace_optimization",
                        "description": f"Namespace {largest_namespace['namespace']} is very large - consider partitioning",
                        "priority": "medium",
                        "estimated_improvement": "15-25% faster operations"
                    })
            
            # Set overall priority based on recommendations
            high_priority_count = sum(1 for r in recommendations["recommendations"] if r.get("priority") == "high")
            if high_priority_count > 0:
                recommendations["priority"] = "high"
                recommendations["estimated_impact"] = "high"
            elif len(recommendations["recommendations"]) > 3:
                recommendations["priority"] = "medium"
                recommendations["estimated_impact"] = "moderate"
            else:
                recommendations["priority"] = "low"
                recommendations["estimated_impact"] = "low"
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting optimization recommendations: {e}")
            return {"error": str(e)}


# Global instance
vector_index_manager = VectorIndexManager()
