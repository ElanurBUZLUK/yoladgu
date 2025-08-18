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
        self.index_name_questions = "ix_questions_content_embedding_cosine"
        self.index_name_errors = "ix_error_patterns_embedding_cosine"
        self.cache_ttl = 1800  # 30 minutes
        self.batch_size = settings.vector_batch_size
        
    def _get_redis_client(self) -> Redis:
        """Get Redis client for distributed locks"""
        return Redis.from_url(settings.redis_url)
        
    @lock_decorator(
        key_builder=lambda self: "lock:vector:create_indexes",
        ttl_ms=120_000,  # 2 minutes lock
        wait_timeout_ms=60_000,  # 1 minute wait
        redis_client_factory=lambda self: self._get_redis_client()
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
        redis_client_factory=lambda self: self._get_redis_client()
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
        redis_client_factory=lambda self: self._get_redis_client()
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
                WHERE indexname = $1
            """, index_name)
            
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


# Global instance
vector_index_manager = VectorIndexManager()
