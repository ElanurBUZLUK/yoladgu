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

logger = logging.getLogger(__name__)


class VectorIndexManager:
    """Production-ready vector index manager with real embeddings"""
    
    def __init__(self):
        self.embedding_dimension = 1536
        self.index_name_questions = "questions_content_embedding_idx"
        self.index_name_errors = "error_patterns_embedding_idx"
        self.cache_ttl = 1800  # 30 minutes
        
    async def create_vector_indexes(self):
        """Create vector indexes for similarity search"""
        
        try:
            # Create pgvector extension
            await database.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Add embedding columns if they don't exist
            await self._add_embedding_columns()
            
            # Create indexes
            await self._create_question_index()
            await self._create_error_pattern_index()
            
            logger.info("✅ Vector indexes created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating vector indexes: {e}")
            return False
    
    async def _add_embedding_columns(self):
        """Add embedding columns to tables"""
        
        # Add embedding column to questions table
        try:
            await database.execute("""
                ALTER TABLE questions 
                ADD COLUMN IF NOT EXISTS content_embedding vector(1536);
            """)
        except Exception as e:
            logger.warning(f"Questions embedding column might already exist: {e}")
        
        # Add embedding column to error_patterns table
        try:
            await database.execute("""
                ALTER TABLE error_patterns 
                ADD COLUMN IF NOT EXISTS embedding vector(1536);
            """)
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
                pattern_text = f"{pattern.error_type} {pattern.pattern_details} {pattern.topic_category}"
                
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
            return await embedding_service.get_embedding(text)
            
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dimension
    
    async def perform_vector_search(
        self,
        query_embedding: List[float],
        table_name: str,
        filters: Dict[str, Any] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        
        try:
            # Build the search query
            if table_name == "questions":
                search_query = f"""
                    SELECT 
                        id,
                        content,
                        topic,
                        difficulty_level,
                        question_type,
                        1 - (content_embedding <=> $1::vector) as similarity
                    FROM questions
                    WHERE content_embedding IS NOT NULL
                """
                
                # Add filters
                if filters:
                    if "subject" in filters:
                        search_query += f" AND subject = '{filters['subject']}'"
                    if "topic" in filters:
                        search_query += f" AND topic = '{filters['topic']}'"
                    if "difficulty_level" in filters:
                        search_query += f" AND difficulty_level = {filters['difficulty_level']}"
                    if "is_active" in filters:
                        search_query += f" AND is_active = {filters['is_active']}"
                
            elif table_name == "error_patterns":
                search_query = f"""
                    SELECT 
                        id,
                        error_type,
                        pattern_details,
                        topic_category,
                        1 - (embedding <=> $1::vector) as similarity
                    FROM error_patterns
                    WHERE embedding IS NOT NULL
                """
                
                # Add filters
                if filters:
                    if "subject" in filters:
                        search_query += f" AND subject = '{filters['subject']}'"
                    if "topic_category" in filters:
                        search_query += f" AND topic_category = '{filters['topic_category']}'"
            
            # Add similarity ordering and limit
            search_query += f"""
                ORDER BY similarity DESC
                LIMIT {limit}
            """
            
            # Execute search
            result = await database.fetch_all(search_query, query_embedding)
            
            # Convert to list of dicts
            search_results = []
            for row in result:
                search_results.append(dict(row))
            
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
                "questions": dict(questions_stats),
                "error_patterns": dict(errors_stats),
                "index_status": {
                    "questions_index": await self._check_index_exists(self.index_name_questions),
                    "errors_index": await self._check_index_exists(self.index_name_errors)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting index statistics: {e}")
            return {}
    
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


# Global instance
vector_index_manager = VectorIndexManager()
