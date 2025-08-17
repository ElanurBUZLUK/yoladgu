#!/usr/bin/env python3
"""
Script to setup pgvector extension and create vector indexes
"""

import asyncio
import logging
from typing import Dict, Any

from app.core.database import database
from app.services.vector_index_manager import vector_index_manager

logger = logging.getLogger(__name__)


async def setup_pgvector_extension():
    """Setup pgvector extension in PostgreSQL"""
    
    try:
        # Check if pgvector extension exists
        result = await database.fetch_one("""
            SELECT 1 FROM pg_available_extensions 
            WHERE name = 'vector'
        """)
        
        if not result:
            logger.error("❌ pgvector extension is not available in PostgreSQL")
            logger.info("Please install pgvector extension first:")
            logger.info("  - For Ubuntu/Debian: sudo apt-get install postgresql-14-pgvector")
            logger.info("  - For Docker: Use postgres:15-pgvector image")
            return False
        
        # Create extension if it doesn't exist
        await database.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("✅ pgvector extension created successfully")
        
        # Verify extension is working
        version_result = await database.fetch_one("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        if version_result:
            logger.info(f"✅ pgvector version: {version_result['extversion']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error setting up pgvector extension: {e}")
        return False


async def create_vector_columns():
    """Create vector columns in existing tables"""
    
    try:
        # Add embedding column to questions table
        await database.execute("""
            ALTER TABLE questions 
            ADD COLUMN IF NOT EXISTS content_embedding vector(1536);
        """)
        logger.info("✅ Added content_embedding column to questions table")
        
        # Add embedding column to error_patterns table
        await database.execute("""
            ALTER TABLE error_patterns 
            ADD COLUMN IF NOT EXISTS embedding vector(1536);
        """)
        logger.info("✅ Added embedding column to error_patterns table")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating vector columns: {e}")
        return False


async def create_vector_indexes():
    """Create vector indexes for similarity search"""
    
    try:
        # Create index for questions content embedding
        await database.execute("""
            CREATE INDEX IF NOT EXISTS questions_content_embedding_idx
            ON questions 
            USING ivfflat (content_embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        logger.info("✅ Created questions vector index")
        
        # Create index for error patterns embedding
        await database.execute("""
            CREATE INDEX IF NOT EXISTS error_patterns_embedding_idx
            ON error_patterns 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 50);
        """)
        logger.info("✅ Created error patterns vector index")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating vector indexes: {e}")
        return False


async def verify_vector_setup():
    """Verify that vector setup is working correctly"""
    
    try:
        # Check if extension exists
        ext_result = await database.fetch_one("""
            SELECT 1 FROM pg_extension WHERE extname = 'vector';
        """)
        
        if not ext_result:
            logger.error("❌ pgvector extension not found")
            return False
        
        # Check if columns exist
        questions_result = await database.fetch_one("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'questions' AND column_name = 'content_embedding';
        """)
        
        if not questions_result:
            logger.error("❌ content_embedding column not found in questions table")
            return False
        
        error_patterns_result = await database.fetch_one("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'error_patterns' AND column_name = 'embedding';
        """)
        
        if not error_patterns_result:
            logger.error("❌ embedding column not found in error_patterns table")
            return False
        
        # Check if indexes exist
        questions_idx_result = await database.fetch_one("""
            SELECT 1 FROM pg_indexes 
            WHERE indexname = 'questions_content_embedding_idx';
        """)
        
        if not questions_idx_result:
            logger.error("❌ questions vector index not found")
            return False
        
        error_patterns_idx_result = await database.fetch_one("""
            SELECT 1 FROM pg_indexes 
            WHERE indexname = 'error_patterns_embedding_idx';
        """)
        
        if not error_patterns_idx_result:
            logger.error("❌ error patterns vector index not found")
            return False
        
        logger.info("✅ Vector setup verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error verifying vector setup: {e}")
        return False


async def test_vector_operations():
    """Test basic vector operations"""
    
    try:
        # Test vector creation
        test_vector = [0.1] * 1536
        vector_str = f"[{','.join(map(str, test_vector))}]"
        
        # Test vector distance calculation
        result = await database.fetch_one(f"""
            SELECT '[0.1,0.1,0.1]'::vector <-> '[0.2,0.2,0.2]'::vector as distance;
        """)
        
        if result:
            distance = result['distance']
            logger.info(f"✅ Vector distance calculation test passed: {distance}")
        
        # Test vector similarity calculation
        result = await database.fetch_one(f"""
            SELECT 1 - ('[0.1,0.1,0.1]'::vector <=> '[0.2,0.2,0.2]'::vector) as similarity;
        """)
        
        if result:
            similarity = result['similarity']
            logger.info(f"✅ Vector similarity calculation test passed: {similarity}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing vector operations: {e}")
        return False


async def get_vector_statistics():
    """Get statistics about vector setup"""
    
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
        error_patterns_stats = await database.fetch_one("""
            SELECT 
                COUNT(*) as total_patterns,
                COUNT(embedding) as patterns_with_embeddings,
                COUNT(*) - COUNT(embedding) as patterns_without_embeddings
            FROM error_patterns
        """)
        
        # Get index statistics
        index_stats = await database.fetch_one("""
            SELECT 
                schemaname,
                tablename,
                indexname,
                indexdef
            FROM pg_indexes 
            WHERE indexname LIKE '%embedding%'
        """)
        
        logger.info("📊 Vector Statistics:")
        logger.info(f"   Questions: {questions_stats['total_questions']} total, {questions_stats['questions_with_embeddings']} with embeddings")
        logger.info(f"   Error Patterns: {error_patterns_stats['total_patterns']} total, {error_patterns_stats['patterns_with_embeddings']} with embeddings")
        
        if index_stats:
            logger.info(f"   Index: {index_stats['indexname']} on {index_stats['tablename']}")
        
        return {
            "questions": dict(questions_stats),
            "error_patterns": dict(error_patterns_stats),
            "index": dict(index_stats) if index_stats else None
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting vector statistics: {e}")
        return {}


async def main():
    """Main setup function"""
    
    logger.info("🚀 Starting pgvector setup...")
    
    try:
        # Connect to database
        await database.connect()
        logger.info("✅ Connected to database")
        
        # Setup pgvector extension
        if not await setup_pgvector_extension():
            return
        
        # Create vector columns
        if not await create_vector_columns():
            return
        
        # Create vector indexes
        if not await create_vector_indexes():
            return
        
        # Verify setup
        if not await verify_vector_setup():
            return
        
        # Test vector operations
        if not await test_vector_operations():
            return
        
        # Get statistics
        stats = await get_vector_statistics()
        
        logger.info("🎉 pgvector setup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        
    finally:
        await database.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
