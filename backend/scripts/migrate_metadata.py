#!/usr/bin/env python3
"""
Metadata Migration Script

This script migrates existing metadata in the vector database to the new standardized format.
It ensures all existing embeddings have consistent metadata structure.
"""

import asyncio
import logging
from typing import Dict, Any, List
import sys
import os
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.database import database
from app.services.metadata_schema_service import metadata_schema_service
from app.services.vector_index_manager import vector_index_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate_existing_metadata():
    """Migrate existing metadata to standardized format"""
    try:
        logger.info("🚀 Starting metadata migration...")
        
        # Get all existing embeddings
        existing_embeddings = await database.fetch_all("""
            SELECT id, obj_ref, namespace, metadata, content
            FROM embeddings
            WHERE is_active = true
            ORDER BY created_at DESC
        """)
        
        logger.info(f"📊 Found {len(existing_embeddings)} existing embeddings to migrate")
        
        migration_stats = {
            "total": len(existing_embeddings),
            "migrated": 0,
            "errors": 0,
            "skipped": 0
        }
        
        for embedding in existing_embeddings:
            try:
                # Extract domain and content type from namespace
                domain, content_type = _extract_domain_and_type(embedding["namespace"])
                
                if not domain or not content_type:
                    logger.warning(f"⚠️ Could not extract domain/type from namespace: {embedding['namespace']}")
                    migration_stats["skipped"] += 1
                    continue
                
                # Validate and standardize existing metadata
                raw_metadata = embedding["metadata"] or {}
                standardized_metadata = metadata_schema_service.validate_existing_metadata(raw_metadata)
                
                # Ensure required fields
                standardized_metadata.update({
                    "domain": domain,
                    "content_type": content_type,
                    "obj_ref": embedding["obj_ref"],
                    "migrated_at": datetime.utcnow().isoformat(),
                    "original_metadata": raw_metadata
                })
                
                # Update embedding with standardized metadata
                await database.execute("""
                    UPDATE embeddings 
                    SET metadata = :metadata, updated_at = NOW()
                    WHERE id = :id
                """, {
                    "metadata": standardized_metadata,
                    "id": embedding["id"]
                })
                
                migration_stats["migrated"] += 1
                logger.info(f"✅ Migrated embedding {embedding['id']} ({domain}/{content_type})")
                
            except Exception as e:
                logger.error(f"❌ Error migrating embedding {embedding['id']}: {e}")
                migration_stats["errors"] += 1
        
        # Log migration summary
        logger.info("📈 Migration Summary:")
        logger.info(f"  Total embeddings: {migration_stats['total']}")
        logger.info(f"  Successfully migrated: {migration_stats['migrated']}")
        logger.info(f"  Errors: {migration_stats['errors']}")
        logger.info(f"  Skipped: {migration_stats['skipped']}")
        
        return migration_stats
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


def _extract_domain_and_type(namespace: str) -> tuple[str, str]:
    """Extract domain and content type from namespace"""
    try:
        if not namespace:
            return None, None
        
        # Handle different namespace formats
        if namespace.startswith("english_"):
            domain = "english"
            if namespace == "english_errors":
                content_type = "error_pattern"
            elif namespace == "english_questions":
                content_type = "question"
            elif namespace == "english_cloze_questions":
                content_type = "cloze_question"
            elif namespace == "english_grammar_rules":
                content_type = "grammar_rule"
            elif namespace == "english_vocabulary":
                content_type = "vocabulary"
            else:
                content_type = "question"
                
        elif namespace.startswith("math_"):
            domain = "math"
            if namespace == "math_errors":
                content_type = "error_pattern"
            elif namespace == "math_questions":
                content_type = "question"
            elif namespace == "math_concepts":
                content_type = "math_concept"
            elif namespace == "math_solutions":
                content_type = "math_solution"
            elif namespace == "math_placement_tests":
                content_type = "placement_test"
            else:
                content_type = "question"
                
        elif namespace.startswith("cefr_"):
            domain = "cefr"
            if namespace == "cefr_rubrics":
                content_type = "cefr_rubric"
            elif namespace == "cefr_examples":
                content_type = "cefr_example"
            else:
                content_type = "cefr_rubric"
                
        elif namespace.startswith("user_assessments"):
            domain = "cefr"
            content_type = "user_assessment"
            
        else:
            # Default fallback
            domain = "general"
            content_type = "question"
        
        return domain, content_type
        
    except Exception as e:
        logger.error(f"Error extracting domain/type from namespace {namespace}: {e}")
        return None, None


async def validate_migration_results():
    """Validate that migration was successful"""
    try:
        logger.info("🔍 Validating migration results...")
        
        # Check for missing required fields
        invalid_embeddings = await database.fetch_all("""
            SELECT id, obj_ref, namespace, metadata
            FROM embeddings
            WHERE is_active = true
            AND (
                metadata->>'domain' IS NULL
                OR metadata->>'content_type' IS NULL
                OR metadata->>'obj_ref' IS NULL
                OR metadata->>'created_at' IS NULL
            )
        """)
        
        if invalid_embeddings:
            logger.warning(f"⚠️ Found {len(invalid_embeddings)} embeddings with invalid metadata:")
            for embedding in invalid_embeddings[:5]:  # Show first 5
                logger.warning(f"  ID: {embedding['id']}, Namespace: {embedding['namespace']}")
        else:
            logger.info("✅ All embeddings have valid metadata structure")
        
        # Check metadata version distribution
        version_stats = await database.fetch_all("""
            SELECT 
                metadata->>'metadata_version' as version,
                COUNT(*) as count
            FROM embeddings
            WHERE is_active = true
            GROUP BY metadata->>'metadata_version'
            ORDER BY count DESC
        """)
        
        logger.info("📊 Metadata version distribution:")
        for stat in version_stats:
            logger.info(f"  Version {stat['version']}: {stat['count']} embeddings")
        
        return len(invalid_embeddings) == 0
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False


async def create_metadata_indexes():
    """Create indexes for optimized metadata queries"""
    try:
        logger.info("🔧 Creating metadata indexes...")
        
        # Create GIN index for metadata JSON fields
        await database.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_metadata_domain 
            ON embeddings USING GIN ((metadata->>'domain'))
        """)
        
        await database.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_metadata_content_type 
            ON embeddings USING GIN ((metadata->>'content_type'))
        """)
        
        await database.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_metadata_user_id 
            ON embeddings USING GIN ((metadata->>'user_id'))
        """)
        
        await database.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_metadata_difficulty 
            ON embeddings USING GIN ((metadata->>'difficulty_level'))
        """)
        
        logger.info("✅ Metadata indexes created successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating metadata indexes: {e}")


async def main():
    """Main migration function"""
    try:
        logger.info("🚀 Starting comprehensive metadata migration...")
        
        # 1. Migrate existing metadata
        migration_stats = await migrate_existing_metadata()
        
        # 2. Validate migration results
        validation_success = await validate_migration_results()
        
        # 3. Create optimized indexes
        await create_metadata_indexes()
        
        # 4. Summary
        logger.info("🎉 Metadata migration completed!")
        logger.info(f"📊 Final Stats: {migration_stats}")
        logger.info(f"✅ Validation: {'PASSED' if validation_success else 'FAILED'}")
        
        if not validation_success:
            logger.warning("⚠️ Some embeddings may have invalid metadata - check logs above")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("⏹️ Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
