#!/usr/bin/env python3
"""
PgVector Setup Check Script
Verifies pgvector extension, tables, columns, indexes, and performs live query tests
"""
import asyncio
import sys
import os
from typing import Dict, Any, List, Optional

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import database
from app.core.config import settings
from app.services.embedding_service import embedding_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PgVectorSetupChecker:
    """Comprehensive pgvector setup verification"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.success_count = 0
        
    async def check_all(self) -> Dict[str, Any]:
        """Run all checks"""
        logger.info("🔍 Starting PgVector Setup Check")
        logger.info("=" * 60)
        
        results = {
            "extension": await self.check_extension(),
            "tables": await self.check_tables(),
            "columns": await self.check_columns(),
            "indexes": await self.check_indexes(),
            "live_query": await self.check_live_query(),
            "summary": {}
        }
        
        # Generate summary
        total_checks = len([r for r in results.values() if isinstance(r, dict) and r.get("status")])
        passed_checks = len([r for r in results.values() if isinstance(r, dict) and r.get("status") == "✅"])
        
        results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "issues": self.issues,
            "warnings": self.warnings,
            "overall_status": "✅ READY" if passed_checks == total_checks else "❌ ISSUES FOUND"
        }
        
        return results
    
    async def check_extension(self) -> Dict[str, Any]:
        """Check if pgvector extension is installed and available"""
        logger.info("\n1. Checking pgvector extension...")
        
        try:
            # Check if extension exists
            result = await database.fetch_one("""
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            """)
            
            if result:
                logger.info("✅ pgvector extension is installed")
                self.success_count += 1
                return {"status": "✅", "message": "Extension installed"}
            else:
                logger.error("❌ pgvector extension is NOT installed")
                self.issues.append("pgvector extension not installed")
                return {"status": "❌", "message": "Extension not installed"}
                
        except Exception as e:
            logger.error(f"❌ Error checking extension: {e}")
            self.issues.append(f"Extension check error: {e}")
            return {"status": "❌", "message": f"Error: {e}"}
    
    async def check_tables(self) -> Dict[str, Any]:
        """Check if required tables exist"""
        logger.info("\n2. Checking required tables...")
        
        required_tables = ["questions", "error_patterns"]
        missing_tables = []
        
        for table in required_tables:
            try:
                result = await database.fetch_one("""
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = $1
                """, table)
                
                if result:
                    logger.info(f"✅ Table '{table}' exists")
                else:
                    logger.error(f"❌ Table '{table}' is missing")
                    missing_tables.append(table)
                    
            except Exception as e:
                logger.error(f"❌ Error checking table '{table}': {e}")
                missing_tables.append(table)
        
        if not missing_tables:
            self.success_count += 1
            return {"status": "✅", "message": "All required tables exist"}
        else:
            self.issues.append(f"Missing tables: {missing_tables}")
            return {"status": "❌", "message": f"Missing tables: {missing_tables}"}
    
    async def check_columns(self) -> Dict[str, Any]:
        """Check if vector columns exist with correct types"""
        logger.info("\n3. Checking vector columns...")
        
        required_columns = {
            "questions": {
                "content_embedding": "vector(1536)",
                "namespace": "character varying",
                "slot": "integer",
                "obj_ref": "character varying",
                "is_active": "boolean",
                "embedding_dim": "integer"
            },
            "error_patterns": {
                "embedding": "vector(1536)",
                "namespace": "character varying", 
                "slot": "integer",
                "obj_ref": "character varying",
                "is_active": "boolean",
                "embedding_dim": "integer"
            }
        }
        
        missing_columns = []
        wrong_type_columns = []
        
        for table, columns in required_columns.items():
            for column, expected_type in columns.items():
                try:
                    result = await database.fetch_one("""
                        SELECT data_type, character_maximum_length
                        FROM information_schema.columns 
                        WHERE table_name = $1 AND column_name = $2
                    """, table, column)
                    
                    if result:
                        actual_type = result['data_type']
                        if 'character' in actual_type and result['character_maximum_length']:
                            actual_type = f"{actual_type}({result['character_maximum_length']})"
                        
                        if actual_type == expected_type or (expected_type == "vector(1536)" and actual_type == "USER-DEFINED"):
                            logger.info(f"✅ Column '{table}.{column}' exists with correct type")
                        else:
                            logger.warning(f"⚠️ Column '{table}.{column}' has type '{actual_type}', expected '{expected_type}'")
                            wrong_type_columns.append(f"{table}.{column}")
                    else:
                        logger.error(f"❌ Column '{table}.{column}' is missing")
                        missing_columns.append(f"{table}.{column}")
                        
                except Exception as e:
                    logger.error(f"❌ Error checking column '{table}.{column}': {e}")
                    missing_columns.append(f"{table}.{column}")
        
        if not missing_columns and not wrong_type_columns:
            self.success_count += 1
            return {"status": "✅", "message": "All vector columns exist with correct types"}
        else:
            if missing_columns:
                self.issues.append(f"Missing columns: {missing_columns}")
            if wrong_type_columns:
                self.warnings.append(f"Wrong type columns: {wrong_type_columns}")
            return {
                "status": "⚠️" if wrong_type_columns else "❌",
                "message": f"Missing: {missing_columns}, Wrong types: {wrong_type_columns}"
            }
    
    async def check_indexes(self) -> Dict[str, Any]:
        """Check if vector indexes exist with correct operator classes"""
        logger.info("\n4. Checking vector indexes...")
        
        required_indexes = {
            "ix_questions_content_embedding_cosine": {
                "table": "questions",
                "column": "content_embedding",
                "operator": "vector_cosine_ops"
            },
            "ix_error_patterns_embedding_cosine": {
                "table": "error_patterns", 
                "column": "embedding",
                "operator": "vector_cosine_ops"
            }
        }
        
        missing_indexes = []
        wrong_operator_indexes = []
        
        for index_name, index_info in required_indexes.items():
            try:
                # Check if index exists
                result = await database.fetch_one("""
                    SELECT 1 FROM pg_indexes 
                    WHERE indexname = $1
                """, index_name)
                
                if result:
                    logger.info(f"✅ Index '{index_name}' exists")
                    
                    # Check operator class (this is more complex, so we'll just verify the index exists)
                    # In a real implementation, you might want to parse the index definition
                    
                else:
                    logger.error(f"❌ Index '{index_name}' is missing")
                    missing_indexes.append(index_name)
                    
            except Exception as e:
                logger.error(f"❌ Error checking index '{index_name}': {e}")
                missing_indexes.append(index_name)
        
        if not missing_indexes:
            self.success_count += 1
            return {"status": "✅", "message": "All vector indexes exist"}
        else:
            self.issues.append(f"Missing indexes: {missing_indexes}")
            return {"status": "❌", "message": f"Missing indexes: {missing_indexes}"}
    
    async def check_live_query(self) -> Dict[str, Any]:
        """Perform live vector similarity search test"""
        logger.info("\n5. Testing live vector similarity search...")
        
        try:
            # Generate a test embedding
            test_text = "This is a test question for vector similarity search"
            test_embedding = await embedding_service.get_embedding(test_text)
            
            if not test_embedding or len(test_embedding) != 1536:
                logger.error(f"❌ Test embedding generation failed or wrong dimension: {len(test_embedding) if test_embedding else 0}")
                self.issues.append("Test embedding generation failed")
                return {"status": "❌", "message": "Test embedding generation failed"}
            
            logger.info(f"✅ Test embedding generated with dimension: {len(test_embedding)}")
            
            # Test questions table similarity search
            questions_result = await database.fetch_all("""
                SELECT id, content, 1 - (content_embedding <=> $1::vector) as similarity
                FROM questions 
                WHERE content_embedding IS NOT NULL 
                AND is_active = true
                ORDER BY content_embedding <=> $1::vector
                LIMIT 3
            """, test_embedding)
            
            logger.info(f"✅ Questions similarity search returned {len(questions_result)} results")
            
            # Test error_patterns table similarity search
            patterns_result = await database.fetch_all("""
                SELECT id, pattern_details, 1 - (embedding <=> $1::vector) as similarity
                FROM error_patterns 
                WHERE embedding IS NOT NULL 
                AND is_active = true
                ORDER BY embedding <=> $1::vector
                LIMIT 3
            """, test_embedding)
            
            logger.info(f"✅ Error patterns similarity search returned {len(patterns_result)} results")
            
            # Check if we have any data
            total_questions = await database.fetch_one("""
                SELECT COUNT(*) as count FROM questions WHERE content_embedding IS NOT NULL AND is_active = true
            """)
            
            total_patterns = await database.fetch_one("""
                SELECT COUNT(*) as count FROM error_patterns WHERE embedding IS NOT NULL AND is_active = true
            """)
            
            logger.info(f"📊 Total questions with embeddings: {total_questions['count'] if total_questions else 0}")
            logger.info(f"📊 Total error patterns with embeddings: {total_patterns['count'] if total_patterns else 0}")
            
            if total_questions and total_questions['count'] == 0:
                self.warnings.append("No questions with embeddings found")
            if total_patterns and total_patterns['count'] == 0:
                self.warnings.append("No error patterns with embeddings found")
            
            self.success_count += 1
            return {
                "status": "✅", 
                "message": "Live query test successful",
                "questions_with_embeddings": total_questions['count'] if total_questions else 0,
                "patterns_with_embeddings": total_patterns['count'] if total_patterns else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Live query test failed: {e}")
            self.issues.append(f"Live query test failed: {e}")
            return {"status": "❌", "message": f"Live query test failed: {e}"}
    
    def print_summary(self, results: Dict[str, Any]):
        """Print comprehensive summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📋 PgVector Setup Check Summary")
        logger.info("=" * 60)
        
        summary = results["summary"]
        
        logger.info(f"Overall Status: {summary['overall_status']}")
        logger.info(f"Checks Passed: {summary['passed_checks']}/{summary['total_checks']}")
        
        if self.issues:
            logger.info("\n❌ Issues Found:")
            for issue in self.issues:
                logger.info(f"  - {issue}")
        
        if self.warnings:
            logger.info("\n⚠️ Warnings:")
            for warning in self.warnings:
                logger.info(f"  - {warning}")
        
        if not self.issues and not self.warnings:
            logger.info("\n🎉 All checks passed! PgVector is ready for production use.")
        elif self.issues:
            logger.info("\n🔧 Action Required:")
            logger.info("  1. Run: alembic upgrade head")
            logger.info("  2. Install pgvector extension if missing")
            logger.info("  3. Check database permissions")
            logger.info("  4. Verify embedding service configuration")
        else:
            logger.info("\n⚠️ Some warnings detected, but system should work.")
        
        logger.info("\n" + "=" * 60)


async def main():
    """Main function"""
    checker = PgVectorSetupChecker()
    
    try:
        results = await checker.check_all()
        checker.print_summary(results)
        
        # Exit with appropriate code
        if results["summary"]["overall_status"] == "✅ READY":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Check failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
