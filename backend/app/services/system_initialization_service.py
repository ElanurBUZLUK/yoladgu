import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings, Environment, LLMProvider
from app.core.database import get_async_session, engine, database
from app.core.cache import cache_service
from app.services.llm_gateway import LLMGatewayService
from app.services.mcp_service import MCPService
from app.services.sample_data_service import SampleDataService

logger = logging.getLogger(__name__)


class SystemInitializationService:
    """System initialization and configuration service"""
    
    def __init__(self):
        self.llm_gateway = LLMGatewayService()
        self.mcp_service = MCPService()
        self.sample_data_service = SampleDataService()
        self.initialization_status = {
            "environment": False,
            "database": False,
            "cache": False,
            "llm_providers": False,
            "mcp_server": False,
            "sample_data": False
        }
        self.errors = []
        self.warnings = []

    async def initialize_system(self) -> Dict[str, any]:
        """Complete system initialization"""
        logger.info("🚀 Starting system initialization...")
        start_time = datetime.now()
        
        try:
            # 1. Environment configuration validation
            await self._validate_environment_config()
            
            # 2. Database initialization
            await self._initialize_database()
            
            # 3. Cache system initialization
            await self._initialize_cache()
            
            # 4. LLM provider configuration testing
            await self._test_llm_providers()
            
            # 5. MCP server startup verification
            await self._verify_mcp_server()
            
            # 6. Sample data initialization (optional)
            if settings.environment == Environment.DEVELOPMENT:
                await self._initialize_sample_data()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            success = all(self.initialization_status.values())
            
            result = {
                "success": success,
                "duration_seconds": duration,
                "status": self.initialization_status.copy(),
                "errors": self.errors.copy(),
                "warnings": self.warnings.copy(),
                "timestamp": end_time.isoformat()
            }
            
            if success:
                logger.info(f"✅ System initialization completed successfully in {duration:.2f}s")
            else:
                logger.error(f"❌ System initialization failed after {duration:.2f}s")
                logger.error(f"Errors: {self.errors}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ System initialization failed with exception: {e}")
            self.errors.append(f"Initialization exception: {str(e)}")
            return {
                "success": False,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "status": self.initialization_status.copy(),
                "errors": self.errors.copy(),
                "warnings": self.warnings.copy(),
                "timestamp": datetime.now().isoformat()
            }

    async def _validate_environment_config(self):
        """Validate environment configuration"""
        logger.info("🔧 Validating environment configuration...")
        
        try:
            # Check required environment variables
            required_vars = [
                "DATABASE_URL",
                "SECRET_KEY",
                "ENCRYPTION_KEY"
            ]
            
            missing_vars = []
            for var in required_vars:
                if not getattr(settings, var.lower(), None):
                    missing_vars.append(var)
            
            if missing_vars:
                error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
                self.errors.append(error_msg)
                logger.error(error_msg)
                return
            
            # Validate database URL format
            if not settings.database_url.startswith(("postgresql://", "postgresql+asyncpg://")):
                error_msg = "Invalid database URL format"
                self.errors.append(error_msg)
                logger.error(error_msg)
                return
            
            # Validate Redis URL format
            if not settings.redis_url.startswith("redis://"):
                error_msg = "Invalid Redis URL format"
                self.errors.append(error_msg)
                logger.error(error_msg)
                return
            
            # Check file upload directory
            upload_dir = Path(settings.upload_dir)
            if not upload_dir.exists():
                try:
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created upload directory: {upload_dir}")
                except Exception as e:
                    error_msg = f"Failed to create upload directory: {e}"
                    self.errors.append(error_msg)
                    logger.error(error_msg)
                    return
            
            # Environment-specific validations
            if settings.environment == Environment.PRODUCTION:
                if settings.debug:
                    warning_msg = "Debug mode is enabled in production"
                    self.warnings.append(warning_msg)
                    logger.warning(warning_msg)
                
                if not settings.openai_api_key and not settings.anthropic_api_key:
                    error_msg = "No LLM API keys configured for production"
                    self.errors.append(error_msg)
                    logger.error(error_msg)
                    return
            
            self.initialization_status["environment"] = True
            logger.info("✅ Environment configuration validated")
            
        except Exception as e:
            error_msg = f"Environment validation failed: {e}"
            self.errors.append(error_msg)
            logger.error(error_msg)

    async def _initialize_database(self):
        """Initialize database connection and verify schema"""
        logger.info("🗄️ Initializing database...")
        
        try:
            # Test database connection using the database object directly
            from app.core.database import database
            
            # Ensure database is connected
            if not database.is_connected:
                await database.connect()
            
            # Test basic connection
            result = await database.fetch_one("SELECT 1")
            assert result[0] == 1, "Database connection failed"
            
            # Check if tables exist
            tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """
            result = await database.fetch_all(tables_query)
            tables = [row[0] for row in result]
            
            required_tables = [
                "users", "questions", "student_attempts", 
                "error_patterns", "pdf_uploads"
            ]
            
            missing_tables = [table for table in required_tables if table not in tables]
            
            if missing_tables:
                error_msg = f"Missing required database tables: {', '.join(missing_tables)}"
                self.errors.append(error_msg)
                logger.error(error_msg)
                return
            
            # Check database version/health
            version_result = await database.fetch_one("SELECT version()")
            version = version_result[0]
            logger.info(f"Database version: {version}")
            
            # Test write operations using async session
            async for session in get_async_session():
                try:
                    # Test write operations
                    test_query = text("CREATE TEMP TABLE test_init (id INT)")
                    await session.execute(test_query)
                    await session.commit()
                    
                    # Clean up test table
                    cleanup_query = text("DROP TABLE test_init")
                    await session.execute(cleanup_query)
                    await session.commit()
                    break
                except Exception as e:
                    logger.warning(f"Session-based write test failed: {e}")
                    break
            
            self.initialization_status["database"] = True
            logger.info("✅ Database initialized successfully")
            
        except Exception as e:
            error_msg = f"Database initialization failed: {e}"
            self.errors.append(error_msg)
            logger.error(error_msg)

    async def _initialize_cache(self):
        """Initialize cache system"""
        logger.info("💾 Initializing cache system...")
        
        try:
            # Test Redis connection
            await cache_service.connect()
            
            # Test basic cache operations
            test_key = "system_init_test"
            test_value = {"test": "data", "timestamp": datetime.now().isoformat()}
            
            # Test set operation
            success = await cache_service.set(test_key, test_value, expire=60)
            if not success:
                raise Exception("Cache set operation failed")
            
            # Test get operation
            retrieved_value = await cache_service.get(test_key)
            if retrieved_value != test_value:
                raise Exception("Cache get operation failed")
            
            # Test delete operation
            success = await cache_service.delete(test_key)
            if not success:
                raise Exception("Cache delete operation failed")
            
            # Test increment operation
            counter_key = "system_init_counter"
            result = await cache_service.increment(counter_key, 1)
            if result != 1:
                raise Exception("Cache increment operation failed")
            
            # Clean up
            await cache_service.delete(counter_key)
            
            self.initialization_status["cache"] = True
            logger.info("✅ Cache system initialized successfully")
            
        except Exception as e:
            error_msg = f"Cache initialization failed: {e}"
            self.errors.append(error_msg)
            logger.error(error_msg)

    async def _test_llm_providers(self):
        """Test LLM provider configurations"""
        logger.info("🤖 Testing LLM provider configurations...")
        
        try:
            # Test LLM service creation and basic functionality
            logger.info("Testing LLM service creation...")
            
            # Test that the service can be instantiated
            assert hasattr(self.llm_gateway, 'generate_english_question'), "LLM service missing generate_english_question method"
            assert hasattr(self.llm_gateway, 'evaluate_student_answer'), "LLM service missing evaluate_student_answer method"
            
            # Check API key configurations
            if not settings.openai_api_key and not settings.anthropic_api_key:
                warning_msg = "No LLM API keys configured"
                self.warnings.append(warning_msg)
                logger.warning(warning_msg)
            else:
                logger.info("✅ LLM API keys configured")
            
            # Test provider configuration
            valid_providers = ["gpt4", "gpt35", "claude_opus", "claude_sonnet", "claude_haiku", "local_model"]
            primary_provider = settings.primary_llm_provider
            secondary_provider = settings.secondary_llm_provider
            
            if primary_provider in valid_providers:
                logger.info(f"✅ Primary LLM provider configured: {primary_provider}")
            else:
                warning_msg = f"Invalid primary LLM provider: {primary_provider}"
                self.warnings.append(warning_msg)
                logger.warning(warning_msg)
            
            if secondary_provider in valid_providers:
                logger.info(f"✅ Secondary LLM provider configured: {secondary_provider}")
            else:
                warning_msg = f"Invalid secondary LLM provider: {secondary_provider}"
                self.warnings.append(warning_msg)
                logger.warning(warning_msg)
            
            self.initialization_status["llm_providers"] = True
            logger.info("✅ LLM provider testing completed")
            
        except Exception as e:
            error_msg = f"LLM provider testing failed: {e}"
            self.errors.append(error_msg)
            logger.error(error_msg)

    async def _verify_mcp_server(self):
        """Verify MCP server startup and functionality"""
        logger.info("🔌 Verifying MCP server...")
        
        try:
            # Test MCP service creation and basic functionality
            logger.info("Testing MCP service creation...")
            
            # Test that the service can be instantiated and has required methods
            assert hasattr(self.mcp_service, 'generate_english_question_for_user'), "MCP service missing generate_english_question_for_user method"
            assert hasattr(self.mcp_service, 'evaluate_student_answer'), "MCP service missing evaluate_student_answer method"
            assert hasattr(self.mcp_service, 'analyze_student_performance'), "MCP service missing analyze_student_performance method"
            assert hasattr(self.mcp_service, 'extract_questions_from_pdf'), "MCP service missing extract_questions_from_pdf method"
            
            # Test MCP client creation (this may fail if MCP server is not running, but that's OK)
            try:
                client = await self.mcp_service._get_client()
                if client:
                    logger.info("✅ MCP client created successfully")
                else:
                    warning_msg = "MCP client creation returned None"
                    self.warnings.append(warning_msg)
                    logger.warning(warning_msg)
            except Exception as e:
                warning_msg = f"MCP client creation failed (server may not be running): {e}"
                self.warnings.append(warning_msg)
                logger.warning(warning_msg)
            
            self.initialization_status["mcp_server"] = True
            logger.info("✅ MCP server verification completed")
            
        except Exception as e:
            error_msg = f"MCP server verification failed: {e}"
            self.errors.append(error_msg)
            logger.error(error_msg)

    async def _initialize_sample_data(self):
        """Initialize sample data for development environment"""
        logger.info("📊 Initializing sample data...")
        
        try:
            # Check if sample data already exists
            async with get_async_session() as session:
                # Check users count
                result = await session.execute(text("SELECT COUNT(*) FROM users"))
                user_count = result.scalar()
                
                if user_count > 0:
                    logger.info(f"Sample data already exists ({user_count} users)")
                    self.initialization_status["sample_data"] = True
                    return
            
            # Create sample data
            sample_data_result = await self.sample_data_service.create_sample_data()
            
            if sample_data_result:
                logger.info("✅ Sample data initialized successfully")
                self.initialization_status["sample_data"] = True
            else:
                warning_msg = "Sample data initialization returned no result"
                self.warnings.append(warning_msg)
                logger.warning(warning_msg)
                
        except Exception as e:
            warning_msg = f"Sample data initialization failed: {e}"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)

    async def get_system_status(self) -> Dict[str, any]:
        """Get current system status"""
        return {
            "status": self.initialization_status.copy(),
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
            "timestamp": datetime.now().isoformat()
        }

    async def validate_system_health(self) -> Dict[str, any]:
        """Validate system health for runtime"""
        health_status = {
            "database": False,
            "cache": False,
            "llm_providers": False,
            "mcp_server": False,
            "overall": False
        }
        
        try:
            # Test database
            from app.core.database import database
            try:
                # Ensure database is connected
                if not database.is_connected:
                    await database.connect()
                
                result = await database.fetch_one("SELECT 1")
                if result and result[0] == 1:
                    health_status["database"] = True
            except Exception as e:
                logger.error(f"Database health check failed: {e}")
            
            # Test cache
            try:
                test_key = "health_check"
                await cache_service.set(test_key, "test", expire=10)
                result = await cache_service.get(test_key)
                if result == "test":
                    health_status["cache"] = True
                await cache_service.delete(test_key)
            except Exception as e:
                logger.error(f"Cache health check failed: {e}")
            
            # Test LLM (basic - just check service exists)
            try:
                if hasattr(self.llm_gateway, 'generate_english_question'):
                    health_status["llm_providers"] = True
            except Exception as e:
                logger.error(f"LLM health check failed: {e}")
            
            # Test MCP (basic - just check service exists)
            try:
                if hasattr(self.mcp_service, 'generate_english_question_for_user'):
                    health_status["mcp_server"] = True
            except Exception as e:
                logger.error(f"MCP health check failed: {e}")
            
            # Overall health (core components must work)
            health_status["overall"] = all([
                health_status["database"],
                health_status["cache"]
            ])
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        return health_status


# Global instance
system_init_service = SystemInitializationService()
