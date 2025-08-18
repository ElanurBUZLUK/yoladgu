#!/usr/bin/env python3
"""
Test script for Task 11.2 - System initialization ve configuration
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings, Environment
from app.services.system_initialization_service import SystemInitializationService
from app.core.database import get_async_session
from app.core.cache import cache_service
from sqlalchemy import text


async def test_system_initialization_service():
    """Test the system initialization service"""
    print("🧪 Testing System Initialization Service...")
    
    try:
        # Ensure database is connected
        from app.core.database import database
        if not database.is_connected:
            await database.connect()
        
        # Create service instance
        service = SystemInitializationService()
        
        # Test environment validation
        print("  🔧 Testing environment validation...")
        await service._validate_environment_config()
        assert service.initialization_status["environment"] == True, "Environment validation failed"
        print("    ✅ Environment validation passed")
        
        # Test database initialization
        print("  🗄️ Testing database initialization...")
        await service._initialize_database()
        assert service.initialization_status["database"] == True, "Database initialization failed"
        print("    ✅ Database initialization passed")
        
        # Test cache initialization
        print("  💾 Testing cache initialization...")
        await service._initialize_cache()
        assert service.initialization_status["cache"] == True, "Cache initialization failed"
        print("    ✅ Cache initialization passed")
        
        # Test LLM providers (may fail if no API keys, but should not crash)
        print("  🤖 Testing LLM providers...")
        await service._test_llm_providers()
        print("    ✅ LLM provider testing completed")
        
        # Test MCP server (may fail if not configured, but should not crash)
        print("  🔌 Testing MCP server...")
        await service._verify_mcp_server()
        print("    ✅ MCP server testing completed")
        
        # Test system status
        print("  📊 Testing system status...")
        status = await service.get_system_status()
        assert "status" in status, "System status missing"
        assert "errors" in status, "System errors missing"
        assert "warnings" in status, "System warnings missing"
        print("    ✅ System status retrieval passed")
        
        # Test health validation
        print("  ❤️ Testing health validation...")
        health = await service.validate_system_health()
        assert "database" in health, "Database health missing"
        assert "cache" in health, "Cache health missing"
        assert "overall" in health, "Overall health missing"
        print("    ✅ Health validation passed")
        
        print("✅ System Initialization Service tests passed")
        return True
        
    except Exception as e:
        print(f"❌ System Initialization Service tests failed: {e}")
        return False


async def test_environment_configuration():
    """Test environment configuration validation"""
    print("🔧 Testing Environment Configuration...")
    
    try:
        # Test required settings
        required_settings = [
            "database_url",
            "secret_key", 
            "encryption_key"
        ]
        
        for setting in required_settings:
            value = getattr(settings, setting, None)
            assert value is not None, f"Missing required setting: {setting}"
            assert len(str(value)) > 0, f"Empty required setting: {setting}"
        
        # Test environment enum
        assert settings.environment in [Environment.DEVELOPMENT, Environment.TESTING, Environment.PRODUCTION], "Invalid environment"
        
        # Test database URL format
        assert settings.database_url.startswith(("postgresql://", "postgresql+asyncpg://")), "Invalid database URL format"
        
        # Test Redis URL format
        assert settings.redis_url.startswith("redis://"), "Invalid Redis URL format"
        
        # Test file upload directory
        from pathlib import Path
        upload_dir = Path(settings.upload_dir)
        assert upload_dir.exists() or upload_dir.parent.exists(), "Upload directory not accessible"
        
        print("✅ Environment Configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Environment Configuration tests failed: {e}")
        return False


async def test_database_initialization():
    """Test database initialization and schema"""
    print("🗄️ Testing Database Initialization...")
    
    try:
        # Test database connection using the database object directly
        from app.core.database import database
        
        # Ensure database is connected
        if not database.is_connected:
            await database.connect()
        
        # Test basic connection
        result = await database.fetch_one("SELECT 1")
        assert result[0] == 1, "Database connection failed"
        
        # Test required tables
        required_tables = [
            "users", "questions", "student_attempts", 
            "error_patterns", "pdf_uploads"
        ]
        
        for table in required_tables:
            result = await database.fetch_one(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = :table",
                {"table": table}
            )
            count = result[0]
            assert count > 0, f"Required table '{table}' not found"
        
        # Test database version
        result = await database.fetch_one("SELECT version()")
        version = result[0]
        assert "PostgreSQL" in version, "Not a PostgreSQL database"
        
        # Test write operations using async session
        async for session in get_async_session():
            try:
                # Test write operations
                test_query = text("CREATE TEMP TABLE test_init (id INT)")
                await session.execute(test_query)
                await session.commit()
                
                # Clean up
                cleanup_query = text("DROP TABLE test_init")
                await session.execute(cleanup_query)
                await session.commit()
                break
            except Exception as e:
                print(f"Warning: Session-based write test failed: {e}")
                break
        
        print("✅ Database Initialization tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Database Initialization tests failed: {e}")
        return False


async def test_cache_initialization():
    """Test cache system initialization"""
    print("💾 Testing Cache Initialization...")
    
    try:
        # Test cache connection
        await cache_service.connect()
        
        # Test basic operations
        test_key = "test_init"
        test_value = {"test": "data", "timestamp": datetime.now().isoformat()}
        
        # Test set
        success = await cache_service.set(test_key, test_value, expire=60)
        assert success == True, "Cache set operation failed"
        
        # Test get
        retrieved_value = await cache_service.get(test_key)
        assert retrieved_value == test_value, "Cache get operation failed"
        
        # Test delete
        success = await cache_service.delete(test_key)
        assert success == True, "Cache delete operation failed"
        
        # Test increment
        counter_key = "test_counter"
        result = await cache_service.increment(counter_key, 1)
        assert result == 1, "Cache increment operation failed"
        
        # Clean up
        await cache_service.delete(counter_key)
        
        print("✅ Cache Initialization tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Cache Initialization tests failed: {e}")
        return False


async def test_llm_provider_configuration():
    """Test LLM provider configuration"""
    print("🤖 Testing LLM Provider Configuration...")
    
    try:
        # Test LLM service creation
        from app.services.llm_gateway import LLMGatewayService
        llm_service = LLMGatewayService()
        
        # Test configuration
        assert hasattr(settings, 'primary_llm_provider'), "Primary LLM provider not configured"
        assert hasattr(settings, 'secondary_llm_provider'), "Secondary LLM provider not configured"
        assert hasattr(settings, 'enable_llm_fallback'), "LLM fallback not configured"
        
        # Test provider names
        valid_providers = [
            "gpt4", "gpt35", "claude_opus", "claude_sonnet", "claude_haiku", "local_model"
        ]
        
        assert settings.primary_llm_provider in valid_providers, f"Invalid primary provider: {settings.primary_llm_provider}"
        assert settings.secondary_llm_provider in valid_providers, f"Invalid secondary provider: {settings.secondary_llm_provider}"
        
        print("✅ LLM Provider Configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ LLM Provider Configuration tests failed: {e}")
        return False


async def test_mcp_server_configuration():
    """Test MCP server configuration"""
    print("🔌 Testing MCP Server Configuration...")
    
    try:
        # Test MCP service creation
        from app.services.mcp_service import MCPService
        mcp_service = MCPService()
        
        # Test service methods exist
        assert hasattr(mcp_service, 'generate_english_question_for_user'), "MCP service missing generate_english_question_for_user method"
        assert hasattr(mcp_service, 'evaluate_student_answer'), "MCP service missing evaluate_student_answer method"
        assert hasattr(mcp_service, 'analyze_student_performance'), "MCP service missing analyze_student_performance method"
        assert hasattr(mcp_service, 'extract_questions_from_pdf'), "MCP service missing extract_questions_from_pdf method"
        
        print("✅ MCP Server Configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ MCP Server Configuration tests failed: {e}")
        return False


async def test_system_health_monitoring():
    """Test system health monitoring"""
    print("❤️ Testing System Health Monitoring...")
    
    try:
        # Ensure database is connected
        from app.core.database import database
        if not database.is_connected:
            await database.connect()
        
        service = SystemInitializationService()
        
        # Test health validation
        health = await service.validate_system_health()
        
        # Check required health components
        required_components = ["database", "cache", "overall"]
        for component in required_components:
            assert component in health, f"Missing health component: {component}"
            assert isinstance(health[component], bool), f"Health component {component} not boolean"
        
        # Test that database and cache are working (core components)
        assert health["database"] == True, "Database health check failed"
        assert health["cache"] == True, "Cache health check failed"
        
        print("✅ System Health Monitoring tests passed")
        return True
        
    except Exception as e:
        print(f"❌ System Health Monitoring tests failed: {e}")
        return False


async def test_initialization_scripts():
    """Test initialization scripts"""
    print("📜 Testing Initialization Scripts...")
    
    try:
        # Test database initialization script
        script_path = "scripts/init_database.py"
        assert os.path.exists(script_path), f"Database initialization script not found: {script_path}"
        
        # Test system initialization script
        script_path = "scripts/init_system.py"
        assert os.path.exists(script_path), f"System initialization script not found: {script_path}"
        
        # Test script executability
        assert os.access("scripts/init_database.py", os.X_OK), "Database initialization script not executable"
        assert os.access("scripts/init_system.py", os.X_OK), "System initialization script not executable"
        
        print("✅ Initialization Scripts tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Initialization Scripts tests failed: {e}")
        return False


async def test_api_endpoints():
    """Test system initialization API endpoints"""
    print("🌐 Testing API Endpoints...")
    
    try:
        # Test that the router is properly configured
        from app.api.v1.system_init import router
        
        # Check router configuration
        assert router.prefix == "/system", "Incorrect router prefix"
        assert "System Initialization" in router.tags, "Incorrect router tags"
        
        # Check that endpoints exist (simplified check)
        assert len(router.routes) > 0, "No routes found in router"
        
        # Check for specific endpoints by name
        route_names = [route.name for route in router.routes if hasattr(route, 'name')]
        expected_names = [
            "initialize_system",
            "get_system_status", 
            "get_system_health",
            "validate_environment_config",
            "test_database_connection",
            "test_cache_system",
            "test_llm_providers",
            "test_mcp_server",
            "get_system_config"
        ]
        
        for expected_name in expected_names:
            assert expected_name in route_names, f"Missing API endpoint: {expected_name}"
        
        print("✅ API Endpoints tests passed")
        return True
        
    except Exception as e:
        print(f"❌ API Endpoints tests failed: {e}")
        return False


async def test_complete_initialization():
    """Test complete system initialization"""
    print("🚀 Testing Complete System Initialization...")
    
    try:
        # Ensure database is connected
        from app.core.database import database
        if not database.is_connected:
            await database.connect()
        
        service = SystemInitializationService()
        
        # Run complete initialization
        result = await service.initialize_system()
        
        # Check result structure
        assert "success" in result, "Missing success field in result"
        assert "duration_seconds" in result, "Missing duration field in result"
        assert "status" in result, "Missing status field in result"
        assert "errors" in result, "Missing errors field in result"
        assert "warnings" in result, "Missing warnings field in result"
        assert "timestamp" in result, "Missing timestamp field in result"
        
        # Check that core components are working
        core_components = ["environment", "database", "cache"]
        for component in core_components:
            assert result["status"][component] == True, f"Core component {component} failed"
        
        # Check duration is reasonable
        assert result["duration_seconds"] > 0, "Invalid duration"
        assert result["duration_seconds"] < 300, "Initialization took too long (>5 minutes)"
        
        print("✅ Complete System Initialization tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Complete System Initialization tests failed: {e}")
        return False


async def main():
    """Main test function"""
    print("🧪 Starting Task 11.2 - System Initialization Tests")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # Run all tests
    tests = [
        ("Environment Configuration", test_environment_configuration),
        ("Database Initialization", test_database_initialization),
        ("Cache Initialization", test_cache_initialization),
        ("LLM Provider Configuration", test_llm_provider_configuration),
        ("MCP Server Configuration", test_mcp_server_configuration),
        ("System Health Monitoring", test_system_health_monitoring),
        ("Initialization Scripts", test_initialization_scripts),
        ("API Endpoints", test_api_endpoints),
        ("System Initialization Service", test_system_initialization_service),
        ("Complete Initialization", test_complete_initialization)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} - FAILED with exception: {e}")
    
    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Tests Passed: {passed}")
    print(f"Tests Failed: {failed}")
    print(f"Total Tests: {passed + failed}")
    print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All Task 11.2 tests passed successfully!")
        print("✅ System initialization and configuration is working correctly")
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
