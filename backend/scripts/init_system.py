#!/usr/bin/env python3
"""
System initialization script
Run this to set up the complete system including database, cache, and configuration
"""

import asyncio
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings, Environment
from app.services.system_initialization_service import SystemInitializationService


async def init_system(create_sample_data: bool = False, force: bool = False):
    """Initialize the complete system"""
    print("🚀 Starting system initialization...")
    print(f"Environment: {settings.environment}")
    print(f"Database: {settings.database_url.split('@')[-1] if '@' in settings.database_url else 'configured'}")
    print(f"Redis: {settings.redis_url.split('@')[-1] if '@' in settings.redis_url else 'configured'}")
    print("-" * 50)
    
    start_time = datetime.now()
    
    try:
        # Create system initialization service
        init_service = SystemInitializationService()
        
        # Run complete system initialization
        result = await init_service.initialize_system()
        
        # Handle sample data creation
        if create_sample_data and settings.environment == Environment.DEVELOPMENT:
            print("\n📊 Creating sample data...")
            try:
                await init_service._initialize_sample_data()
                print("✅ Sample data created successfully")
            except Exception as e:
                print(f"⚠️ Sample data creation failed: {e}")
        
        # Print results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 50)
        print("📋 INITIALIZATION RESULTS")
        print("=" * 50)
        
        print(f"Duration: {duration:.2f} seconds")
        print(f"Overall Success: {'✅ YES' if result['success'] else '❌ NO'}")
        
        print("\nComponent Status:")
        for component, status in result['status'].items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component.replace('_', ' ').title()}")
        
        if result['errors']:
            print(f"\n❌ Errors ({len(result['errors'])}):")
            for error in result['errors']:
                print(f"  - {error}")
        
        if result['warnings']:
            print(f"\n⚠️ Warnings ({len(result['warnings'])}):")
            for warning in result['warnings']:
                print(f"  - {warning}")
        
        if result['success']:
            print(f"\n🎉 System initialization completed successfully!")
            print(f"📊 You can now start the application with: python run_dev.py")
            
            if settings.environment == Environment.DEVELOPMENT:
                print(f"🔗 API Documentation: http://localhost:8000/docs")
                print(f"🔗 Health Check: http://localhost:8000/health")
        else:
            print(f"\n❌ System initialization failed!")
            print(f"Please check the errors above and fix them before starting the application.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ System initialization failed with exception: {e}")
        sys.exit(1)


async def validate_environment():
    """Validate environment configuration only"""
    print("🔧 Validating environment configuration...")
    
    try:
        init_service = SystemInitializationService()
        await init_service._validate_environment_config()
        
        if init_service.initialization_status["environment"]:
            print("✅ Environment configuration is valid")
        else:
            print("❌ Environment configuration has issues:")
            for error in init_service.errors:
                print(f"  - {error}")
            for warning in init_service.warnings:
                print(f"  - {warning}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        sys.exit(1)


async def test_components():
    """Test individual system components"""
    print("🧪 Testing system components...")
    
    try:
        init_service = SystemInitializationService()
        
        # Test database
        print("\n🗄️ Testing database...")
        await init_service._initialize_database()
        print(f"  Database: {'✅ OK' if init_service.initialization_status['database'] else '❌ FAILED'}")
        
        # Test cache
        print("\n💾 Testing cache...")
        await init_service._initialize_cache()
        print(f"  Cache: {'✅ OK' if init_service.initialization_status['cache'] else '❌ FAILED'}")
        
        # Test LLM providers
        print("\n🤖 Testing LLM providers...")
        await init_service._test_llm_providers()
        print(f"  LLM Providers: {'✅ OK' if init_service.initialization_status['llm_providers'] else '❌ FAILED'}")
        
        # Test MCP server
        print("\n🔌 Testing MCP server...")
        await init_service._verify_mcp_server()
        print(f"  MCP Server: {'✅ OK' if init_service.initialization_status['mcp_server'] else '❌ FAILED'}")
        
        # Print summary
        print("\n📊 Component Test Summary:")
        for component, status in init_service.initialization_status.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component.replace('_', ' ').title()}")
        
        if init_service.errors:
            print(f"\n❌ Errors:")
            for error in init_service.errors:
                print(f"  - {error}")
        
        if init_service.warnings:
            print(f"\n⚠️ Warnings:")
            for warning in init_service.warnings:
                print(f"  - {warning}")
                
    except Exception as e:
        print(f"❌ Component testing failed: {e}")
        sys.exit(1)


async def create_env_template():
    """Create .env template file"""
    print("📝 Creating .env template...")
    
    env_template = """# Adaptive Question System Environment Configuration

# App Configuration
APP_NAME=Adaptive Question System
VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/adaptive_question_system
TEST_DATABASE_URL=postgresql://username:password@localhost:5432/adaptive_question_system_test

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Security Configuration
SECRET_KEY=your-secret-key-here-make-it-long-and-random
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
ENCRYPTION_KEY=your-encryption-key-here-make-it-long-and-random

# LLM Configuration
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
PRIMARY_LLM_PROVIDER=gpt4
SECONDARY_LLM_PROVIDER=claude_haiku
DAILY_LLM_BUDGET=100.0

# LLM Fallback Settings
ENABLE_LLM_FALLBACK=true
FALLBACK_TO_TEMPLATES=true
MAX_RETRY_ATTEMPTS=3

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# File Upload Configuration
MAX_FILE_SIZE=10485760
UPLOAD_DIR=./uploads

# Rate Limiting
RATE_LIMIT_ENABLED=true

# Monitoring
PROMETHEUS_ENABLED=true
"""
    
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️ .env file already exists. Skipping template creation.")
    else:
        with open(env_file, "w") as f:
            f.write(env_template)
        print("✅ .env template created successfully")
        print("📝 Please edit .env file with your actual configuration values")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="System initialization script")
    parser.add_argument(
        "--env-only", 
        action="store_true", 
        help="Validate environment configuration only"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Test system components only"
    )
    parser.add_argument(
        "--sample-data", 
        action="store_true", 
        help="Create sample data (development only)"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force initialization even if errors occur"
    )
    parser.add_argument(
        "--create-env", 
        action="store_true", 
        help="Create .env template file"
    )
    
    args = parser.parse_args()
    
    if args.create_env:
        asyncio.run(create_env_template())
        return
    
    if args.env_only:
        asyncio.run(validate_environment())
        return
    
    if args.test:
        asyncio.run(test_components())
        return
    
    # Full system initialization
    asyncio.run(init_system(
        create_sample_data=args.sample_data,
        force=args.force
    ))


if __name__ == "__main__":
    main()
