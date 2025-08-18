#!/usr/bin/env python3
"""
System setup script for Adaptive Learning Platform
"""
import asyncio
import os
import sys
import logging
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import database
from app.services.vector_index_manager import vector_index_manager
from app.services.sample_data_service import sample_data_service
from app.core.config import settings

logger = logging.getLogger(__name__)

async def setup_database():
    """Setup database and run migrations"""
    print("🗄️  Setting up database...")
    
    try:
        # Test database connection
        await database.connect()
        print("✅ Database connection successful")
        
        # Run migrations (this would be done via alembic)
        print("📋 Running database migrations...")
        # Note: In production, this would be: alembic upgrade head
        
        await database.disconnect()
        print("✅ Database setup completed")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

async def setup_vector_indexes():
    """Setup vector indexes for similarity search"""
    print("🔍 Setting up vector indexes...")
    
    try:
        # Check if pgvector extension is available
        print("🔧 Checking pgvector extension...")
        pgvector_available = await vector_index_manager._check_pgvector_extension()
        
        if not pgvector_available:
            print("❌ pgvector extension not available")
            print("Please install pgvector extension:")
            print("  Ubuntu/Debian: sudo apt-get install postgresql-13-pgvector")
            print("  CentOS/RHEL: sudo yum install pgvector_13")
            print("  macOS: brew install pgvector")
            print("  Or run: ./scripts/install_pgvector.sh")
            return False
        
        print("✅ pgvector extension is available")
        
        # Run Alembic migrations
        print("📋 Running Alembic migrations...")
        import subprocess
        try:
            result = subprocess.run(
                ["alembic", "upgrade", "head"],
                capture_output=True,
                text=True,
                cwd="."
            )
            
            if result.returncode == 0:
                print("✅ Alembic migrations completed successfully")
            else:
                print(f"❌ Alembic migration failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("❌ alembic command not found")
            print("Please install alembic: pip install alembic")
            return False
        
        # Verify indexes exist
        success = await vector_index_manager.create_vector_indexes()
        
        if success:
            print("✅ Vector indexes verified successfully")
            
            # Get statistics
            stats = await vector_index_manager.get_index_statistics()
            print(f"📊 Index statistics: {stats}")
            
            # Verify improved indexes
            questions_index = await vector_index_manager._check_index_exists("ix_questions_content_embedding_cosine")
            errors_index = await vector_index_manager._check_index_exists("ix_error_patterns_embedding_cosine")
            embeddings_index = await vector_index_manager._check_index_exists("ix_embeddings_embedding_cosine")
            
            print(f"   - Questions cosine index: {'✅' if questions_index else '❌'}")
            print(f"   - Error patterns cosine index: {'✅' if errors_index else '❌'}")
            print(f"   - Embeddings cosine index: {'✅' if embeddings_index else '❌'}")
            
            return True
        else:
            print("❌ Vector index verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Vector index setup failed: {e}")
        return False

async def setup_sample_data():
    """Setup sample data for testing"""
    print("📝 Setting up sample data...")
    
    try:
        # Connect to database
        await database.connect()
        
        # Create sample data
        results = await sample_data_service.create_sample_data(database)
        
        print(f"✅ Sample data created:")
        print(f"   - Users: {len(results.get('users', []))}")
        print(f"   - Questions: {len(results.get('questions', []))}")
        print(f"   - Attempts: {len(results.get('attempts', []))}")
        print(f"   - Error Patterns: {len(results.get('error_patterns', []))}")
        print(f"   - PDF Uploads: {len(results.get('pdf_uploads', []))}")
        
        await database.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Sample data setup failed: {e}")
        return False

async def setup_embeddings():
    """Setup embeddings for existing data"""
    print("🧠 Setting up embeddings...")
    
    try:
        # Connect to database
        await database.connect()
        
        # Update embeddings in batches
        batch_results = await vector_index_manager.batch_update_embeddings(database, batch_size=50)
        
        print(f"✅ Embeddings updated:")
        print(f"   - Questions: {batch_results.get('questions_updated', 0)}")
        print(f"   - Error Patterns: {batch_results.get('patterns_updated', 0)}")
        print(f"   - Total: {batch_results.get('total_updated', 0)}")
        
        await database.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Embeddings setup failed: {e}")
        return False

async def health_check():
    """Perform system health check"""
    print("🏥 Performing health check...")
    
    try:
        # Connect to database
        await database.connect()
        
        # Check vector index health
        vector_health = await vector_index_manager.health_check()
        
        print(f"✅ Health check results:")
        print(f"   - Vector system status: {vector_health.get('status', 'unknown')}")
        print(f"   - PgVector available: {vector_health.get('pgvector_available', False)}")
        print(f"   - Indexes created: {vector_health.get('indexes_created', {})}")
        
        await database.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

async def main():
    """Main setup function"""
    print("🚀 Starting Adaptive Learning Platform setup...")
    print("=" * 50)
    
    # Check environment variables
    print("🔧 Checking environment configuration...")
    required_vars = [
        'DATABASE_URL',
        'SECRET_KEY',
        'REDIS_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please check your .env file")
        return False
    
    print("✅ Environment configuration OK")
    
    # Setup steps
    steps = [
        ("Database", setup_database),
        ("Vector Indexes", setup_vector_indexes),
        ("Sample Data", setup_sample_data),
        ("Embeddings", setup_embeddings),
        ("Health Check", health_check)
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        try:
            success = await step_func()
            results[step_name] = success
            if not success:
                print(f"⚠️  {step_name} step failed, but continuing...")
        except Exception as e:
            print(f"❌ {step_name} step error: {e}")
            results[step_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("📋 Setup Summary:")
    print("="*50)
    
    all_success = True
    for step_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {step_name}: {status}")
        if not success:
            all_success = False
    
    if all_success:
        print("\n🎉 All setup steps completed successfully!")
        print("You can now start the application with: python run_dev.py")
    else:
        print("\n⚠️  Some setup steps failed. Please check the errors above.")
        print("You may need to fix the issues before starting the application.")
    
    return all_success

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run setup
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
