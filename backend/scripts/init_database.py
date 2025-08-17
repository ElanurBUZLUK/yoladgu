#!/usr/bin/env python3
"""
Database initialization script
Run this to set up the database schema and perform initial setup
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings
from app.core.database import engine, Base
from app.models import user, question, student_attempt, error_pattern, pdf_upload


async def init_database():
    """Initialize database schema"""
    print("🗄️ Initializing database schema...")
    
    try:
        # Create all tables
        async with engine.begin() as conn:
            # Create tables
            await conn.run_sync(Base.metadata.create_all)
            print("✅ Database tables created")
            
            # Create indexes for better performance
            await create_indexes(conn)
            print("✅ Database indexes created")
            
            # Create initial admin user if not exists
            await create_initial_admin(conn)
            print("✅ Initial admin user created")
            
            # Verify schema
            await verify_schema(conn)
            print("✅ Database schema verified")
            
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise


async def create_indexes(conn):
    """Create database indexes for better performance"""
    indexes = [
        # Users table indexes
        "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
        "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
        "CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)",
        "CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at)",
        
        # Questions table indexes
        "CREATE INDEX IF NOT EXISTS idx_questions_subject ON questions(subject)",
        "CREATE INDEX IF NOT EXISTS idx_questions_difficulty ON questions(difficulty_level)",
        "CREATE INDEX IF NOT EXISTS idx_questions_topic ON questions(topic_category)",
        "CREATE INDEX IF NOT EXISTS idx_questions_created_at ON questions(created_at)",
        
        # Student attempts table indexes
        "CREATE INDEX IF NOT EXISTS idx_attempts_user_id ON student_attempts(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_attempts_question_id ON student_attempts(question_id)",
        "CREATE INDEX IF NOT EXISTS idx_attempts_is_correct ON student_attempts(is_correct)",
        "CREATE INDEX IF NOT EXISTS idx_attempts_attempt_date ON student_attempts(attempt_date)",
        "CREATE INDEX IF NOT EXISTS idx_attempts_user_question ON student_attempts(user_id, question_id)",
        
        # Error patterns table indexes
        "CREATE INDEX IF NOT EXISTS idx_error_patterns_user_id ON error_patterns(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_error_patterns_error_type ON error_patterns(error_type)",
        "CREATE INDEX IF NOT EXISTS idx_error_patterns_subject ON error_patterns(subject)",
        "CREATE INDEX IF NOT EXISTS idx_error_patterns_last_occurrence ON error_patterns(last_occurrence)",
        
        # PDF uploads table indexes
        "CREATE INDEX IF NOT EXISTS idx_pdf_uploads_uploaded_by ON pdf_uploads(uploaded_by)",
        "CREATE INDEX IF NOT EXISTS idx_pdf_uploads_subject ON pdf_uploads(subject)",
        "CREATE INDEX IF NOT EXISTS idx_pdf_uploads_processing_status ON pdf_uploads(processing_status)",
        "CREATE INDEX IF NOT EXISTS idx_pdf_uploads_created_at ON pdf_uploads(created_at)"
    ]
    
    for index_sql in indexes:
        try:
            await conn.execute(text(index_sql))
        except Exception as e:
            print(f"Warning: Failed to create index: {e}")


async def create_initial_admin(conn):
    """Create initial admin user if not exists"""
    try:
        # Check if admin user exists
        result = await conn.execute(
            text("SELECT COUNT(*) FROM users WHERE role = 'ADMIN'")
        )
        admin_count = result.scalar()
        
        if admin_count == 0:
            # Create initial admin user
            from app.core.security import security_service
            hashed_password = security_service.get_password_hash("admin123")
            
            admin_user_sql = text("""
                INSERT INTO users (
                    id, username, email, hashed_password, role, 
                    current_math_level, current_english_level, learning_style, 
                    is_active, created_at, updated_at
                ) VALUES (
                    gen_random_uuid(), 'admin', 'admin@system.local', :password, 'ADMIN',
                    1, 1, 'MIXED', 'true', NOW(), NOW()
                )
            """)
            
            await conn.execute(admin_user_sql, {"password": hashed_password})
            print("✅ Initial admin user created (username: admin, password: admin123)")
        else:
            print("ℹ️ Admin user already exists")
            
    except Exception as e:
        print(f"Warning: Failed to create admin user: {e}")


async def verify_schema(conn):
    """Verify database schema is correct"""
    try:
        # Check required tables
        required_tables = [
            "users", "questions", "student_attempts", 
            "error_patterns", "pdf_uploads"
        ]
        
        for table in required_tables:
            result = await conn.execute(
                text("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = :table"),
                {"table": table}
            )
            count = result.scalar()
            if count == 0:
                raise Exception(f"Required table '{table}' not found")
        
        # Check required columns in users table
        required_user_columns = [
            "id", "username", "email", "hashed_password", "role",
            "current_math_level", "current_english_level", "learning_style",
            "is_active", "created_at", "updated_at"
        ]
        
        for column in required_user_columns:
            result = await conn.execute(
                text("""
                    SELECT COUNT(*) FROM information_schema.columns 
                    WHERE table_name = 'users' AND column_name = :column
                """),
                {"column": column}
            )
            count = result.scalar()
            if count == 0:
                raise Exception(f"Required column '{column}' not found in users table")
        
        print("✅ Database schema verification completed")
        
    except Exception as e:
        print(f"❌ Schema verification failed: {e}")
        raise


async def create_sample_data():
    """Create sample data for development"""
    try:
        from app.services.sample_data_service import SampleDataService
        
        sample_service = SampleDataService()
        result = await sample_service.create_sample_data()
        
        if result:
            print("✅ Sample data created successfully")
        else:
            print("⚠️ Sample data creation returned no result")
            
    except Exception as e:
        print(f"⚠️ Sample data creation failed: {e}")


async def main():
    """Main initialization function"""
    print("🚀 Starting database initialization...")
    
    try:
        # Initialize database schema
        await init_database()
        
        # Create sample data for development
        if settings.environment.value == "development":
            print("📊 Creating sample data for development...")
            await create_sample_data()
        
        print("✅ Database initialization completed successfully!")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
