#!/usr/bin/env python3
"""
Test script for Task 11.1: Sample Data Creation
Tests sample data creation functionality
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.sample_data_service import sample_data_service
from app.core.database import get_async_session
from app.models.user import User, UserRole, LearningStyle
from app.models.question import Question, Subject, QuestionType
from app.models.student_attempt import StudentAttempt
from app.models.error_pattern import ErrorPattern
from app.models.pdf_upload import PDFUpload
from sqlalchemy import select, func


async def test_sample_data_service():
    """Test sample data service functionality"""
    print("Testing Sample Data Service...")
    
    # Get database session
    async for db in get_async_session():
        try:
            # Test service initialization
            assert sample_data_service is not None
            assert hasattr(sample_data_service, 'create_sample_data')
            assert hasattr(sample_data_service, 'create_sample_users')
            assert hasattr(sample_data_service, 'create_sample_questions')
            print("✅ Sample data service initialization test passed")
            
            # Test sample users creation
            users = await sample_data_service.create_sample_users(db)
            assert isinstance(users, list)
            assert len(users) > 0
            
            # Check user types
            user_roles = [user["role"] for user in users]
            assert "admin" in user_roles
            assert "teacher" in user_roles
            assert "student" in user_roles
            print("✅ Sample users creation test passed")
            
            # Test sample questions creation
            questions = await sample_data_service.create_sample_questions(db)
            assert isinstance(questions, list)
            assert len(questions) > 0
            
            # Check question subjects
            question_subjects = [q["subject"] for q in questions]
            assert "math" in question_subjects
            assert "english" in question_subjects
            print("✅ Sample questions creation test passed")
            
            # Test sample attempts creation
            attempts = await sample_data_service.create_sample_attempts(db, users, questions)
            assert isinstance(attempts, list)
            assert len(attempts) > 0
            print("✅ Sample attempts creation test passed")
            
            # Test sample error patterns creation
            error_patterns = await sample_data_service.create_sample_error_patterns(db, users, questions)
            assert isinstance(error_patterns, list)
            assert len(error_patterns) > 0
            print("✅ Sample error patterns creation test passed")
            
            # Test sample PDF uploads creation
            pdf_uploads = await sample_data_service.create_sample_pdf_uploads(db, users)
            assert isinstance(pdf_uploads, list)
            assert len(pdf_uploads) > 0
            print("✅ Sample PDF uploads creation test passed")
            
            # Test complete sample data creation
            results = await sample_data_service.create_sample_data(db)
            assert isinstance(results, dict)
            assert "users" in results
            assert "questions" in results
            assert "attempts" in results
            assert "error_patterns" in results
            assert "pdf_uploads" in results
            print("✅ Complete sample data creation test passed")
            
            # Test data clearing
            deleted_counts = await sample_data_service.clear_sample_data(db)
            assert isinstance(deleted_counts, dict)
            assert "users" in deleted_counts
            assert "questions" in deleted_counts
            assert "student_attempts" in deleted_counts
            assert "error_patterns" in deleted_counts
            assert "pdf_uploads" in deleted_counts
            print("✅ Sample data clearing test passed")
            
            break
            
        except Exception as e:
            print(f"❌ Sample data service test failed: {str(e)}")
            raise


async def test_sample_users_data():
    """Test sample users data quality"""
    print("Testing Sample Users Data Quality...")
    
    async for db in get_async_session():
        try:
            # Create sample users
            users = await sample_data_service.create_sample_users(db)
            
            # Test user data structure
            for user in users:
                assert "id" in user
                assert "username" in user
                assert "email" in user
                assert "role" in user
                assert user["role"] in ["admin", "teacher", "student"]
            
            # Test admin users
            admin_users = [u for u in users if u["role"] == "admin"]
            assert len(admin_users) >= 2  # At least 2 admin users
            
            # Test teacher users
            teacher_users = [u for u in users if u["role"] == "teacher"]
            assert len(teacher_users) >= 3  # At least 3 teacher users
            
            # Test student users
            student_users = [u for u in users if u["role"] == "student"]
            assert len(student_users) >= 8  # At least 8 student users
            
            # Test email format
            for user in users:
                assert "@" in user["email"]
                assert "." in user["email"]
            
            print("✅ Sample users data quality test passed")
            
            # Clean up
            await sample_data_service.clear_sample_data(db)
            break
            
        except Exception as e:
            print(f"❌ Sample users data quality test failed: {str(e)}")
            raise


async def test_sample_questions_data():
    """Test sample questions data quality"""
    print("Testing Sample Questions Data Quality...")
    
    async for db in get_async_session():
        try:
            # Create sample questions
            questions = await sample_data_service.create_sample_questions(db)
            
            # Test question data structure
            for question in questions:
                assert "id" in question
                assert "content" in question
                assert "subject" in question
                assert "difficulty_level" in question
                assert question["subject"] in ["math", "english"]
                assert 1 <= question["difficulty_level"] <= 5
            
            # Test math questions
            math_questions = [q for q in questions if q["subject"] == "math"]
            assert len(math_questions) >= 10  # At least 10 math questions
            
            # Test English questions
            english_questions = [q for q in questions if q["subject"] == "english"]
            assert len(english_questions) >= 10  # At least 10 English questions
            
            # Test difficulty levels
            difficulty_levels = [q["difficulty_level"] for q in questions]
            assert 1 in difficulty_levels  # Level 1 questions exist
            assert 5 in difficulty_levels  # Level 5 questions exist
            
            # Test question content quality
            for question in questions:
                assert len(question["content"]) > 5  # Content is not too short
                assert len(question["content"]) >= 8  # Has reasonable content length
            
            print("✅ Sample questions data quality test passed")
            
            # Clean up
            await sample_data_service.clear_sample_data(db)
            break
            
        except Exception as e:
            print(f"❌ Sample questions data quality test failed: {str(e)}")
            raise


async def test_sample_attempts_data():
    """Test sample attempts data quality"""
    print("Testing Sample Attempts Data Quality...")
    
    async for db in get_async_session():
        try:
            # Create sample users and questions first
            users = await sample_data_service.create_sample_users(db)
            questions = await sample_data_service.create_sample_questions(db)
            
            # Create sample attempts
            attempts = await sample_data_service.create_sample_attempts(db, users, questions)
            
            # Test attempts data structure
            for attempt in attempts:
                assert "id" in attempt
                assert "user_id" in attempt
                assert "question_id" in attempt
                assert "is_correct" in attempt
                assert "difficulty_level" in attempt
                assert isinstance(attempt["is_correct"], bool)
                assert 1 <= attempt["difficulty_level"] <= 5
            
            # Test attempts distribution
            student_users = [u for u in users if u["role"] == "student"]
            expected_attempts = len(student_users) * 10  # 10 attempts per student
            assert len(attempts) >= expected_attempts
            
            # Test correct/incorrect distribution
            correct_attempts = [a for a in attempts if a["is_correct"]]
            incorrect_attempts = [a for a in attempts if not a["is_correct"]]
            assert len(correct_attempts) > 0
            assert len(incorrect_attempts) > 0
            
            print("✅ Sample attempts data quality test passed")
            
            # Clean up
            await sample_data_service.clear_sample_data(db)
            break
            
        except Exception as e:
            print(f"❌ Sample attempts data quality test failed: {str(e)}")
            raise


async def test_sample_error_patterns_data():
    """Test sample error patterns data quality"""
    print("Testing Sample Error Patterns Data Quality...")
    
    async for db in get_async_session():
        try:
            # Create sample users first
            users = await sample_data_service.create_sample_users(db)
            
            # Create sample error patterns
            error_patterns = await sample_data_service.create_sample_error_patterns(db, users, [])
            
            # Test error patterns data structure
            for pattern in error_patterns:
                assert "id" in pattern
                assert "user_id" in pattern
                assert "pattern_type" in pattern
                assert "frequency" in pattern
                assert pattern["frequency"] >= 5
            
            # Test pattern types
            pattern_types = [p["pattern_type"] for p in error_patterns]
            expected_types = ["calculation_error", "concept_misunderstanding", "careless_mistakes", 
                            "grammar_error", "vocabulary_error"]
            
            for expected_type in expected_types:
                assert expected_type in pattern_types
            
            # Test frequency distribution
            frequencies = [p["frequency"] for p in error_patterns]
            assert max(frequencies) <= 15  # Max frequency should be reasonable
            
            print("✅ Sample error patterns data quality test passed")
            
            # Clean up
            await sample_data_service.clear_sample_data(db)
            break
            
        except Exception as e:
            print(f"❌ Sample error patterns data quality test failed: {str(e)}")
            raise


async def test_sample_pdf_uploads_data():
    """Test sample PDF uploads data quality"""
    print("Testing Sample PDF Uploads Data Quality...")
    
    async for db in get_async_session():
        try:
            # Create sample users first
            users = await sample_data_service.create_sample_users(db)
            
            # Create sample PDF uploads
            pdf_uploads = await sample_data_service.create_sample_pdf_uploads(db, users)
            
            # Test PDF uploads data structure
            for upload in pdf_uploads:
                assert "id" in upload
                assert "filename" in upload
                assert "subject" in upload
                assert "processing_status" in upload
                assert "uploaded_by" in upload
                assert upload["subject"] in ["math", "english"]
                assert upload["processing_status"] in ["completed", "pending"]
            
            # Test file types
            filenames = [u["filename"] for u in pdf_uploads]
            assert "math_basics.pdf" in filenames
            assert "english_grammar.pdf" in filenames
            assert "advanced_math.pdf" in filenames
            assert "literature_analysis.pdf" in filenames
            
            # Test subjects
            math_uploads = [u for u in pdf_uploads if u["subject"] == "math"]
            english_uploads = [u for u in pdf_uploads if u["subject"] == "english"]
            assert len(math_uploads) >= 2
            assert len(english_uploads) >= 2
            
            print("✅ Sample PDF uploads data quality test passed")
            
            # Clean up
            await sample_data_service.clear_sample_data(db)
            break
            
        except Exception as e:
            print(f"❌ Sample PDF uploads data quality test failed: {str(e)}")
            raise


async def test_database_integration():
    """Test database integration with sample data"""
    print("Testing Database Integration...")
    
    async for db in get_async_session():
        try:
            # Create complete sample data
            results = await sample_data_service.create_sample_data(db)
            
            # Verify data in database
            user_count = await db.execute(select(func.count(User.id)))
            user_count = user_count.scalar()
            assert user_count > 0
            
            question_count = await db.execute(select(func.count(Question.id)))
            question_count = question_count.scalar()
            assert question_count > 0
            
            attempt_count = await db.execute(select(func.count(StudentAttempt.id)))
            attempt_count = attempt_count.scalar()
            assert attempt_count > 0
            
            error_pattern_count = await db.execute(select(func.count(ErrorPattern.id)))
            error_pattern_count = error_pattern_count.scalar()
            assert error_pattern_count > 0
            
            pdf_upload_count = await db.execute(select(func.count(PDFUpload.id)))
            pdf_upload_count = pdf_upload_count.scalar()
            assert pdf_upload_count > 0
            
            # Test relationships
            users = await db.execute(select(User))
            users = users.scalars().all()
            
            for user in users:
                if user.role == UserRole.STUDENT:
                    # Check if student has attempts
                    attempts = await db.execute(
                        select(StudentAttempt).where(StudentAttempt.user_id == user.id)
                    )
                    attempts = attempts.scalars().all()
                    assert len(attempts) > 0
                    
                    # Check if student has error patterns
                    patterns = await db.execute(
                        select(ErrorPattern).where(ErrorPattern.user_id == user.id)
                    )
                    patterns = patterns.scalars().all()
                    assert len(patterns) > 0
                
                elif user.role == UserRole.TEACHER:
                    # Check if teacher has PDF uploads
                    uploads = await db.execute(
                        select(PDFUpload).where(PDFUpload.uploaded_by == user.id)
                    )
                    uploads = uploads.scalars().all()
                    assert len(uploads) > 0
            
            print("✅ Database integration test passed")
            
            # Clean up
            await sample_data_service.clear_sample_data(db)
            break
            
        except Exception as e:
            print(f"❌ Database integration test failed: {str(e)}")
            raise


async def test_sample_data_consistency():
    """Test sample data consistency and relationships"""
    print("Testing Sample Data Consistency...")
    
    async for db in get_async_session():
        try:
            # Create sample data
            results = await sample_data_service.create_sample_data(db)
            
            # Test user-role consistency
            users = await db.execute(select(User))
            users = users.scalars().all()
            
            for user in users:
                # Test learning style consistency
                assert user.learning_style in [LearningStyle.VISUAL, LearningStyle.AUDITORY, 
                                             LearningStyle.KINESTHETIC, LearningStyle.MIXED]
                
                # Test level consistency
                assert 1 <= user.current_math_level <= 5
                assert 1 <= user.current_english_level <= 5
            
            # Test question consistency
            questions = await db.execute(select(Question))
            questions = questions.scalars().all()
            
            for question in questions:
                # Test difficulty consistency
                assert 1 <= question.difficulty_level <= 5
                assert question.difficulty_level == question.original_difficulty
                
                # Test subject consistency
                assert question.subject in [Subject.MATH, Subject.ENGLISH]
                
                # Test question type consistency
                assert question.question_type == QuestionType.MULTIPLE_CHOICE
            
            # Test attempt consistency
            attempts = await db.execute(select(StudentAttempt))
            attempts = attempts.scalars().all()
            
            for attempt in attempts:
                # Test time spent consistency
                assert attempt.time_spent > 0
                assert attempt.time_spent <= 300  # Max 5 minutes
                
                # Test difficulty consistency
                assert 1 <= attempt.question.difficulty_level <= 5
            
            print("✅ Sample data consistency test passed")
            
            # Clean up
            await sample_data_service.clear_sample_data(db)
            break
            
        except Exception as e:
            print(f"❌ Sample data consistency test failed: {str(e)}")
            raise


async def main():
    """Main test function"""
    print("🚀 Starting Task 11.1 Tests: Sample Data Creation")
    print("=" * 60)
    
    try:
        # Test sample data service
        await test_sample_data_service()
        print()
        
        # Test sample users data quality
        await test_sample_users_data()
        print()
        
        # Test sample questions data quality
        await test_sample_questions_data()
        print()
        
        # Test sample attempts data quality
        await test_sample_attempts_data()
        print()
        
        # Test sample error patterns data quality
        await test_sample_error_patterns_data()
        print()
        
        # Test sample PDF uploads data quality
        await test_sample_pdf_uploads_data()
        print()
        
        # Test database integration
        await test_database_integration()
        print()
        
        # Test data consistency
        await test_sample_data_consistency()
        print()
        
        print("🎉 All Task 11.1 tests passed successfully!")
        print("✅ Sample data service implemented")
        print("✅ Sample users creation working")
        print("✅ Sample questions creation working")
        print("✅ Sample attempts creation working")
        print("✅ Sample error patterns creation working")
        print("✅ Sample PDF uploads creation working")
        print("✅ Database integration verified")
        print("✅ Data consistency validated")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
