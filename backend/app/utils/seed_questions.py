"""
Database seeding utility for sample questions
"""

import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import AsyncSessionLocal
from app.models.question import Question
from app.data.sample_math_questions import SAMPLE_MATH_QUESTIONS
from app.data.sample_english_questions import SAMPLE_ENGLISH_QUESTIONS
import uuid


async def seed_sample_questions():
    """Seed the database with sample math and English questions"""
    
    async with AsyncSessionLocal() as db:
        try:
            print("🌱 Seeding sample questions...")
            
            # Check if questions already exist
            from sqlalchemy import select, func
            result = await db.execute(select(func.count(Question.id)))
            existing_count = result.scalar()
            
            if existing_count > 0:
                print(f"   Database already has {existing_count} questions. Skipping seed.")
                return
            
            # Create math questions
            questions_created = 0
            
            print("   Creating math questions...")
            for question_data in SAMPLE_MATH_QUESTIONS:
                db_question = Question(
                    id=uuid.uuid4(),
                    subject=question_data["subject"],
                    content=question_data["content"],
                    question_type=question_data["question_type"],
                    difficulty_level=question_data["difficulty_level"],
                    original_difficulty=question_data["difficulty_level"],
                    topic_category=question_data["topic_category"],
                    correct_answer=question_data["correct_answer"],
                    options=question_data.get("options"),
                    source_type=question_data["source_type"],
                    pdf_source_path=question_data.get("pdf_source_path"),
                    question_metadata=question_data.get("question_metadata", {})
                )
                
                db.add(db_question)
                questions_created += 1
            
            print("   Creating English questions...")
            for question_data in SAMPLE_ENGLISH_QUESTIONS:
                db_question = Question(
                    id=uuid.uuid4(),
                    subject=question_data["subject"],
                    content=question_data["content"],
                    question_type=question_data["question_type"],
                    difficulty_level=question_data["difficulty_level"],
                    original_difficulty=question_data["difficulty_level"],
                    topic_category=question_data["topic_category"],
                    correct_answer=question_data["correct_answer"],
                    options=question_data.get("options"),
                    source_type=question_data["source_type"],
                    pdf_source_path=question_data.get("pdf_source_path"),
                    question_metadata=question_data.get("question_metadata", {})
                )
                
                db.add(db_question)
                questions_created += 1
            
            await db.commit()
            print(f"✅ Successfully created {questions_created} sample questions")
            
            # Print summary by subject and difficulty level
            from sqlalchemy import select
            from app.models.question import Subject
            
            for subject in [Subject.MATH, Subject.ENGLISH]:
                subject_result = await db.execute(
                    select(func.count(Question.id)).where(Question.subject == subject)
                )
                subject_count = subject_result.scalar()
                print(f"   {subject.value.title()}: {subject_count} questions")
                
                for level in range(1, 6):
                    level_result = await db.execute(
                        select(func.count(Question.id)).where(
                            Question.subject == subject,
                            Question.difficulty_level == level
                        )
                    )
                    level_count = level_result.scalar()
                    print(f"     Level {level}: {level_count} questions")
            
        except Exception as e:
            await db.rollback()
            print(f"❌ Error seeding questions: {e}")
            raise


async def clear_all_questions():
    """Clear all questions from database (for testing)"""
    
    async with AsyncSessionLocal() as db:
        try:
            from sqlalchemy import delete
            await db.execute(delete(Question))
            await db.commit()
            print("🗑️  All questions cleared from database")
        except Exception as e:
            await db.rollback()
            print(f"❌ Error clearing questions: {e}")
            raise


if __name__ == "__main__":
    # Run seeding
    asyncio.run(seed_sample_questions())