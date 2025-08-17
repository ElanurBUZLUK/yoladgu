import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user import User, UserRole, LearningStyle
from app.models.question import Question, Subject, QuestionType, SourceType
from app.models.student_attempt import StudentAttempt
from app.models.error_pattern import ErrorPattern
from app.models.pdf_upload import PDFUpload, ProcessingStatus, VirusScanStatus
from app.core.security import security_service
from app.repositories.user_repository import user_repository
from app.repositories.base_repository import BaseRepository


class SampleDataService:
    """Sample data creation service - veritabanını örnek verilerle doldurur"""
    
    def __init__(self):
        self.user_repo = user_repository
        self.question_repo = BaseRepository(Question)
        self.attempt_repo = BaseRepository(StudentAttempt)
        self.error_repo = BaseRepository(ErrorPattern)
        self.pdf_repo = BaseRepository(PDFUpload)
    
    async def create_sample_data(self, db: AsyncSession) -> Dict[str, Any]:
        """Tüm sample data'yı oluştur"""
        
        print("🚀 Sample data creation başlatılıyor...")
        
        results = {
            "users": [],
            "questions": [],
            "attempts": [],
            "error_patterns": [],
            "pdf_uploads": []
        }
        
        try:
            # Sample users oluştur
            users = await self.create_sample_users(db)
            results["users"] = users
            
            # Sample questions oluştur
            questions = await self.create_sample_questions(db)
            results["questions"] = questions
            
            # Sample student attempts oluştur
            attempts = await self.create_sample_attempts(db, users, questions)
            results["attempts"] = attempts
            
            # Sample error patterns oluştur
            error_patterns = await self.create_sample_error_patterns(db, users, questions)
            results["error_patterns"] = error_patterns
            
            # Sample PDF uploads oluştur
            pdf_uploads = await self.create_sample_pdf_uploads(db, users)
            results["pdf_uploads"] = pdf_uploads
            
            print(f"✅ Sample data creation tamamlandı!")
            print(f"📊 Oluşturulan veriler:")
            print(f"   - Users: {len(users)}")
            print(f"   - Questions: {len(questions)}")
            print(f"   - Attempts: {len(attempts)}")
            print(f"   - Error Patterns: {len(error_patterns)}")
            print(f"   - PDF Uploads: {len(pdf_uploads)}")
            
            return results
            
        except Exception as e:
            print(f"❌ Sample data creation hatası: {str(e)}")
            raise
    
    async def create_sample_users(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Sample users oluştur"""
        
        print("👥 Sample users oluşturuluyor...")
        
        sample_users = [
            # Admin users
            {
                "username": "admin",
                "email": "admin@adaptivelearning.com",
                "password": "admin123",
                "role": UserRole.ADMIN,
                "current_math_level": 5,
                "current_english_level": 5,
                "learning_style": LearningStyle.MIXED
            },
            {
                "username": "superadmin",
                "email": "superadmin@adaptivelearning.com",
                "password": "super123",
                "role": UserRole.ADMIN,
                "current_math_level": 5,
                "current_english_level": 5,
                "learning_style": LearningStyle.VISUAL
            },
            
            # Teacher users
            {
                "username": "teacher_math",
                "email": "math.teacher@school.com",
                "password": "teacher123",
                "role": UserRole.TEACHER,
                "current_math_level": 5,
                "current_english_level": 4,
                "learning_style": LearningStyle.AUDITORY
            },
            {
                "username": "teacher_english",
                "email": "english.teacher@school.com",
                "password": "teacher123",
                "role": UserRole.TEACHER,
                "current_math_level": 3,
                "current_english_level": 5,
                "learning_style": LearningStyle.KINESTHETIC
            },
            {
                "username": "teacher_science",
                "email": "science.teacher@school.com",
                "password": "teacher123",
                "role": UserRole.TEACHER,
                "current_math_level": 4,
                "current_english_level": 4,
                "learning_style": LearningStyle.VISUAL
            },
            
            # Student users - Math focus
            {
                "username": "student_math1",
                "email": "math.student1@school.com",
                "password": "student123",
                "role": UserRole.STUDENT,
                "current_math_level": 3,
                "current_english_level": 2,
                "learning_style": LearningStyle.VISUAL
            },
            {
                "username": "student_math2",
                "email": "math.student2@school.com",
                "password": "student123",
                "role": UserRole.STUDENT,
                "current_math_level": 4,
                "current_english_level": 2,
                "learning_style": LearningStyle.AUDITORY
            },
            {
                "username": "student_math3",
                "email": "math.student3@school.com",
                "password": "student123",
                "role": UserRole.STUDENT,
                "current_math_level": 2,
                "current_english_level": 1,
                "learning_style": LearningStyle.KINESTHETIC
            },
            
            # Student users - English focus
            {
                "username": "student_eng1",
                "email": "english.student1@school.com",
                "password": "student123",
                "role": UserRole.STUDENT,
                "current_math_level": 2,
                "current_english_level": 4,
                "learning_style": LearningStyle.AUDITORY
            },
            {
                "username": "student_eng2",
                "email": "english.student2@school.com",
                "password": "student123",
                "role": UserRole.STUDENT,
                "current_math_level": 1,
                "current_english_level": 3,
                "learning_style": LearningStyle.VISUAL
            },
            {
                "username": "student_eng3",
                "email": "english.student3@school.com",
                "password": "student123",
                "role": UserRole.STUDENT,
                "current_math_level": 2,
                "current_english_level": 5,
                "learning_style": LearningStyle.KINESTHETIC
            },
            
            # Balanced students
            {
                "username": "student_balanced1",
                "email": "balanced.student1@school.com",
                "password": "student123",
                "role": UserRole.STUDENT,
                "current_math_level": 3,
                "current_english_level": 3,
                "learning_style": LearningStyle.MIXED
            },
            {
                "username": "student_balanced2",
                "email": "balanced.student2@school.com",
                "password": "student123",
                "role": UserRole.STUDENT,
                "current_math_level": 4,
                "current_english_level": 4,
                "learning_style": LearningStyle.VISUAL
            }
        ]
        
        created_users = []
        
        for user_data in sample_users:
            # Check if user already exists
            existing_user = await self.user_repo.get_by_email(db, user_data["email"])
            if existing_user:
                print(f"   ⚠️  User already exists: {user_data['email']}")
                created_users.append({
                    "id": str(existing_user.id),
                    "username": existing_user.username,
                    "email": existing_user.email,
                    "role": existing_user.role.value
                })
                continue
            
            # Create new user
            hashed_password = security_service.get_password_hash(user_data["password"])
            
            user = User(
                id=str(uuid.uuid4()),
                username=user_data["username"],
                email=user_data["email"],
                hashed_password=hashed_password,
                role=user_data["role"],
                current_math_level=user_data["current_math_level"],
                current_english_level=user_data["current_english_level"],
                learning_style=user_data["learning_style"],
                is_active="true"
            )
            
            db.add(user)
            await db.commit()
            await db.refresh(user)
            
            created_users.append({
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": user.role.value
            })
            
            print(f"   ✅ User created: {user.username} ({user.role.value})")
        
        return created_users
    
    async def create_sample_questions(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Sample questions oluştur"""
        
        print("❓ Sample questions oluşturuluyor...")
        
        sample_questions = [
            # Math Questions - Level 1
            {
                "subject": Subject.MATH,
                "content": "2 + 3 = ?",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 1,
                "topic_category": "Basic Addition",
                "correct_answer": "5",
                "options": ["3", "4", "5", "6"],
                "source_type": SourceType.MANUAL
            },
            {
                "subject": Subject.MATH,
                "content": "What is 5 - 2?",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 1,
                "topic_category": "Basic Subtraction",
                "correct_answer": "3",
                "options": ["2", "3", "4", "5"],
                "source_type": SourceType.MANUAL
            },
            {
                "subject": Subject.MATH,
                "content": "3 x 4 = ?",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 1,
                "topic_category": "Basic Multiplication",
                "correct_answer": "12",
                "options": ["8", "10", "12", "14"],
                "source_type": SourceType.MANUAL
            },
            
            # Math Questions - Level 2
            {
                "subject": Subject.MATH,
                "content": "Solve: 15 ÷ 3 = ?",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 2,
                "topic_category": "Basic Division",
                "correct_answer": "5",
                "options": ["3", "4", "5", "6"],
                "source_type": SourceType.MANUAL
            },
            {
                "subject": Subject.MATH,
                "content": "What is 25% of 80?",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 2,
                "topic_category": "Percentages",
                "correct_answer": "20",
                "options": ["15", "20", "25", "30"],
                "source_type": SourceType.MANUAL
            },
            {
                "subject": Subject.MATH,
                "content": "Find the area of a rectangle with length 6 and width 4.",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 2,
                "topic_category": "Geometry",
                "correct_answer": "24",
                "options": ["20", "22", "24", "26"],
                "source_type": SourceType.MANUAL
            },
            
            # Math Questions - Level 3
            {
                "subject": Subject.MATH,
                "content": "Solve: 2x + 5 = 13",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 3,
                "topic_category": "Linear Equations",
                "correct_answer": "4",
                "options": ["3", "4", "5", "6"],
                "source_type": SourceType.MANUAL
            },
            {
                "subject": Subject.MATH,
                "content": "What is the square root of 64?",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 3,
                "topic_category": "Square Roots",
                "correct_answer": "8",
                "options": ["6", "7", "8", "9"],
                "source_type": SourceType.MANUAL
            },
            {
                "subject": Subject.MATH,
                "content": "Find the perimeter of a square with side length 7.",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 3,
                "topic_category": "Geometry",
                "correct_answer": "28",
                "options": ["24", "26", "28", "30"],
                "source_type": SourceType.MANUAL
            },
            
            # Math Questions - Level 4
            {
                "subject": Subject.MATH,
                "content": "Solve: x² - 4 = 0",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 4,
                "topic_category": "Quadratic Equations",
                "correct_answer": "±2",
                "options": ["±1", "±2", "±3", "±4"],
                "source_type": SourceType.MANUAL
            },
            {
                "subject": Subject.MATH,
                "content": "What is the slope of the line y = 3x + 2?",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 4,
                "topic_category": "Linear Functions",
                "correct_answer": "3",
                "options": ["2", "3", "4", "5"],
                "source_type": SourceType.MANUAL
            },
            
            # Math Questions - Level 5
            {
                "subject": Subject.MATH,
                "content": "Find the derivative of f(x) = x³ + 2x² - 5x + 1",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 5,
                "topic_category": "Calculus",
                "correct_answer": "3x² + 4x - 5",
                "options": ["3x² + 4x - 5", "3x² + 2x - 5", "x² + 4x - 5", "3x² + 4x"],
                "source_type": SourceType.MANUAL
            },
            
            # English Questions - Level 1
            {
                "subject": Subject.ENGLISH,
                "content": "Choose the correct article: ___ apple is red.",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 1,
                "topic_category": "Articles",
                "correct_answer": "An",
                "options": ["A", "An", "The", "None"],
                "source_type": SourceType.MANUAL
            },
            {
                "subject": Subject.ENGLISH,
                "content": "What is the plural of 'book'?",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 1,
                "topic_category": "Plurals",
                "correct_answer": "Books",
                "options": ["Book", "Books", "Bookes", "Bookies"],
                "source_type": SourceType.MANUAL
            },
            {
                "subject": Subject.ENGLISH,
                "content": "Choose the correct verb: She ___ to school every day.",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 1,
                "topic_category": "Present Simple",
                "correct_answer": "goes",
                "options": ["go", "goes", "going", "went"],
                "source_type": SourceType.MANUAL
            },
            
            # English Questions - Level 2
            {
                "subject": Subject.ENGLISH,
                "content": "Choose the correct past tense: Yesterday, I ___ to the store.",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 2,
                "topic_category": "Past Simple",
                "correct_answer": "went",
                "options": ["go", "goes", "going", "went"],
                "source_type": SourceType.MANUAL
            },
            {
                "subject": Subject.ENGLISH,
                "content": "What is the comparative form of 'big'?",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 2,
                "topic_category": "Comparatives",
                "correct_answer": "bigger",
                "options": ["big", "bigger", "biggest", "more big"],
                "source_type": SourceType.MANUAL
            },
            {
                "subject": Subject.ENGLISH,
                "content": "Choose the correct pronoun: ___ is my friend.",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 2,
                "topic_category": "Pronouns",
                "correct_answer": "He",
                "options": ["Him", "He", "His", "Himself"],
                "source_type": SourceType.MANUAL
            },
            
            # English Questions - Level 3
            {
                "subject": Subject.ENGLISH,
                "content": "Choose the correct conditional: If it rains, I ___ stay home.",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 3,
                "topic_category": "Conditionals",
                "correct_answer": "will",
                "options": ["would", "will", "would have", "had"],
                "source_type": SourceType.MANUAL
            },
            {
                "subject": Subject.ENGLISH,
                "content": "What is the passive voice of 'The cat catches the mouse'?",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 3,
                "topic_category": "Passive Voice",
                "correct_answer": "The mouse is caught by the cat",
                "options": [
                    "The mouse is caught by the cat",
                    "The mouse was caught by the cat",
                    "The mouse catches the cat",
                    "The cat is caught by the mouse"
                ],
                "source_type": SourceType.MANUAL
            },
            
            # English Questions - Level 4
            {
                "subject": Subject.ENGLISH,
                "content": "Choose the correct phrasal verb: The meeting was ___ due to bad weather.",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 4,
                "topic_category": "Phrasal Verbs",
                "correct_answer": "called off",
                "options": ["called off", "called up", "called out", "called in"],
                "source_type": SourceType.MANUAL
            },
            {
                "subject": Subject.ENGLISH,
                "content": "What is the meaning of 'ubiquitous'?",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 4,
                "topic_category": "Advanced Vocabulary",
                "correct_answer": "Present everywhere",
                "options": [
                    "Present everywhere",
                    "Very large",
                    "Extremely small",
                    "Very expensive"
                ],
                "source_type": SourceType.MANUAL
            },
            
            # English Questions - Level 5
            {
                "subject": Subject.ENGLISH,
                "content": "Analyze the literary device in: 'The wind whispered through the trees.'",
                "question_type": QuestionType.MULTIPLE_CHOICE,
                "difficulty_level": 5,
                "topic_category": "Literary Devices",
                "correct_answer": "Personification",
                "options": ["Metaphor", "Simile", "Personification", "Alliteration"],
                "source_type": SourceType.MANUAL
            }
        ]
        
        created_questions = []
        
        for question_data in sample_questions:
            # Check if question already exists
            existing_question = await db.execute(
                select(Question).where(Question.content == question_data["content"])
            )
            existing_question = existing_question.scalar_one_or_none()
            
            if existing_question:
                print(f"   ⚠️  Question already exists: {question_data['content'][:30]}...")
                created_questions.append({
                    "id": str(existing_question.id),
                    "content": existing_question.content,
                    "subject": existing_question.subject.value,
                    "difficulty_level": existing_question.difficulty_level,
                    "correct_answer": existing_question.correct_answer
                })
                continue
            
            # Create new question
            question = Question(
                id=str(uuid.uuid4()),
                subject=question_data["subject"],
                content=question_data["content"],
                question_type=question_data["question_type"],
                difficulty_level=question_data["difficulty_level"],
                topic_category=question_data["topic_category"],
                correct_answer=question_data["correct_answer"],
                options=question_data["options"],
                source_type=question_data["source_type"],
                original_difficulty=question_data["difficulty_level"]
            )
            
            db.add(question)
            await db.commit()
            await db.refresh(question)
            
            created_questions.append({
                "id": str(question.id),
                "content": question.content,
                "subject": question.subject.value,
                "difficulty_level": question.difficulty_level,
                "correct_answer": question.correct_answer
            })
            
            print(f"   ✅ Question created: {question.content[:30]}... (Level {question.difficulty_level})")
        
        return created_questions
    
    async def create_sample_attempts(self, db: AsyncSession, users: List[Dict], questions: List[Dict]) -> List[Dict[str, Any]]:
        """Sample student attempts oluştur"""
        
        print("📝 Sample student attempts oluşturuluyor...")
        
        # Student users'ları filtrele
        student_users = [u for u in users if u["role"] == "student"]
        
        if not student_users:
            print("   ⚠️  No student users found for creating attempts")
            return []
        
        created_attempts = []
        
        for student in student_users:
            # Her öğrenci için 5-15 arası attempt oluştur
            num_attempts = 10  # Her öğrenci için 10 attempt
            
            for i in range(num_attempts):
                # Rastgele bir soru seç
                question = questions[i % len(questions)]
                
                # Rastgele doğru/yanlış cevap (öğrenci seviyesine göre)
                student_level = 3  # Default level
                question_level = question["difficulty_level"]
                
                # Seviye farkına göre doğruluk oranı belirle
                if question_level <= student_level:
                    is_correct = True  # 80% doğru
                else:
                    is_correct = False  # 60% yanlış
                
                # Rastgele zaman (son 30 gün içinde)
                random_days = i % 30
                attempt_time = datetime.utcnow() - timedelta(days=random_days)
                
                attempt = StudentAttempt(
                    id=str(uuid.uuid4()),
                    user_id=student["id"],
                    question_id=question["id"],
                    student_answer=question["correct_answer"] if is_correct else "wrong_answer",
                    is_correct=is_correct,
                    time_spent=30 + (i % 60),  # 30-90 saniye arası
                    attempt_date=attempt_time
                )
                
                db.add(attempt)
                created_attempts.append({
                    "id": str(attempt.id),
                    "user_id": student["id"],
                    "question_id": question["id"],
                    "is_correct": is_correct,
                    "difficulty_level": question_level
                })
        
        await db.commit()
        print(f"   ✅ {len(created_attempts)} attempts created")
        
        return created_attempts
    
    async def create_sample_error_patterns(self, db: AsyncSession, users: List[Dict], questions: List[Dict]) -> List[Dict[str, Any]]:
        """Sample error patterns oluştur"""
        
        print("❌ Sample error patterns oluşturuluyor...")
        
        # Student users'ları filtrele
        student_users = [u for u in users if u["role"] == "student"]
        
        if not student_users:
            print("   ⚠️  No student users found for creating error patterns")
            return []
        
        error_patterns = [
            {
                "pattern_type": "calculation_error",
                "description": "Basic arithmetic mistakes",
                "common_mistakes": ["Addition errors", "Subtraction errors", "Multiplication errors"]
            },
            {
                "pattern_type": "concept_misunderstanding",
                "description": "Fundamental concept confusion",
                "common_mistakes": ["Wrong formula application", "Concept confusion", "Misinterpretation"]
            },
            {
                "pattern_type": "careless_mistakes",
                "description": "Simple oversight errors",
                "common_mistakes": ["Sign errors", "Decimal placement", "Unit conversion"]
            },
            {
                "pattern_type": "grammar_error",
                "description": "English grammar mistakes",
                "common_mistakes": ["Verb tense errors", "Article mistakes", "Pronoun confusion"]
            },
            {
                "pattern_type": "vocabulary_error",
                "description": "Vocabulary and word choice errors",
                "common_mistakes": ["Wrong word choice", "Spelling errors", "Context misunderstanding"]
            }
        ]
        
        created_patterns = []
        
        for student in student_users:
            for pattern_data in error_patterns:
                # Check if pattern already exists for this user
                existing_pattern = await db.execute(
                    select(ErrorPattern).where(
                        ErrorPattern.user_id == student["id"],
                        ErrorPattern.error_type == pattern_data["pattern_type"]
                    )
                )
                existing_pattern = existing_pattern.scalar_one_or_none()
                
                if existing_pattern:
                    # Update existing pattern instead of skipping
                    existing_pattern.error_count += 1
                    existing_pattern.last_occurrence = datetime.utcnow()
                    created_patterns.append({
                        "id": str(existing_pattern.id),
                        "user_id": student["id"],
                        "pattern_type": pattern_data["pattern_type"],
                        "frequency": existing_pattern.error_count
                    })
                    continue
                
                pattern = ErrorPattern(
                    id=str(uuid.uuid4()),
                    user_id=student["id"],
                    subject=Subject.MATH,  # Default subject
                    error_type=pattern_data["pattern_type"],
                    error_count=5 + (len(created_patterns) % 10),  # 5-15 arası
                    last_occurrence=datetime.utcnow() - timedelta(days=len(created_patterns) % 7),
                    topic_category="General",
                    difficulty_level=3
                )
                
                db.add(pattern)
                created_patterns.append({
                    "id": str(pattern.id),
                    "user_id": student["id"],
                    "pattern_type": pattern_data["pattern_type"],
                    "frequency": pattern.error_count
                })
        
        await db.commit()
        print(f"   ✅ {len(created_patterns)} error patterns created")
        
        return created_patterns
    
    async def create_sample_pdf_uploads(self, db: AsyncSession, users: List[Dict]) -> List[Dict[str, Any]]:
        """Sample PDF uploads oluştur"""
        
        print("📄 Sample PDF uploads oluşturuluyor...")
        
        # Teacher users'ları filtrele
        teacher_users = [u for u in users if u["role"] == "teacher"]
        
        if not teacher_users:
            print("   ⚠️  No teacher users found for creating PDF uploads")
            return []
        
        sample_pdfs = [
            {
                "filename": "math_basics.pdf",
                "subject": Subject.MATH,
                "description": "Basic mathematics concepts and exercises",
                "file_size": 2048576,  # 2MB
                "status": "completed"
            },
            {
                "filename": "english_grammar.pdf",
                "subject": Subject.ENGLISH,
                "description": "English grammar rules and practice",
                "file_size": 1536000,  # 1.5MB
                "status": "completed"
            },
            {
                "filename": "advanced_math.pdf",
                "subject": Subject.MATH,
                "description": "Advanced mathematics topics",
                "file_size": 3145728,  # 3MB
                "status": "processing"
            },
            {
                "filename": "literature_analysis.pdf",
                "subject": Subject.ENGLISH,
                "description": "Literature analysis and interpretation",
                "file_size": 2621440,  # 2.5MB
                "status": "completed"
            }
        ]
        
        created_uploads = []
        
        for i, pdf_data in enumerate(sample_pdfs):
            teacher = teacher_users[i % len(teacher_users)]
            
            upload = PDFUpload(
                id=uuid.uuid4(),
                uploaded_by=uuid.UUID(teacher["id"]),
                filename=pdf_data["filename"],
                subject=pdf_data["subject"],
                file_path=f"/uploads/{pdf_data['filename']}",
                questions_extracted=15 if pdf_data["status"] == "completed" else 0,
                processing_status=ProcessingStatus.COMPLETED if pdf_data["status"] == "completed" else ProcessingStatus.PENDING,
                quality_score=0.8 + (i % 20) / 100,
                virus_scan_status=VirusScanStatus.CLEAN
            )
            
            db.add(upload)
            created_uploads.append({
                "id": str(upload.id),
                "filename": upload.filename,
                "subject": upload.subject.value,
                "processing_status": upload.processing_status.value,
                "uploaded_by": teacher["id"]
            })
        
        await db.commit()
        print(f"   ✅ {len(created_uploads)} PDF uploads created")
        
        return created_uploads
    
    async def clear_sample_data(self, db: AsyncSession) -> Dict[str, int]:
        """Sample data'yı temizle"""
        
        print("🧹 Sample data temizleniyor...")
        
        deleted_counts = {}
        
        # Delete in reverse order to avoid foreign key constraints
        # First get all IDs for each entity
        error_patterns = await db.execute(select(ErrorPattern))
        error_pattern_ids = [ep.id for ep in error_patterns.scalars().all()]
        deleted_counts["error_patterns"] = await self.error_repo.bulk_delete(db, error_pattern_ids) if error_pattern_ids else 0
        
        attempts = await db.execute(select(StudentAttempt))
        attempt_ids = [a.id for a in attempts.scalars().all()]
        deleted_counts["student_attempts"] = await self.attempt_repo.bulk_delete(db, attempt_ids) if attempt_ids else 0
        
        questions = await db.execute(select(Question))
        question_ids = [q.id for q in questions.scalars().all()]
        deleted_counts["questions"] = await self.question_repo.bulk_delete(db, question_ids) if question_ids else 0
        
        pdf_uploads = await db.execute(select(PDFUpload))
        pdf_upload_ids = [p.id for p in pdf_uploads.scalars().all()]
        deleted_counts["pdf_uploads"] = await self.pdf_repo.bulk_delete(db, pdf_upload_ids) if pdf_upload_ids else 0
        
        # Delete users (except admin)
        admin_users = await db.execute(
            select(User).where(User.role == UserRole.ADMIN)
        )
        admin_users = admin_users.scalars().all()
        
        all_users = await db.execute(select(User))
        all_users = all_users.scalars().all()
        
        non_admin_users = [u for u in all_users if u.role != UserRole.ADMIN]
        deleted_counts["users"] = len(non_admin_users)
        
        for user in non_admin_users:
            await db.delete(user)
        
        await db.commit()
        
        print(f"✅ Sample data cleared:")
        for entity, count in deleted_counts.items():
            print(f"   - {entity}: {count} deleted")
        
        return deleted_counts


# Global sample data service instance
sample_data_service = SampleDataService()
