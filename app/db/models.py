from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Float, Enum, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.db.database import Base

class UserRole(enum.Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    grade = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.STUDENT)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    solutions = relationship("Solution", back_populates="user")
    study_plans = relationship("StudyPlan", back_populates="user")
    responses = relationship("StudentResponse", back_populates="student")
    profile = relationship("StudentProfile", uselist=False, back_populates="student")

class Subject(Base):
    __tablename__ = "subjects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    topics = relationship("Topic", back_populates="subject")
    skills = relationship("Skill", back_populates="subject")

class Topic(Base):
    __tablename__ = "topics"
    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("subjects.id"))
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    subject = relationship("Subject", back_populates="topics")
    questions = relationship("Question", back_populates="topic")

class Skill(Base):
    __tablename__ = "skills"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    difficulty_level = Column(Integer, default=1)
    subject_id = Column(Integer, ForeignKey("subjects.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    question_skills = relationship("QuestionSkill", back_populates="skill")
    subject = relationship("Subject", back_populates="skills")

class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)  # Ana soru metni
    question_type = Column(String, default="multiple_choice")  # multiple_choice, true_false, open_ended
    difficulty_level = Column(Integer, default=1)  # 1-5 arası zorluk
    subject_id = Column(Integer, ForeignKey("subjects.id"))
    topic_id = Column(Integer, ForeignKey("topics.id"))
    options = Column(JSON)  # Şık seçenekleri
    correct_answer = Column(String, nullable=False)
    explanation = Column(Text)  # Çözüm açıklaması
    tags = Column(JSON)  # Etiketler
    created_by = Column(Integer, ForeignKey("users.id"))
    is_active = Column(Boolean, default=True)
    bert_sim = Column(JSON)  # Embedding vektörü (eski alan)
    embedding = Column(Text)  # Vector embedding - JSON formatında saklanıyor  
    embedding_vector = Column(Text)  # pgvector integration için
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    topic = relationship("Topic", back_populates="questions")
    subject = relationship("Subject")
    creator = relationship("User")
    solutions = relationship("Solution", back_populates="question")
    responses = relationship("StudentResponse", back_populates="question")
    question_skills = relationship("QuestionSkill", back_populates="question")

class QuestionSkill(Base):
    __tablename__ = "question_skills"
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"))
    skill_id = Column(Integer, ForeignKey("skills.id"))
    weight = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    question = relationship("Question", back_populates="question_skills")
    skill = relationship("Skill", back_populates="question_skills")

class Solution(Base):
    __tablename__ = "solutions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    question_id = Column(Integer, ForeignKey("questions.id"))
    solved_at = Column(DateTime, default=datetime.utcnow)
    is_correct = Column(Boolean)
    duration = Column(Float)  # saniye cinsinden
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="solutions")
    question = relationship("Question", back_populates="solutions")

class StudyPlan(Base):
    __tablename__ = "study_plans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    target_success = Column(Float)  # % olarak hedef başarı
    is_active = Column(Boolean, default=True)

    user = relationship("User", back_populates="study_plans")
    items = relationship("PlanItem", back_populates="plan")

class PlanItem(Base):
    __tablename__ = "plan_items"
    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("study_plans.id"))
    subject_id = Column(Integer, ForeignKey("subjects.id"))
    topic_id = Column(Integer, ForeignKey("topics.id"))
    question_count = Column(Integer)
    difficulty = Column(String)
    estimated_time = Column(Integer)  # dakika cinsinden
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    plan = relationship("StudyPlan", back_populates="items")

class StudentResponse(Base):
    __tablename__ = "student_responses"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"))
    question_id = Column(Integer, ForeignKey("questions.id"))
    answer = Column(String, nullable=False)
    is_correct = Column(Boolean, nullable=False)
    response_time = Column(Float)
    confidence_level = Column(Integer)
    feedback = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    student = relationship("User", back_populates="responses")
    question = relationship("Question", back_populates="responses")

class StudentProfile(Base):
    __tablename__ = "student_profiles"

    student_id = Column(Integer, ForeignKey("users.id"), primary_key=True, index=True)
    level = Column(Float, default=1.0)
    min_level = Column(Float, default=1.0)
    max_level = Column(Float, default=5.0)
    total_questions_answered = Column(Integer, default=0)
    total_correct_answers = Column(Integer, default=0)
    average_response_time = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    student = relationship("User", back_populates="profile")

class QuizSession(Base):
    __tablename__ = "quiz_sessions"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    total_questions = Column(Integer, nullable=False)
    correct_answers = Column(Integer, nullable=False)
    wrong_answers = Column(Integer, nullable=False)
    accuracy_percentage = Column(Float, nullable=False)
    total_time_seconds = Column(Integer, nullable=False)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    student = relationship("User") 