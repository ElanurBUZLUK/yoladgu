from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Float, Enum
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class UserRole(enum.Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.STUDENT)
    grade = Column(String)  # Sınıf bilgisi
    last_login = Column(DateTime, default=datetime.utcnow)

    solutions = relationship("Solution", back_populates="user")
    study_plans = relationship("StudyPlan", back_populates="user")
    responses = relationship("StudentResponse", back_populates="student")
    profile = relationship("StudentProfile", uselist=False, back_populates="student")

class Subject(Base):
    __tablename__ = "subjects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)

    topics = relationship("Topic", back_populates="subject")

class Topic(Base):
    __tablename__ = "topics"
    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("subjects.id"))
    name = Column(String, nullable=False)
    description = Column(Text)

    subject = relationship("Subject", back_populates="topics")
    questions = relationship("Question", back_populates="topic")

class Skill(Base):
    __tablename__ = "skills"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    difficulty_level = Column(Integer, default=1)
    subject_id = Column(Integer, ForeignKey("subjects.id"))

    question_skills = relationship("QuestionSkill", back_populates="skill")

class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("topics.id"))
    text = Column(Text, nullable=False)
    difficulty = Column(String)  # kolay, orta, zor
    correct_answer = Column(String)

    topic = relationship("Topic", back_populates="questions")
    solutions = relationship("Solution", back_populates="question")
    responses = relationship("StudentResponse", back_populates="question")
    question_skills = relationship("QuestionSkill", back_populates="question")

class QuestionSkill(Base):
    __tablename__ = "question_skills"
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"))
    skill_id = Column(Integer, ForeignKey("skills.id"))
    weight = Column(Float, default=1.0)

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

    user = relationship("User", back_populates="solutions")
    question = relationship("Question", back_populates="solutions")

class StudyPlan(Base):
    __tablename__ = "study_plans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    target_success = Column(Float)  # % olarak hedef başarı

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

    plan = relationship("StudyPlan", back_populates="items")
    # subject ve topic ilişkileri istersen eklenebilir 

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

    student = relationship("User", back_populates="responses")
    question = relationship("Question", back_populates="responses")

class StudentProfile(Base):
    __tablename__ = "student_profiles"

    student_id = Column(Integer, ForeignKey("users.id"), primary_key=True, index=True)
    level = Column(Float)
    min_level = Column(Float)
    max_level = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    student = relationship("User", back_populates="profile") 