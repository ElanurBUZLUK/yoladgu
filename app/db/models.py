from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Float
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="student")  # student, teacher, admin
    grade = Column(String)  # Sınıf bilgisi
    last_login = Column(DateTime, default=datetime.utcnow)

    solutions = relationship("Solution", back_populates="user")
    study_plans = relationship("StudyPlan", back_populates="user")

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

class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("topics.id"))
    text = Column(Text, nullable=False)
    difficulty = Column(String)  # kolay, orta, zor
    correct_answer = Column(String)

    topic = relationship("Topic", back_populates="questions")
    solutions = relationship("Solution", back_populates="question")

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