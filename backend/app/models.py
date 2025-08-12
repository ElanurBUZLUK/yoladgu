from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Boolean, Text, ForeignKey, DateTime, Numeric, SmallInteger, Float, JSON
from sqlalchemy.sql import func
from datetime import datetime
from app.core.db import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    full_name: Mapped[str | None] = mapped_column(String, nullable=True)
    role: Mapped[str] = mapped_column(String, default="student")
    is_email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    # Adaptive: skill
    skill_rating: Mapped[float] = mapped_column(Float, default=0.0)
    skill_rating_var: Mapped[float | None] = mapped_column(Float, nullable=True)

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    token: Mapped[str] = mapped_column(Text)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=False))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())

class TeacherStudentRequest(Base):
    __tablename__ = "teacher_student_requests"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    teacher_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())

class Assignment(Base):
    __tablename__ = "assignments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    teacher_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    title: Mapped[str] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    due_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    topic: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())

class Submission(Base):
    __tablename__ = "submissions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    assignment_id: Mapped[int] = mapped_column(ForeignKey("assignments.id", ondelete="CASCADE"))
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())

class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    event_type: Mapped[str] = mapped_column(String)
    payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())


class Question(Base):
    __tablename__ = "questions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    difficulty_rating: Mapped[float] = mapped_column(Float, default=0.0)
    difficulty_level: Mapped[int] = mapped_column(SmallInteger, default=2)
    t_ref_ms: Mapped[int] = mapped_column(Integer, default=10000)
    last_recalibrated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)


class Attempt(Base):
    __tablename__ = "attempts"
    attempt_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id", ondelete="CASCADE"))
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False)
    time_ms: Mapped[int] = mapped_column(Integer)
    reward_time_adj: Mapped[float] = mapped_column(Float)
    expected: Mapped[float] = mapped_column(Float)
    delta: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())


# --- Dynamic Ensemble / A-B infra ---

class EnsembleWeights(Base):
    __tablename__ = "ensemble_weights"
    variant: Mapped[str] = mapped_column(String, primary_key=True)
    # JSON column to store weight dictionary, e.g. {"cf":0.25,"bandit":0.35,"online":0.40,"retr":0.0,"peer":0.0}
    weights: Mapped[dict] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())


class ABAssignment(Base):
    __tablename__ = "ab_assignments"
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    variant: Mapped[str] = mapped_column(ForeignKey("ensemble_weights.variant", ondelete="CASCADE"))
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())


class PolicyVariant(Base):
    __tablename__ = "policy_variants"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    parameters: Mapped[dict] = mapped_column(JSON)
    assignment_rule: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), server_default=func.now())
