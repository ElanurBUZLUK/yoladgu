"""
Database models using SQLModel.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field, Column, JSON, Text, Index
from sqlalchemy import DateTime, func
import uuid


def generate_uuid() -> str:
    """Generate UUID string."""
    return str(uuid.uuid4())


# TimestampMixin removed - timestamps added directly to each model


class User(SQLModel, table=True):
    """User model for students, teachers, and admins."""
    
    __tablename__ = "users"
    
    id: str = Field(default_factory=generate_uuid, primary_key=True)
    tenant_id: str = Field(index=True)
    username: str = Field(unique=True, index=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
    
    # Profile information
    grade: Optional[str] = None
    lang: str = Field(default="tr")
    consent_flag: str = Field(default="none")  # none, analytics, personalization
    
    # Learning state - IRT parameters
    theta_math: Optional[float] = Field(default=0.0)
    theta_en: Optional[float] = Field(default=0.0)
    
    # Error profiles as JSON
    error_profile_math: Optional[Dict[str, float]] = Field(default={}, sa_column=Column(JSON))
    error_profile_en: Optional[Dict[str, float]] = Field(default={}, sa_column=Column(JSON))
    
    # User segments and preferences
    segments: Optional[List[str]] = Field(default=[], sa_column=Column(JSON))
    preferences: Optional[Dict[str, Any]] = Field(default={}, sa_column=Column(JSON))
    
    # Status and role
    is_active: bool = Field(default=True)
    role: str = Field(default="student")  # student, teacher, admin, service
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    )
    
    __table_args__ = (
        Index("idx_users_tenant_role", "tenant_id", "role"),
        Index("idx_users_lang_grade", "lang", "grade"),
    )


class MathItem(SQLModel, table=True):
    """Math question items."""
    
    __tablename__ = "items_math"
    
    id: str = Field(default_factory=generate_uuid, primary_key=True)
    tenant_id: str = Field(index=True)
    
    # Question content
    stem: str = Field(sa_column=Column(Text))
    params: Optional[Dict[str, Any]] = Field(default={}, sa_column=Column(JSON))
    solution: Optional[str] = Field(sa_column=Column(Text))
    answer_key: str
    choices: Optional[List[str]] = Field(default=[], sa_column=Column(JSON))
    
    # Metadata
    skills: List[str] = Field(sa_column=Column(JSON))
    bloom_level: Optional[str] = None  # remember, understand, apply, analyze, evaluate, create
    topic: Optional[str] = None
    
    # IRT parameters
    difficulty_a: float = Field(default=1.0)  # discrimination parameter
    difficulty_b: float = Field(default=0.0)  # difficulty parameter
    
    # Language and source
    lang: str = Field(default="tr")
    source: Optional[str] = None
    generator: Optional[str] = None  # template_id if generated
    
    # Quality and status
    status: str = Field(default="active")  # draft, active, retired
    pedagogy_flags: Optional[Dict[str, Any]] = Field(default={}, sa_column=Column(JSON))
    review_status: Optional[str] = None  # pending, approved, rejected
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    )
    
    __table_args__ = (
        Index("idx_math_items_skills", "skills"),
        Index("idx_math_items_difficulty", "difficulty_b"),
        Index("idx_math_items_lang_status", "lang", "status"),
        Index("idx_math_items_tenant_active", "tenant_id", "status"),
    )


class EnglishItem(SQLModel, table=True):
    """English cloze question items."""
    
    __tablename__ = "items_en"
    
    id: str = Field(default_factory=generate_uuid, primary_key=True)
    tenant_id: str = Field(index=True)
    
    # Question content
    passage: str = Field(sa_column=Column(Text))
    blanks: List[Dict[str, Any]] = Field(sa_column=Column(JSON))  # [{span, skill_tag, answer, distractors, rationale}]
    
    # Metadata
    level_cefr: str  # A1, A2, B1, B2, C1
    topic: Optional[str] = None
    error_tags: List[str] = Field(default=[], sa_column=Column(JSON))  # prepositions, articles, etc.
    
    # Language and source
    lang: str = Field(default="en")
    source: Optional[str] = None
    generator: Optional[str] = None
    
    # Quality and status
    status: str = Field(default="active")  # draft, active, retired
    ambiguity_flag: bool = Field(default=False)
    review_status: Optional[str] = None
    pedagogy_flags: Optional[Dict[str, Any]] = Field(default={}, sa_column=Column(JSON))
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    )
    
    __table_args__ = (
        Index("idx_en_items_cefr", "level_cefr"),
        Index("idx_en_items_error_tags", "error_tags"),
        Index("idx_en_items_lang_status", "lang", "status"),
        Index("idx_en_items_tenant_active", "tenant_id", "status"),
    )


class Attempt(SQLModel, table=True):
    """Student attempts at answering questions."""
    
    __tablename__ = "attempts"
    
    id: str = Field(default_factory=generate_uuid, primary_key=True)
    user_id: str = Field(foreign_key="users.id", index=True)
    item_id: str = Field(index=True)  # Can reference either math or english items
    
    # Attempt data
    answer: str
    correct: bool
    time_ms: Optional[int] = None
    hints_used: int = Field(default=0)
    
    # Context information
    context: Optional[Dict[str, Any]] = Field(default={}, sa_column=Column(JSON))  # device, session_id, etc.
    
    # Metadata
    item_type: str  # math, english
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    )
    
    __table_args__ = (
        Index("idx_attempts_user_time", "user_id", "created_at"),
        Index("idx_attempts_item_correct", "item_id", "correct"),
        Index("idx_attempts_user_item", "user_id", "item_id"),
    )


class RetrievalLog(SQLModel, table=True):
    """Logs for retrieval operations."""
    
    __tablename__ = "retrieval_logs"
    
    id: str = Field(default_factory=generate_uuid, primary_key=True)
    request_id: str = Field(index=True)
    user_id: str = Field(foreign_key="users.id", index=True)
    
    # Query information
    query_repr: Dict[str, Any] = Field(sa_column=Column(JSON))  # skills, level, topic, etc.
    
    # Results
    candidates: List[Dict[str, Any]] = Field(sa_column=Column(JSON))  # doc_id, scores, metadata
    
    # Features and context
    features: Optional[Dict[str, Any]] = Field(default={}, sa_column=Column(JSON))
    
    # Performance metrics
    retrieval_time_ms: Optional[int] = None
    total_candidates: int = Field(default=0)
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    )
    
    __table_args__ = (
        Index("idx_retrieval_logs_request", "request_id"),
        Index("idx_retrieval_logs_user_time", "user_id", "created_at"),
    )


class Decision(SQLModel, table=True):
    """Bandit algorithm decisions."""
    
    __tablename__ = "decisions"
    
    id: str = Field(default_factory=generate_uuid, primary_key=True)
    request_id: str = Field(index=True)
    user_id: str = Field(foreign_key="users.id", index=True)
    
    # Bandit information
    policy_id: str
    bandit_version: str
    
    # Decision data
    arms: List[Dict[str, Any]] = Field(sa_column=Column(JSON))  # arm_id, item_id, propensity, scores
    chosen_arm_id: str
    
    # Context and features
    context_features: Optional[Dict[str, Any]] = Field(default={}, sa_column=Column(JSON))
    
    # Performance metrics
    serving_time_ms: Optional[int] = None
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    )
    
    __table_args__ = (
        Index("idx_decisions_request", "request_id"),
        Index("idx_decisions_user_time", "user_id", "created_at"),
        Index("idx_decisions_policy", "policy_id", "bandit_version"),
    )


class Event(SQLModel, table=True):
    """General event logging."""
    
    __tablename__ = "events"
    
    id: str = Field(default_factory=generate_uuid, primary_key=True)
    request_id: Optional[str] = Field(index=True)
    user_id: Optional[str] = Field(foreign_key="users.id", index=True)
    
    # Event information
    event_type: str  # impression, click, completion, correct, feedback, etc.
    payload: Dict[str, Any] = Field(sa_column=Column(JSON))
    
    # Metadata
    source: Optional[str] = None  # api, batch, etc.
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    )
    
    __table_args__ = (
        Index("idx_events_type_time", "event_type", "created_at"),
        Index("idx_events_user_type", "user_id", "event_type"),
        Index("idx_events_request", "request_id"),
    )


class MetricsDaily(SQLModel, table=True):
    """Daily aggregated metrics."""
    
    __tablename__ = "metrics_daily"
    
    id: str = Field(default_factory=generate_uuid, primary_key=True)
    date: datetime = Field(index=True)
    tenant_id: str = Field(index=True)
    
    # Performance metrics
    p95_latency: Optional[float] = None
    error_rate: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    
    # ML metrics
    faithfulness: Optional[float] = None
    difficulty_match: Optional[float] = None
    coverage: Optional[float] = None
    exploration_ratio: Optional[float] = None
    
    # Business metrics
    total_attempts: int = Field(default=0)
    total_users: int = Field(default=0)
    success_rate: Optional[float] = None
    
    __table_args__ = (
        Index("idx_metrics_daily_date_tenant", "date", "tenant_id"),
    )


class Feedback(SQLModel, table=True):
    """User feedback on questions and system."""
    
    __tablename__ = "feedback"
    
    id: str = Field(default_factory=generate_uuid, primary_key=True)
    user_id: str = Field(foreign_key="users.id", index=True)
    item_id: str = Field(index=True)
    
    # Feedback data
    rating: Optional[int] = None  # 1-5 scale
    flags: Optional[Dict[str, bool]] = Field(default={}, sa_column=Column(JSON))  # too_hard, too_easy, misleading
    comment: Optional[str] = Field(sa_column=Column(Text))
    
    # Metadata
    item_type: str  # math, english
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    )
    
    __table_args__ = (
        Index("idx_feedback_item_rating", "item_id", "rating"),
        Index("idx_feedback_user_time", "user_id", "created_at"),
    )