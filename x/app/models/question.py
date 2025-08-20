from sqlalchemy import Column, String, Integer, Text, DateTime, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid
import enum
from app.core.database import Base


class Subject(str, enum.Enum):
    MATH = "math"
    ENGLISH = "english"


class QuestionType(str, enum.Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    OPEN_ENDED = "open_ended"
    FILL_BLANK = "fill_blank"
    TRUE_FALSE = "true_false"


class SourceType(str, enum.Enum):
    PDF = "pdf"
    GENERATED = "generated"
    MANUAL = "manual"


class DifficultyLevel(int, enum.Enum):
    VERY_EASY = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    VERY_HARD = 5


class Question(Base):
    __tablename__ = "questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject = Column(SQLEnum(Subject), nullable=False, index=True)
    content = Column(Text, nullable=False)
    question_type = Column(SQLEnum(QuestionType), nullable=False)
    difficulty_level = Column(Integer, nullable=False, index=True)
    original_difficulty = Column(Integer, nullable=False)  # Orijinal zorluk seviyesi
    topic_category = Column(String(100), nullable=False, index=True)
    correct_answer = Column(Text, nullable=True)
    options = Column(JSONB, nullable=True)  # Çoktan seçmeli için seçenekler
    source_type = Column(SQLEnum(SourceType), default=SourceType.MANUAL, nullable=False)
    pdf_source_path = Column(String(500), nullable=True)
    question_metadata = Column(JSONB, nullable=True)
    
    # Matematik soru seçim sistemi için ek alanlar
    estimated_difficulty = Column(Integer, nullable=True, index=True)  # Sürekli ölçek (0.0-5.0)
    freshness_score = Column(Integer, nullable=True)  # 0-1; yeni/az görülmüş sorulara +
    last_seen_at = Column(DateTime(timezone=True), nullable=True)  # Son görülme zamanı
    quality_flags = Column(JSONB, nullable=True)  # {ambiguous: False, reviewed: True}
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self):
        return f"<Question(id={self.id}, subject={self.subject}, difficulty={self.difficulty_level})>"