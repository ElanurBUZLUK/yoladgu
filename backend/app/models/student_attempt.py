from sqlalchemy import Column, String, Integer, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


class StudentAttempt(Base):
    __tablename__ = "student_attempts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    question_id = Column(UUID(as_uuid=True), ForeignKey("questions.id"), nullable=False, index=True)
    student_answer = Column(Text, nullable=True)
    is_correct = Column(Boolean, nullable=False)
    time_spent = Column(Integer, nullable=True)  # saniye cinsinden
    attempt_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    error_category = Column(String(100), nullable=True)
    grammar_errors = Column(JSONB, nullable=True)  # İngilizce için grammar hataları
    vocabulary_errors = Column(JSONB, nullable=True)  # İngilizce için vocabulary hataları

    # Relationships
    user = relationship("User", backref="attempts")
    question = relationship("Question", backref="attempts")

    def __repr__(self):
        return f"<StudentAttempt(id={self.id}, user_id={self.user_id}, is_correct={self.is_correct})>"