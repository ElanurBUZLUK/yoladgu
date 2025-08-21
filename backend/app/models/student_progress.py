from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.database_enhanced import Base
from app.models.question import Subject

class StudentProgress(Base):
    """Student progress tracking model"""
    
    __tablename__ = "student_progress"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(String, nullable=False, index=True)
    subject = Column(Subject, nullable=False)
    score = Column(Float, nullable=False)  # Percentage score (0-100)
    total_questions = Column(Integer, nullable=False)
    correct_answers = Column(Integer, nullable=False)
    time_spent_seconds = Column(Integer, nullable=False)
    difficulty_level = Column(String, nullable=False)
    completed_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Additional fields for analytics
    topic_category = Column(String, nullable=True)
    error_patterns = Column(Text, nullable=True)  # JSON string of error patterns
    improvement_areas = Column(Text, nullable=True)  # JSON string of areas to improve
    
    def __repr__(self):
        return f"<StudentProgress(id={self.id}, student_id={self.student_id}, subject={self.subject}, score={self.score})>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "student_id": self.student_id,
            "subject": self.subject.value,
            "score": self.score,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "time_spent_seconds": self.time_spent_seconds,
            "difficulty_level": self.difficulty_level,
            "completed_at": self.completed_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "topic_category": self.topic_category,
            "error_patterns": self.error_patterns,
            "improvement_areas": self.improvement_areas
        }
