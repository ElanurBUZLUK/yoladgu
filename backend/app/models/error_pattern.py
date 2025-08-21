from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.database import Base
from app.models.question import Subject


class ErrorPattern(Base):
    __tablename__ = "error_patterns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    subject = Column(SQLEnum(Subject), nullable=False, index=True)
    error_type = Column(String(100), nullable=False, index=True)
    error_count = Column(Integer, default=1, nullable=False)
    last_occurrence = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    topic_category = Column(String(100), nullable=True, index=True)
    difficulty_level = Column(Integer, nullable=True)

    # Relationships
    user = relationship("User", backref="error_patterns")

    def __repr__(self):
        return f"<ErrorPattern(id={self.id}, user_id={self.user_id}, error_type={self.error_type})>"