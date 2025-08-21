from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.database import Base


class MathErrorDetail(Base):
    __tablename__ = "math_error_details"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    attempt_id = Column(UUID(as_uuid=True), ForeignKey("student_attempts.id"), nullable=False, index=True)
    operation = Column(String(50), nullable=True)  # addition, subtraction, multiplication, division
    math_concept = Column(String(100), nullable=True)  # fractions, equations, geometry, etc.
    error_step = Column(Integer, nullable=True)  # Hatanın yapıldığı çözüm adımı
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    attempt = relationship("StudentAttempt", backref="math_error_details")

    def __repr__(self):
        return f"<MathErrorDetail(id={self.id}, attempt_id={self.attempt_id}, operation={self.operation})>"