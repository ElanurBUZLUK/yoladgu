from sqlalchemy import Column, Integer, DateTime, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class SpacedRepetition(Base):
    __tablename__ = "spaced_repetition"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), primary_key=True)
    question_id = Column(UUID(as_uuid=True), ForeignKey("questions.id"), primary_key=True)
    next_review_at = Column(DateTime(timezone=True), nullable=False, index=True)
    ease_factor = Column(Float, default=2.5, nullable=False)
    review_count = Column(Integer, default=0, nullable=False)
    last_reviewed = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", backref="spaced_repetitions")
    question = relationship("Question", backref="spaced_repetitions")

    def __repr__(self):
        return f"<SpacedRepetition(user_id={self.user_id}, question_id={self.question_id}, next_review={self.next_review_at})>"