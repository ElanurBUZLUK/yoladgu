from sqlalchemy import Column, String, Integer, DateTime, Float, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum
from app.core.database import Base
from app.models.question import Subject


class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VirusScanStatus(str, enum.Enum):
    PENDING = "pending"
    CLEAN = "clean"
    INFECTED = "infected"
    FAILED = "failed"


class PDFUpload(Base):
    __tablename__ = "pdf_uploads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    subject = Column(SQLEnum(Subject), nullable=False, index=True)
    questions_extracted = Column(Integer, default=0, nullable=False)
    processing_status = Column(SQLEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False)
    quality_score = Column(Float, nullable=True)  # PDF kalite puanı
    virus_scan_status = Column(SQLEnum(VirusScanStatus), default=VirusScanStatus.PENDING, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    uploader = relationship("User", backref="pdf_uploads")

    def __repr__(self):
        return f"<PDFUpload(id={self.id}, filename={self.filename}, status={self.processing_status})>"