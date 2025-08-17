from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    VIRUS_SCANNING = "virus_scanning"
    PROCESSING = "processing"
    EXTRACTING_QUESTIONS = "extracting_questions"
    VALIDATING_QUESTIONS = "validating_questions"
    COMPLETED = "completed"
    FAILED = "failed"


class VirusScanStatus(str, Enum):
    PENDING = "pending"
    CLEAN = "clean"
    INFECTED = "infected"
    FAILED = "failed"
    SKIPPED = "skipped"


class Subject(str, Enum):
    MATH = "math"
    ENGLISH = "english"


class PDFUploadRequest(BaseModel):
    subject: Subject = Field(..., description="Subject for the PDF content")
    description: Optional[str] = Field(None, max_length=500, description="Optional description of the PDF content")


class FileValidationResult(BaseModel):
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    file_info: Dict[str, Any] = {}


class PDFUploadResponse(BaseModel):
    upload_id: str
    filename: str
    subject: str
    file_size: int
    status: ProcessingStatus
    virus_scan_status: VirusScanStatus
    message: str
    upload_url: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True


class PDFUploadProgress(BaseModel):
    upload_id: str
    filename: str
    status: ProcessingStatus
    progress_percentage: int = Field(ge=0, le=100)
    questions_extracted: int = Field(ge=0)
    quality_score: Optional[float] = Field(None, ge=0, le=100)
    virus_scan_status: VirusScanStatus
    created_at: str
    subject: str
    error_message: Optional[str] = None
    estimated_completion: Optional[str] = None


class PDFUploadSummary(BaseModel):
    upload_id: str
    filename: str
    subject: str
    status: ProcessingStatus
    questions_extracted: int
    quality_score: Optional[float]
    created_at: str
    file_size: Optional[int] = None


class PDFUploadList(BaseModel):
    uploads: List[PDFUploadSummary]
    total_count: int
    page: int
    page_size: int
    has_next: bool


class PDFProcessingResult(BaseModel):
    upload_id: str
    success: bool
    questions_extracted: int
    quality_score: float
    processing_time_seconds: float
    extracted_questions: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []


class PDFMetadata(BaseModel):
    file_size: int
    page_count: Optional[int] = None
    pdf_version: Optional[str] = None
    encrypted: bool = False
    document_info: Optional[Dict[str, str]] = None
    created_at: str
    modified_at: str
    extraction_method: str = "basic"
    pdf_parsing_error: Optional[str] = None


class PDFUploadStatistics(BaseModel):
    total_uploads: int
    successful_uploads: int
    failed_uploads: int
    pending_uploads: int
    total_questions_extracted: int
    average_quality_score: Optional[float]
    upload_size_stats: Dict[str, Any]
    subject_distribution: Dict[str, int]
    daily_upload_count: int
    monthly_upload_count: int


class PDFSecurityScan(BaseModel):
    scan_result: str
    clean: bool
    message: str
    scan_time: Optional[str] = None
    threat_details: Optional[List[str]] = None


class PDFUploadSettings(BaseModel):
    max_file_size_mb: int = Field(default=50, ge=1, le=200)
    max_daily_uploads: int = Field(default=20, ge=1, le=100)
    allowed_extensions: List[str] = [".pdf"]
    virus_scan_enabled: bool = True
    auto_process: bool = True
    quality_threshold: float = Field(default=0.7, ge=0, le=1)


class PDFUploadError(BaseModel):
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    upload_id: Optional[str] = None


class PDFBatchUploadRequest(BaseModel):
    subject: Subject
    description: Optional[str] = None
    auto_process: bool = True
    notify_on_completion: bool = True


class PDFBatchUploadResponse(BaseModel):
    batch_id: str
    uploaded_files: List[PDFUploadResponse]
    failed_files: List[Dict[str, str]]
    total_files: int
    successful_uploads: int
    failed_uploads: int


class PDFQuestionExtraction(BaseModel):
    question_id: str
    content: str
    question_type: str
    difficulty_level: int
    topic_category: str
    correct_answer: Optional[str] = None
    options: Optional[List[str]] = None
    page_number: int
    confidence_score: float
    extraction_method: str


class PDFProcessingConfig(BaseModel):
    extract_images: bool = True
    preserve_formatting: bool = True
    ocr_enabled: bool = False
    language: str = "tr"  # Turkish by default
    quality_threshold: float = 0.7
    max_questions_per_page: int = 10


class PDFUploadAnalytics(BaseModel):
    user_id: str
    period_days: int
    uploads_count: int
    questions_extracted: int
    average_quality: Optional[float]
    success_rate: float
    most_uploaded_subject: str
    upload_frequency: str  # daily, weekly, monthly
    storage_used_mb: float


# Request/Response models for API endpoints
class UploadProgressRequest(BaseModel):
    upload_id: str


class DeleteUploadRequest(BaseModel):
    upload_id: str
    confirm: bool = Field(default=False, description="Confirmation flag for deletion")


class PDFUploadListRequest(BaseModel):
    subject: Optional[Subject] = None
    status: Optional[ProcessingStatus] = None
    limit: int = Field(default=20, ge=1, le=100)
    skip: int = Field(default=0, ge=0)
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order: asc or desc")

    @validator('sort_by')
    def validate_sort_by(cls, v):
        allowed_fields = ['created_at', 'filename', 'status', 'questions_extracted', 'quality_score']
        if v not in allowed_fields:
            raise ValueError(f'sort_by must be one of: {allowed_fields}')
        return v

    @validator('sort_order')
    def validate_sort_order(cls, v):
        if v.lower() not in ['asc', 'desc']:
            raise ValueError('sort_order must be "asc" or "desc"')
        return v.lower()


class PDFReprocessRequest(BaseModel):
    upload_id: str
    config: Optional[PDFProcessingConfig] = None
    force: bool = Field(default=False, description="Force reprocessing even if already completed")


class PDFDownloadRequest(BaseModel):
    upload_id: str
    include_metadata: bool = Field(default=False)
    format: str = Field(default="original", description="Download format: original, processed, metadata")

    @validator('format')
    def validate_format(cls, v):
        allowed_formats = ['original', 'processed', 'metadata']
        if v not in allowed_formats:
            raise ValueError(f'format must be one of: {allowed_formats}')
        return v