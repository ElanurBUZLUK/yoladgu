from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from app.models.question import Subject


class ErrorPatternBase(BaseModel):
    user_id: UUID
    subject: Subject
    error_type: str = Field(..., min_length=1, max_length=100)
    error_count: Optional[int] = Field(1, ge=0)
    topic_category: Optional[str] = Field(None, max_length=100)
    difficulty_level: Optional[int] = Field(None, ge=1, le=5)


class ErrorPatternCreate(ErrorPatternBase):
    pass


class ErrorPatternUpdate(BaseModel):
    error_count: Optional[int] = Field(None, ge=0)
    last_occurrence: Optional[str] = None # Will be updated by DB
    topic_category: Optional[str] = Field(None, max_length=100)
    difficulty_level: Optional[int] = Field(None, ge=1, le=5)


class ErrorPatternInDB(ErrorPatternBase):
    id: UUID
    last_occurrence: str # Using str for datetime for simplicity in Pydantic

    class Config:
        from_attributes = True

class ClassifiedErrorSchema(BaseModel):
    error_type: str = Field(..., description="Categorized type of error, e.g., 'Grammar: Subject-Verb Agreement', 'Spelling', 'Punctuation: Missing Comma'")
    explanation: Optional[str] = Field(None, description="Brief explanation of the error.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the classification (0.0 to 1.0).")
    suggested_correction: Optional[str] = Field(None, description="Suggested correction for the error.")
    relevant_rule: Optional[str] = Field(None, description="Relevant grammar rule or concept.")
