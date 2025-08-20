from pydantic import BaseModel, Field
from typing import Dict, Optional, Any

class CEFRSkillScores(BaseModel):
    grammar: str = Field(..., description="CEFR level for grammar skill (e.g., A1, B2)")
    vocab: str = Field(..., description="CEFR level for vocabulary skill (e.g., A1, B2)")
    reading: str = Field(..., description="CEFR level for reading skill (e.g., A1, B2)")
    # Add other skills as needed, e.g., writing, listening, speaking
    writing: Optional[str] = Field(None, description="CEFR level for writing skill (e.g., A1, B2)")
    listening: Optional[str] = Field(None, description="CEFR level for listening skill (e.g., A1, B2)")
    speaking: Optional[str] = Field(None, description="CEFR level for speaking skill (e.g., A1, B2)")

class CEFRAssessmentResponse(BaseModel):
    overall_level: str = Field(..., description="Overall CEFR level (e.g., B1)")
    skills: CEFRSkillScores = Field(..., description="Detailed CEFR levels for individual skills")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the assessment (0.0 to 1.0)")
    assessment_details: Optional[Dict[str, Any]] = Field(None, description="Additional details about the assessment")

class CEFRAssessmentRequest(BaseModel):
    user_id: str = Field(..., description="ID of the user being assessed")
    assessment_text: str = Field(..., description="Text provided by the user for assessment (e.g., writing sample)")
    assessment_type: str = Field("writing_sample", description="Type of assessment (e.g., 'writing_sample', 'speaking_transcript')")
