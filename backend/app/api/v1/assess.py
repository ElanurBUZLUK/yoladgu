from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
import logging
from typing import Dict, Any, List, Optional

from app.services.cefr_assessment_service import cefr_assessment_service
from app.utils.distlock_idem import idempotency_decorator, IdempotencyConfig
import hashlib
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/assess", tags=["cefr-assessment"])

class CefrRequest(BaseModel):
    samples: List[str] = Field(..., description="List of text samples to assess for CEFR level")
    user_id: Optional[str] = Field(None, description="User ID for personalized assessment")
    assessment_type: str = Field("general", description="Type of assessment (general, reading, writing, etc.)")

class AssessRequest(BaseModel):
    # Define fields for AssessRequest based on expected input
    # For now, a placeholder
    text: str = Field(..., description="Text to be assessed for CEFR level")
    # Add other fields as needed for the assessment service

class CEFRAssessmentResponse(BaseModel):
    overall_level: str
    skills: Dict[str, Any]
    confidence: float

def _cefr_assessment_key_builder(*args, **kwargs) -> str:
    """Build idempotency key for CEFR assessment"""
    request = None
    for arg in args:
        if isinstance(arg, CefrRequest):
            request = arg
            break
    
    if not request:
        request = kwargs.get('req')
    
    if not request:
        return "cefr_assessment:default"
    
    # Create deterministic key based on request parameters
    key_data = {
        "user_id": request.user_id,
        "samples": request.samples,
        "assessment_type": request.assessment_type
    }
    
    key_string = json.dumps(key_data, sort_keys=True)
    return f"cefr_assessment:{hashlib.md5(key_string.encode()).hexdigest()}"

@idempotency_decorator(
    key_builder=_cefr_assessment_key_builder,
    config=IdempotencyConfig(scope="cefr_assessment", ttl_seconds=600)
)
async def _assess_cefr_internal(req: CefrRequest, svc = Depends(lambda: cefr_assessment_service)):
    """Internal CEFR assessment function with idempotency"""
    assessment_result = await svc.assess_cefr_level(
        user_id=req.user_id,
        assessment_text="\n".join(req.samples),
        assessment_type=req.assessment_type
    )
    return assessment_result

@router.post("/cefr", response_model=CEFRAssessmentResponse)
async def assess_cefr(req: CefrRequest, svc = Depends(lambda: cefr_assessment_service)):
    """
    Assesses the CEFR level of the provided text samples.
    """
    try:
        return await _assess_cefr_internal(req, svc)
    except Exception as e:
        logger.exception(f"Error in CEFR assessment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CEFR assessment failed: {str(e)}"
        )