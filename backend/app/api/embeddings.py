import re
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.core.database import get_db
from app.services.embedding import EmbeddingService
from app.core.security import verify_token

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"], dependencies=[Depends(verify_token)])

class BatchEmbedRequest(BaseModel):
    texts: list[str]
    namespace: str
    slot: str = "blue"

EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_REGEX = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
PII_PLACEHOLDERS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]"
}

def mask_pii(text: str) -> str:
    text = EMAIL_REGEX.sub(PII_PLACEHOLDERS["email"], text)
    text = PHONE_REGEX.sub(PII_PLACEHOLDERS["phone"], text)
    return text

@router.post("/batch")
async def batch_embed(
    request: BatchEmbedRequest,
    db = Depends(get_db),
    svc: EmbeddingService = Depends()
):
    try:
        masked_texts = [mask_pii(text) for text in request.texts]
        results = await svc.process_batch(
            texts=masked_texts,
            namespace=request.namespace,
            slot=request.slot
        )
        
        return {
            "status": "success",
            "processed": len(results),
            "pii_masked": len(masked_texts),
            "details": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding processing failed: {str(e)}"
        )