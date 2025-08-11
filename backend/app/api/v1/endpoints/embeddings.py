from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from app.services.embedding_api import get_embedding_provider


router = APIRouter(tags=["embeddings"])


class EmbedIn(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=128)


class EmbedOut(BaseModel):
    vectors: List[List[float]]
    dim: int


@router.post("/embeddings", response_model=EmbedOut)
async def embed(body: EmbedIn):
    prov = get_embedding_provider()
    try:
        vecs = prov.embed_batch(body.texts)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding provider error: {e}")
    dim = len(vecs[0]) if vecs else 0
    return {"vectors": vecs, "dim": dim}


