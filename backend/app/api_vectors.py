from fastapi import APIRouter, Depends
from app.schemas import VectorQuery, RawSearch
from app.core.deps import get_embedding_service, get_index_manager
from app.services.embedding_service import EmbeddingService
from app.services.vector_index_manager import VectorIndexManager
import numpy as np

router = APIRouter(prefix="/vectors", tags=["vectors"])

@router.post("/search")
async def search(q: VectorQuery, emb: EmbeddingService = Depends(get_embedding_service), mgr: VectorIndexManager = Depends(get_index_manager)):
    vec = await emb.compute(q.text)
    ids, dists = mgr.search(np.array(vec, dtype=np.float32), q.k)
    return {"results": [{"id": i, "dist": float(d)} for i,d in zip(ids,dists)]}

@router.post("/search-raw")
async def search_raw(q: RawSearch, mgr: VectorIndexManager = Depends(get_index_manager)):
    ids, dists = mgr.search(np.array(q.embedding, dtype=np.float32), q.k)
    return {"results": [{"id": i, "dist": float(d)} for i,d in zip(ids,dists)]}
