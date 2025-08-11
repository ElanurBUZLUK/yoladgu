from fastapi import APIRouter, Depends
from app.schemas import VectorQuery, RawSearch
from app.core.deps import get_index_manager
from app.services.embedding_api import get_embedding_provider
from app.services.vector_index_manager import VectorIndexManager
import numpy as np

router = APIRouter(prefix="/vectors", tags=["vectors"])

@router.post("/search")
async def search(q: VectorQuery, mgr: VectorIndexManager = Depends(get_index_manager)):
    prov = get_embedding_provider()
    vec = prov.embed_one(q.text)
    ids, dists = mgr.search(np.array(vec, dtype=np.float32), q.k)
    return {"results": [{"id": i, "dist": float(d)} for i,d in zip(ids,dists)]}

@router.post("/search-raw")
async def search_raw(q: RawSearch, mgr: VectorIndexManager = Depends(get_index_manager)):
    ids, dists = mgr.search(np.array(q.embedding, dtype=np.float32), q.k)
    return {"results": [{"id": i, "dist": float(d)} for i,d in zip(ids,dists)]}
