from fastapi import APIRouter, Depends, HTTPException
from app.core.deps import get_index_manager
from app.services.vector_index_manager import VectorIndexManager

router = APIRouter(prefix="/index", tags=["index-admin"])

@router.get("/stats")
def stats(mgr: VectorIndexManager = Depends(get_index_manager)):
    return mgr.stats()

@router.post("/swap")
def swap(mgr: VectorIndexManager = Depends(get_index_manager)):
    res = mgr.swap()
    if not res.get("swapped"):
        raise HTTPException(400, detail=res.get("reason", "swap failed"))
    return res
