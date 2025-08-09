from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.core.deps import get_index_manager, require_roles
from app.services.vector_index_manager import VectorIndexManager

router = APIRouter(prefix="/index", tags=["index-admin"])

@router.get("/stats")
def stats(
    mgr: VectorIndexManager = Depends(get_index_manager),
    user=Depends(require_roles("admin")),
):
    return mgr.stats()

@router.post("/swap")
def swap(
    mgr: VectorIndexManager = Depends(get_index_manager),
    user=Depends(require_roles("admin")),
):
    res = mgr.swap()
    if not res.get("swapped"):
        raise HTTPException(400, detail=res.get("reason", "swap failed"))
    return res

@router.get("/tombstones")
def tombstones_list(
    mgr: VectorIndexManager = Depends(get_index_manager),
    user=Depends(require_roles("admin")),
):
    return mgr.list_tombstones()

@router.post("/tombstones")
def tombstones_add(
    ids: List[int],
    mgr: VectorIndexManager = Depends(get_index_manager),
    user=Depends(require_roles("admin")),
):
    return mgr.add_tombstone(ids)

@router.delete("/tombstones")
def tombstones_remove(
    ids: List[int],
    mgr: VectorIndexManager = Depends(get_index_manager),
    user=Depends(require_roles("admin")),
):
    return mgr.remove_tombstone(ids)
