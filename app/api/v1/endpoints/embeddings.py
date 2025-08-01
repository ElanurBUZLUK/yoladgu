"""
Enhanced Embedding API Endpoints
SBERT tabanlı semantic similarity ve embedding yönetimi için API
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Optional
import structlog

from app.services.enhanced_embedding_service import enhanced_embedding_service
from app.core.config import settings

logger = structlog.get_logger()

router = APIRouter()

@router.post("/compute")
async def compute_embedding(text: str):
    """Tek metin için embedding hesapla"""
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        embedding = enhanced_embedding_service.compute_embedding_cached(text)
        
        return {
            "text": text,
            "embedding": embedding,
            "dimensions": len(embedding),
            "model": enhanced_embedding_service.current_model_key
        }
        
    except Exception as e:
        logger.error("compute_embedding_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/similarity") 
async def semantic_similarity(request: dict):
    """İki metin arasındaki semantic benzerlik"""
    try:
        text1 = request.get("text1")
        text2 = request.get("text2")
        
        if not text1 or not text2:
            raise HTTPException(status_code=400, detail="Both texts are required")
        
        similarity = enhanced_embedding_service.semantic_similarity(text1, text2)
        
        return {
            "text1": text1,
            "text2": text2,
            "similarity": similarity,
            "model": enhanced_embedding_service.current_model_key
        }
        
    except Exception as e:
        logger.error("semantic_similarity_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def semantic_search(
    query: str,
    question_pool: List[Dict],
    top_k: int = Query(10, ge=1, le=50),
    similarity_threshold: float = Query(0.6, ge=0.0, le=1.0)
):
    """Semantic arama yap"""
    try:
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not question_pool:
            raise HTTPException(status_code=400, detail="Question pool cannot be empty")
        
        results = enhanced_embedding_service.semantic_search(
            query, question_pool, top_k, similarity_threshold
        )
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "search_params": {
                "top_k": top_k,
                "similarity_threshold": similarity_threshold
            }
        }
        
    except Exception as e:
        logger.error("semantic_search_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clustering")
async def semantic_clustering(
    texts: List[str],
    n_clusters: int = Query(5, ge=2, le=20)
):
    """Metinleri semantic olarak kümelere ayır"""
    try:
        if not texts or len(texts) < 2:
            raise HTTPException(status_code=400, detail="At least 2 texts required")
        
        if len(texts) < n_clusters:
            raise HTTPException(
                status_code=400, 
                detail=f"Number of texts ({len(texts)}) must be >= n_clusters ({n_clusters})"
            )
        
        clustering_result = enhanced_embedding_service.semantic_clustering(texts, n_clusters)
        
        return {
            "input_texts": texts,
            "n_clusters": n_clusters,
            "clustering_result": clustering_result,
            "model": enhanced_embedding_service.current_model_key
        }
        
    except Exception as e:
        logger.error("semantic_clustering_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/outliers")
async def find_outliers(
    texts: List[str],
    threshold: float = Query(0.3, ge=0.0, le=1.0)
):
    """Semantic outlier'ları bul"""
    try:
        if not texts or len(texts) < 3:
            raise HTTPException(status_code=400, detail="At least 3 texts required")
        
        outliers = enhanced_embedding_service.find_semantic_outliers(texts, threshold)
        
        return {
            "input_texts": texts,
            "threshold": threshold,
            "outliers": outliers,
            "outlier_count": len(outliers)
        }
        
    except Exception as e:
        logger.error("find_outliers_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_available_models():
    """Kullanılabilir SBERT modellerini listele"""
    try:
        return {
            "current_model": enhanced_embedding_service.current_model_key,
            "available_models": enhanced_embedding_service.available_models,
            "model_info": {
                "name": enhanced_embedding_service.model_name,
                "dimensions": enhanced_embedding_service.embedding_dim
            }
        }
        
    except Exception as e:
        logger.error("get_models_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/switch")
async def switch_model(model_key: str):
    """SBERT modelini değiştir"""
    try:
        if model_key not in enhanced_embedding_service.available_models:
            available = list(enhanced_embedding_service.available_models.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model key. Available: {available}"
            )
        
        success = enhanced_embedding_service.switch_model(model_key)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to switch model")
        
        return {
            "message": "Model switched successfully",
            "new_model": model_key,
            "model_info": enhanced_embedding_service.available_models[model_key]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("switch_model_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_embedding_stats():
    """Embedding service istatistikleri"""
    try:
        stats = enhanced_embedding_service.get_enhanced_stats()
        return stats
        
    except Exception as e:
        logger.error("get_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/clear")
async def clear_cache():
    """Embedding cache'ini temizle"""
    try:
        success = enhanced_embedding_service.clear_cache()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
        
        return {
            "message": "Cache cleared successfully",
            "model": enhanced_embedding_service.current_model_key
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("clear_cache_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/update")
async def batch_update_embeddings(batch_size: int = Query(50, ge=1, le=200)):
    """Toplu embedding güncelleme"""
    try:
        result = await enhanced_embedding_service.batch_update_embeddings(batch_size)
        
        return {
            "message": "Batch update completed",
            "result": result,
            "batch_size": batch_size
        }
        
    except Exception as e:
        logger.error("batch_update_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))