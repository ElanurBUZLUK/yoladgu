"""
Enhanced Embedding API Endpoints
SBERT tabanlı semantic similarity ve embedding yönetimi için API
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Optional
import structlog

from app.services.enhanced_embedding_service import enhanced_embedding_service
from app.core.config import settings
from pydantic import BaseModel
from typing import Dict, Any

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

# === VECTOR STORE ENDPOINTS ===

class VectorSearchRequest(BaseModel):
    query: str
    k: int = 10
    similarity_threshold: float = 0.7
    filters: Optional[Dict[str, Any]] = None

class StoreQuestionRequest(BaseModel):
    question_id: int
    question_text: str
    metadata: Optional[Dict[str, Any]] = None
    subject_id: Optional[int] = None
    topic_id: Optional[int] = None
    difficulty_level: Optional[int] = None

@router.post("/vector/search")
async def vector_semantic_search(request: VectorSearchRequest):
    """
    Vector Database üzerinden O(log N) hızında semantic arama
    Büyük soru havuzlarında çok daha hızlı sonuçlar verir
    """
    try:
        results = await enhanced_embedding_service.semantic_search_vector_db(
            query_text=request.query,
            k=request.k,
            similarity_threshold=request.similarity_threshold,
            filters=request.filters
        )
        
        return {
            "query": request.query,
            "results": results,
            "total_found": len(results),
            "search_params": {
                "k": request.k,
                "similarity_threshold": request.similarity_threshold,
                "filters": request.filters
            }
        }
        
    except Exception as e:
        logger.error("vector_search_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vector/store")
async def store_question_embedding(request: StoreQuestionRequest):
    """
    Yeni soru embedding'ini vector store'a kaydet
    Yeni sorular eklendiğinde hemen aranabilir hale gelir
    """
    try:
        success = await enhanced_embedding_service.store_question_embedding(
            question_id=request.question_id,
            question_text=request.question_text,
            metadata=request.metadata,
            subject_id=request.subject_id,
            topic_id=request.topic_id,
            difficulty_level=request.difficulty_level
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store embedding")
        
        return {
            "message": "Question embedding stored successfully",
            "question_id": request.question_id,
            "stored": success
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("store_embedding_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vector/stats")
async def get_vector_store_stats():
    """Vector store performans istatistikleri"""
    try:
        stats = await enhanced_embedding_service.get_vector_store_stats()
        
        return {
            "vector_store_stats": stats,
            "timestamp": enhanced_embedding_service.stats
        }
        
    except Exception as e:
        logger.error("vector_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vector/initialize")
async def initialize_vector_store():
    """Vector store'u başlat (sistem kurulumu için)"""
    try:
        await enhanced_embedding_service.initialize_vector_store()
        
        return {
            "message": "Vector store initialized successfully",
            "status": "ready"
        }
        
    except Exception as e:
        logger.error("vector_init_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vector/batch-store")
async def batch_store_embeddings(batch_size: int = Query(100, ge=1, le=500)):
    """
    Mevcut tüm soruları vector store'a toplu kaydetme
    İlk kurulum veya büyük veri güncellemeleri için
    """
    try:
        stats = await enhanced_embedding_service.batch_update_embeddings(
            batch_size=batch_size,
            force_recompute=False
        )
        
        return {
            "message": "Batch vector store update completed",
            "stats": stats,
            "batch_size": batch_size
        }
        
    except Exception as e:
        logger.error("batch_vector_store_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))