"""
Embeddings Endpoints
Embedding işlemleri endpointleri
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
import structlog
from app.services.enhanced_embedding_service import enhanced_embedding_service
from app.services.vector_store_service import vector_store_service
from app.schemas.vector import (
    EmbeddingRequest, EmbeddingResponse,
    EmbeddingComputeRequest, EmbeddingComputeResponse,
    BatchEmbeddingRequest, BatchEmbeddingResponse,
    SemanticSimilarityRequest, SemanticSimilarityResponse,
    SemanticClusteringRequest, SemanticClusteringResponse,
    OutlierDetectionRequest, OutlierDetectionResponse
)

logger = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

# New batch processing models
class BatchEmbeddingWithMetadataRequest(BaseModel):
    texts_with_metadata: List[Dict[str, Any]]

class BatchSimilarityAnalysisRequest(BaseModel):
    texts: List[str]
    similarity_threshold: float = 0.8

class BatchQualityCheckRequest(BaseModel):
    texts: List[str]

class BatchOptimizationRequest(BaseModel):
    texts: List[str]
    target_dimensions: Optional[int] = None

class ComputeAndStoreRequest(BaseModel):
    """Embedding hesaplama ve kaydetme isteği"""
    text: str = Field(..., description="Hesaplanacak metin")
    question_id: int = Field(..., description="Soru ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Ek metadata")

@router.post("/compute", response_model=EmbeddingResponse)
async def compute_embedding_with_cache(request: EmbeddingRequest):
    """Embedding hesaplama ve cache işlemleri"""
    try:
        # 1. Embedding hesaplama (cache varsa EnhancedEmbeddingService.compute_embedding_cached)
        embedding = await enhanced_embedding_service.compute_embedding_cached(request.text)
        
        # 2. Cache durumunu kontrol et (Redis'te emb:query:{hash} olarak saklanır)
        cache_key = enhanced_embedding_service.get_embedding_cache_key(request.text)
        cached = enhanced_embedding_service.redis.exists(cache_key)
        
        # 3. DB'ye Upsert (VectorStoreService.upsert)
        # Note: This would require question_id and metadata from the request
        # For now, we'll just return the embedding without upserting
        
        return EmbeddingResponse(
            embedding=embedding,
            dim=len(embedding),
            cached=bool(cached)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding computation failed: {str(e)}")


@router.post("/compute-and-store", response_model=EmbeddingResponse)
async def compute_embedding_and_store(request: ComputeAndStoreRequest):
    """Embedding hesaplama, cache ve DB'ye kaydetme"""
    try:
        # 1. Embedding hesaplama (cache varsa EnhancedEmbeddingService.compute_embedding_cached)
        embedding = await enhanced_embedding_service.compute_embedding_cached(request.text)
        
        # 2. Cache durumunu kontrol et (Redis'te emb:query:{hash} olarak saklanır)
        cache_key = enhanced_embedding_service.get_embedding_cache_key(request.text)
        cached = enhanced_embedding_service.redis.exists(cache_key)
        
        # 3. DB'ye Upsert (VectorStoreService.upsert)
        upsert_success = await vector_store_service.upsert(
            question_id=request.question_id,
            embedding=embedding,
            metadata=request.metadata
        )
        
        if not upsert_success:
            logger.warning("upsert_failed", question_id=request.question_id)
        
        return EmbeddingResponse(
            embedding=embedding,
            dim=len(embedding),
            cached=bool(cached)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding computation and storage failed: {str(e)}")


@router.post("/batch/compute", response_model=List[List[float]])
async def compute_embeddings_batch(request: BatchEmbeddingRequest):
    """Çoklu metin için embedding hesapla"""
    try:
        embeddings = await enhanced_embedding_service.compute_embeddings_batch_cached_async(
            request.texts
        )
        return embeddings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch embedding computation failed: {str(e)}")

@router.post("/batch/compute-with-metadata", response_model=List[Dict[str, Any]])
async def compute_embeddings_batch_with_metadata(request: BatchEmbeddingWithMetadataRequest):
    """Metadata ile birlikte çoklu embedding hesapla"""
    try:
        results = await enhanced_embedding_service.compute_embeddings_batch_with_metadata(
            request.texts_with_metadata
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch embedding with metadata failed: {str(e)}")

@router.post("/batch/similarity-analysis", response_model=Dict[str, Any])
async def batch_similarity_analysis(request: BatchSimilarityAnalysisRequest):
    """Çoklu metin için gelişmiş benzerlik analizi"""
    try:
        analysis = await enhanced_embedding_service.batch_similarity_analysis(
            request.texts, request.similarity_threshold
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch similarity analysis failed: {str(e)}")

@router.post("/batch/quality-check", response_model=Dict[str, Any])
async def batch_embedding_quality_check(request: BatchQualityCheckRequest):
    """Çoklu embedding için kalite kontrolü"""
    try:
        quality_metrics = await enhanced_embedding_service.batch_embedding_quality_check(
            request.texts
        )
        return quality_metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch quality check failed: {str(e)}")

@router.post("/batch/optimize", response_model=Dict[str, Any])
async def batch_embedding_optimization(request: BatchOptimizationRequest):
    """Çoklu embedding için optimizasyon"""
    try:
        optimization_result = await enhanced_embedding_service.batch_embedding_optimization(
            request.texts, request.target_dimensions
        )
        return optimization_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch optimization failed: {str(e)}")

@router.post("/compute", response_model=EmbeddingComputeResponse)
async def compute_embedding(request: EmbeddingComputeRequest):
    """Tek metin için embedding hesapla"""
    try:
        embedding = await enhanced_embedding_service.compute_embedding_cached(request.text)
        return EmbeddingComputeResponse(
            text=request.text,
            embedding=embedding,
            dimensions=len(embedding),
            model=enhanced_embedding_service.model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding computation failed: {str(e)}")

@router.post("/similarity", response_model=SemanticSimilarityResponse)
async def compute_semantic_similarity(request: SemanticSimilarityRequest):
    """İki metin arasındaki semantik benzerliği hesapla"""
    try:
        similarity = enhanced_embedding_service.semantic_similarity(
            request.text1, request.text2
        )
        return SemanticSimilarityResponse(
            text1=request.text1,
            text2=request.text2,
            similarity=similarity,
            model=enhanced_embedding_service.model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity computation failed: {str(e)}")

@router.post("/clustering", response_model=SemanticClusteringResponse)
async def semantic_clustering(request: SemanticClusteringRequest):
    """Metinleri semantik olarak kümele"""
    try:
        clustering_result = enhanced_embedding_service.semantic_clustering(
            request.texts, request.n_clusters
        )
        return SemanticClusteringResponse(
            texts=request.texts,
            n_clusters=request.n_clusters,
            clusters=clustering_result["clusters"],
            cluster_centers=clustering_result["cluster_centers"],
            model=enhanced_embedding_service.model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

@router.post("/outliers", response_model=OutlierDetectionResponse)
async def find_semantic_outliers(request: OutlierDetectionRequest):
    """Semantik aykırı değerleri tespit et"""
    try:
        outliers = enhanced_embedding_service.find_semantic_outliers(
            request.texts, request.threshold
        )
        return OutlierDetectionResponse(
            texts=request.texts,
            threshold=request.threshold,
            outliers=outliers,
            model=enhanced_embedding_service.model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outlier detection failed: {str(e)}")

@router.get("/stats")
async def get_embedding_stats():
    """Embedding servisi istatistiklerini getir"""
    try:
        stats = enhanced_embedding_service.get_enhanced_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.post("/clear-cache")
async def clear_embedding_cache():
    """Embedding cache'ini temizle"""
    try:
        success = await enhanced_embedding_service.clear_cache()
        return {"success": success, "message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.get("/models")
async def get_available_models():
    """Kullanılabilir embedding modellerini listele"""
    try:
        return {
            "available_models": enhanced_embedding_service.available_models,
            "current_model": enhanced_embedding_service.current_model_key,
            "current_model_name": enhanced_embedding_service.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

@router.post("/switch-model/{model_key}")
async def switch_embedding_model(model_key: str):
    """Embedding modelini değiştir"""
    try:
        success = enhanced_embedding_service.switch_model(model_key)
        if success:
            return {
                "success": True,
                "message": f"Model switched to {model_key}",
                "current_model": enhanced_embedding_service.current_model_key,
                "current_model_name": enhanced_embedding_service.model_name
            }
        else:
            raise HTTPException(status_code=400, detail=f"Invalid model key: {model_key}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch model: {str(e)}")
