"""
Advanced vector operations API with FAISS advanced indexes and embedding model management.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel
import numpy as np

from app.services.index_backends.faiss_advanced_index import FAISSAdvancedIndexBackend
from app.services.embedding_service import embedding_service
from app.db.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/vector/advanced", tags=["Advanced Vector Operations"])


class AdvancedIndexRequest(BaseModel):
    """Request model for creating advanced FAISS indexes."""
    index_type: str  # "ivf", "hnsw", "pq", "sq", "ivfpq", "ivfsq"
    metric: str = "ip"  # "ip" for cosine, "l2" for euclidean
    nlist: Optional[int] = 100
    nprobe: Optional[int] = 10
    m: Optional[int] = 8
    bits: Optional[int] = 8
    ef_construction: Optional[int] = 200
    ef_search: Optional[int] = 128
    max_elements: Optional[int] = 1000000
    index_path: Optional[str] = None


class HyperparameterOptimizationRequest(BaseModel):
    """Request model for hyperparameter optimization."""
    index_type: str
    training_vectors: List[List[float]]
    validation_queries: List[List[float]]
    validation_ground_truth: List[List[str]]
    param_grid: Optional[Dict[str, List[Any]]] = None


class EmbeddingModelRequest(BaseModel):
    """Request model for embedding model operations."""
    model_name: str
    test_texts: Optional[List[str]] = None
    evaluation_metrics: Optional[List[str]] = None


class BenchmarkRequest(BaseModel):
    """Request model for benchmarking."""
    models: Optional[List[str]] = None
    test_texts: List[str]
    evaluation_metrics: Optional[List[str]] = None


# Global advanced index instances
advanced_indexes: Dict[str, FAISSAdvancedIndexBackend] = {}


@router.post("/indexes/create")
async def create_advanced_index(
    request: AdvancedIndexRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create a new advanced FAISS index."""
    try:
        # Validate index type
        valid_types = ["ivf", "hnsw", "pq", "sq", "ivfpq", "ivfsq"]
        if request.index_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid index type. Must be one of: {valid_types}"
            )
        
        # Create index path if not provided
        if not request.index_path:
            request.index_path = f"data/faiss_{request.index_type}_index.index"
        
        # Create advanced index
        index = FAISSAdvancedIndexBackend(
            vector_size=384,  # Default size, can be made configurable
            index_type=request.index_type,
            metric=request.metric,
            nlist=request.nlist,
            nprobe=request.nprobe,
            m=request.m,
            bits=request.bits,
            ef_construction=request.ef_construction,
            ef_search=request.ef_search,
            max_elements=request.max_elements,
            index_path=request.index_path
        )
        
        # Initialize index
        success = await index.initialize()
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize {request.index_type} index"
            )
        
        # Store index instance
        index_id = f"{request.index_type}_{len(advanced_indexes)}"
        advanced_indexes[index_id] = index
        
        # Get initial stats
        stats = await index.get_stats()
        
        return {
            "success": True,
            "index_id": index_id,
            "index_type": request.index_type,
            "index_path": request.index_path,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indexes/list")
async def list_advanced_indexes():
    """List all advanced FAISS indexes."""
    try:
        indexes = []
        for index_id, index in advanced_indexes.items():
            stats = await index.get_stats()
            indexes.append({
                "index_id": index_id,
                "index_type": index.index_type,
                "stats": stats
            })
        
        return {
            "total_indexes": len(indexes),
            "indexes": indexes
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/indexes/{index_id}/add")
async def add_to_advanced_index(
    index_id: str,
    item_id: str = Query(..., description="Item ID"),
    text: str = Query(..., description="Text content to encode"),
    metadata: Optional[Dict[str, Any]] = None,
    db: AsyncSession = Depends(get_db)
):
    """Add an item to an advanced FAISS index."""
    try:
        if index_id not in advanced_indexes:
            raise HTTPException(
                status_code=404,
                detail=f"Index {index_id} not found"
            )
        
        index = advanced_indexes[index_id]
        
        # Encode text using current embedding model
        vector = await embedding_service.encode_text(text)
        vector_array = np.array([vector], dtype=np.float32)
        
        # Add to index
        success = await index.add_items(
            vectors=vector_array,
            ids=[item_id],
            metadata=[metadata or {}]
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to add item to index"
            )
        
        # Get updated stats
        stats = await index.get_stats()
        
        return {
            "success": True,
            "item_id": item_id,
            "index_stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/indexes/{index_id}/search")
async def search_advanced_index(
    index_id: str,
    query: str = Query(..., description="Search query"),
    k: int = Query(10, description="Number of results"),
    filters: Optional[Dict[str, Any]] = None,
    db: AsyncSession = Depends(get_db)
):
    """Search an advanced FAISS index."""
    try:
        if index_id not in advanced_indexes:
            raise HTTPException(
                status_code=404,
                detail=f"Index {index_id} not found"
            )
        
        index = advanced_indexes[index_id]
        
        # Encode query
        query_vector = await embedding_service.encode_text(query)
        
        # Search index
        results = await index.search(
            query_vector=query_vector,
            k=k,
            filters=filters
        )
        
        return {
            "query": query,
            "k": k,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/indexes/{index_id}/optimize")
async def optimize_hyperparameters(
    index_id: str,
    request: HyperparameterOptimizationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Optimize hyperparameters for an advanced FAISS index."""
    try:
        if index_id not in advanced_indexes:
            raise HTTPException(
                status_code=404,
                detail=f"Index {index_id} not found"
            )
        
        index = advanced_indexes[index_id]
        
        # Convert lists to numpy arrays
        training_vectors = np.array(request.training_vectors, dtype=np.float32)
        validation_queries = np.array(request.validation_queries, dtype=np.float32)
        
        # Run optimization in background
        def run_optimization():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    index.optimize_hyperparameters(
                        training_vectors=training_vectors,
                        validation_queries=validation_queries,
                        validation_ground_truth=request.validation_ground_truth,
                        param_grid=request.param_grid
                    )
                )
                print(f"✅ Hyperparameter optimization completed for {index_id}")
                print(f"🎯 Best parameters: {result.get('best_params', {})}")
                print(f"🏆 Best score: {result.get('best_score', 0.0):.4f}")
            finally:
                loop.close()
        
        background_tasks.add_task(run_optimization)
        
        return {
            "success": True,
            "message": "Hyperparameter optimization started in background",
            "index_id": index_id,
            "index_type": index.index_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indexes/{index_id}/stats")
async def get_advanced_index_stats(
    index_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get statistics for an advanced FAISS index."""
    try:
        if index_id not in advanced_indexes:
            raise HTTPException(
                status_code=404,
                detail=f"Index {index_id} not found"
            )
        
        index = advanced_indexes[index_id]
        stats = await index.get_stats()
        
        return {
            "index_id": index_id,
            "index_type": index.index_type,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embedding/models/load")
async def load_embedding_model(
    request: EmbeddingModelRequest,
    db: AsyncSession = Depends(get_db)
):
    """Load a specific embedding model."""
    try:
        success = await embedding_service.load_model(request.model_name)
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model {request.model_name}"
            )
        
        model_info = embedding_service.get_current_model_info()
        
        return {
            "success": True,
            "model_loaded": request.model_name,
            "model_info": model_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/embedding/models/list")
async def list_embedding_models():
    """List all available embedding models."""
    try:
        models = embedding_service.list_available_models()
        current_model = embedding_service.get_current_model_info()
        
        return {
            "available_models": models,
            "current_model": current_model
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embedding/models/evaluate")
async def evaluate_embedding_model(
    request: EmbeddingModelRequest,
    db: AsyncSession = Depends(get_db)
):
    """Evaluate an embedding model's performance."""
    try:
        if not request.test_texts:
            raise HTTPException(
                status_code=400,
                detail="test_texts is required for evaluation"
            )
        
        # Evaluate model
        results = await embedding_service.evaluate_model_performance(
            model_name=request.model_name,
            test_texts=request.test_texts,
            evaluation_metrics=request.evaluation_metrics
        )
        
        if "error" in results:
            raise HTTPException(
                status_code=500,
                detail=results["error"]
            )
        
        return {
            "success": True,
            "evaluation_results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embedding/models/compare")
async def compare_embedding_models(
    model_names: List[str],
    test_texts: List[str],
    evaluation_metrics: Optional[List[str]] = None,
    db: AsyncSession = Depends(get_db)
):
    """Compare multiple embedding models."""
    try:
        if not test_texts:
            raise HTTPException(
                status_code=400,
                detail="test_texts is required for comparison"
            )
        
        # Compare models
        results = await embedding_service.compare_models(
            model_names=model_names,
            test_texts=test_texts,
            evaluation_metrics=evaluation_metrics
        )
        
        return {
            "success": True,
            "comparison_results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embedding/models/benchmark")
async def benchmark_embedding_models(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Benchmark multiple embedding models."""
    try:
        if not request.test_texts:
            raise HTTPException(
                status_code=400,
                detail="test_texts is required for benchmarking"
            )
        
        # Run benchmark in background
        def run_benchmark():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    embedding_service.benchmark_models(
                        test_texts=request.test_texts,
                        models=request.models
                    )
                )
                print("✅ Benchmark completed successfully")
                for model_name, model_results in result.items():
                    if "error" not in model_results:
                        print(f"📊 {model_name}: {model_results.get('encoding_speed', {}).get('texts_per_second', 0):.2f} texts/sec")
            finally:
                loop.close()
        
        background_tasks.add_task(run_benchmark)
        
        return {
            "success": True,
            "message": "Benchmark started in background",
            "models_to_benchmark": request.models or list(embedding_service.model_configs.keys()),
            "test_size": len(request.test_texts)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/embedding/stats")
async def get_embedding_stats():
    """Get embedding service statistics."""
    try:
        stats = await embedding_service.get_stats()
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/indexes/{index_id}/save")
async def save_advanced_index(
    index_id: str,
    path: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Save an advanced FAISS index to disk."""
    try:
        if index_id not in advanced_indexes:
            raise HTTPException(
                status_code=404,
                detail=f"Index {index_id} not found"
            )
        
        index = advanced_indexes[index_id]
        success = await index.save_index(path)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save index"
            )
        
        return {
            "success": True,
            "index_id": index_id,
            "saved_path": path or index.index_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/indexes/{index_id}")
async def delete_advanced_index(
    index_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete an advanced FAISS index."""
    try:
        if index_id not in advanced_indexes:
            raise HTTPException(
                status_code=404,
                detail=f"Index {index_id} not found"
            )
        
        # Remove from memory
        del advanced_indexes[index_id]
        
        return {
            "success": True,
            "message": f"Index {index_id} deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/{index_type}")
async def get_index_health(
    index_type: str,
    vector_size: int = 384,
    metric: str = "ip"
):
    """Get health status of a specific index type."""
    try:
        # Create index instance
        index = FAISSAdvancedIndexBackend(
            vector_size=vector_size,
            index_type=index_type,
            metric=metric
        )
        
        # Initialize if not already done
        if not index._initialized:
            await index.initialize()
        
        # Get health status
        health = await index.health_check()
        
        return {
            "status": "success",
            "index_type": index_type,
            "health": health
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
