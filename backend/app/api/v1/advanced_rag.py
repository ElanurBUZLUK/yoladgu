"""
Advanced RAG System API endpoints.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File
from pydantic import BaseModel
import os

from app.services.advanced_rag_system import advanced_rag_system
from app.db.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession


router = APIRouter(prefix="/advanced-rag", tags=["Advanced RAG System"])


class DocumentRequest(BaseModel):
    """Request model for adding documents."""
    documents: List[str]
    metadata_list: Optional[List[Dict[str, Any]]] = None


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str
    k: int = 5


class IndexInitializationRequest(BaseModel):
    """Request model for initializing vector store."""
    index_type: str = "ivf"
    nlist: int = 100
    nprobe: int = 10


class BatchProcessingRequest(BaseModel):
    """Request model for batch document processing."""
    documents: List[str]
    batch_size: int = 100


class SimilaritySearchRequest(BaseModel):
    """Request model for similarity search."""
    query: str
    k: int = 5
    threshold: float = 0.5


@router.post("/initialize")
async def initialize_vector_store(
    request: IndexInitializationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Initialize the advanced FAISS vector store."""
    try:
        vector_store = await advanced_rag_system.initialize_vector_store(
            index_type=request.index_type
        )
        
        return {
            "success": True,
            "index_type": request.index_type,
            "nlist": request.nlist,
            "nprobe": request.nprobe,
            "message": "Vector store initialized successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/add")
async def add_documents(
    request: DocumentRequest,
    db: AsyncSession = Depends(get_db)
):
    """Add documents to the vector store."""
    try:
        if not advanced_rag_system.vector_store:
            raise HTTPException(
                status_code=400,
                detail="Vector store not initialized. Call /initialize first."
            )
        
        success = await advanced_rag_system.add_documents(
            documents=request.documents,
            metadata_list=request.metadata_list
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to add documents to vector store"
            )
        
        return {
            "success": True,
            "documents_added": len(request.documents),
            "message": "Documents added successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query_rag_system(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """Query the RAG system with a question."""
    try:
        if not advanced_rag_system.vector_store:
            raise HTTPException(
                status_code=400,
                detail="Vector store not initialized. Call /initialize first."
            )
        
        answer, results = await advanced_rag_system.query(
            question=request.question,
            k=request.k
        )
        
        return {
            "success": True,
            "question": request.question,
            "answer": answer,
            "results": results,
            "num_results": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/batch")
async def batch_process_documents(
    request: BatchProcessingRequest,
    db: AsyncSession = Depends(get_db)
):
    """Process documents in batches for large datasets."""
    try:
        if not advanced_rag_system.vector_store:
            raise HTTPException(
                status_code=400,
                detail="Vector store not initialized. Call /initialize first."
            )
        
        success = await advanced_rag_system.batch_process_documents(
            documents=request.documents,
            batch_size=request.batch_size
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Batch processing failed"
            )
        
        return {
            "success": True,
            "documents_processed": len(request.documents),
            "batch_size": request.batch_size,
            "message": "Batch processing completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/similar")
async def search_similar_documents(
    request: SimilaritySearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """Search for similar documents with similarity threshold."""
    try:
        if not advanced_rag_system.vector_store:
            raise HTTPException(
                status_code=400,
                detail="Vector store not initialized. Call /initialize first."
            )
        
        results = await advanced_rag_system.search_similar_documents(
            query=request.query,
            k=request.k,
            threshold=request.threshold
        )
        
        return {
            "success": True,
            "query": request.query,
            "results": results,
            "num_results": len(results),
            "threshold": request.threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    index_type: str = Query("ivf", description="FAISS index type"),
    db: AsyncSession = Depends(get_db)
):
    """Upload and process documents to create vector store."""
    try:
        # Initialize vector store if not already done
        if not advanced_rag_system.vector_store:
            await advanced_rag_system.initialize_vector_store(index_type=index_type)
        
        # Process uploaded files
        processed_documents = []
        
        for file in files:
            if file.filename.endswith('.txt'):
                # Process text file
                content = await file.read()
                text_content = content.decode('utf-8')
                processed_documents.append(text_content)
                
            elif file.filename.endswith('.pdf'):
                # PDF processing would be added here
                processed_documents.append(f"PDF content from {file.filename}")
                
            elif file.filename.endswith('.docx'):
                # DOCX processing would be added here
                processed_documents.append(f"DOCX content from {file.filename}")
                
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}"
                )
        
        if not processed_documents:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from uploaded files"
            )
        
        # Add documents to vector store
        success = await advanced_rag_system.add_documents(processed_documents)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to add documents to vector store"
            )
        
        return {
            "success": True,
            "files_processed": len(files),
            "documents_added": len(processed_documents),
            "index_type": index_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_rag_system_stats():
    """Get comprehensive RAG system statistics."""
    try:
        stats = {
            "vector_store_initialized": advanced_rag_system.vector_store is not None,
            "embedding_model_loaded": advanced_rag_system.embedding_model is not None
        }
        
        if advanced_rag_system.vector_store:
            vector_stats = await advanced_rag_system.get_vector_store_stats()
            stats["vector_store_stats"] = vector_stats
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize")
async def optimize_index_parameters(
    documents: List[str],
    db: AsyncSession = Depends(get_db)
):
    """Optimize index parameters based on document characteristics."""
    try:
        recommendations = await advanced_rag_system.optimize_index(documents)
        
        return {
            "success": True,
            "num_documents": len(documents),
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save")
async def save_vector_store(
    path: str = Query(..., description="Path to save vector store"),
    db: AsyncSession = Depends(get_db)
):
    """Save vector store to disk."""
    try:
        if not advanced_rag_system.vector_store:
            raise HTTPException(
                status_code=400,
                detail="Vector store not initialized. Call /initialize first."
            )
        
        success = await advanced_rag_system.save_vector_store(path)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save vector store"
            )
        
        return {
            "success": True,
            "path": path,
            "message": "Vector store saved successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_vector_store(
    path: str = Query(..., description="Path to load vector store from"),
    db: AsyncSession = Depends(get_db)
):
    """Load vector store from disk."""
    try:
        success = await advanced_rag_system.load_vector_store(path)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to load vector store"
            )
        
        return {
            "success": True,
            "path": path,
            "message": "Vector store loaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for the RAG system."""
    try:
        status = {
            "status": "healthy",
            "vector_store": advanced_rag_system.vector_store is not None,
            "embedding_model": advanced_rag_system.embedding_model is not None
        }
        
        if advanced_rag_system.vector_store:
            stats = await advanced_rag_system.get_vector_store_stats()
            status["vector_store_stats"] = stats
        
        return status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
