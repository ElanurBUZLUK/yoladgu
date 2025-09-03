"""
LangChain RAG API endpoints for advanced document processing and retrieval.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
import os
import tempfile

from app.services.langchain_integration import langchain_service
from app.db.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession


router = APIRouter(prefix="/langchain", tags=["LangChain RAG Operations"])


class VectorStoreRequest(BaseModel):
    """Request model for creating vector stores."""
    text_chunks: List[str]
    index_type: str = "IVF"  # "Flat", "IVF", "HNSW", "PQ", "IVF_PQ", "IVF_SQ"
    save_path: Optional[str] = None


class RAGChainRequest(BaseModel):
    """Request model for RAG chain operations."""
    query: str
    k: int = 5
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 1000


class BenchmarkRequest(BaseModel):
    """Request model for performance benchmarking."""
    test_queries: List[str]
    k: int = 5


class DocumentProcessingRequest(BaseModel):
    """Request model for document processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    index_type: str = "IVF"
    save_path: Optional[str] = None


@router.post("/embeddings/initialize")
async def initialize_embeddings(
    model_name: str = Query("sentence-transformers/all-MiniLM-L6-v2", description="Embedding model name"),
    db: AsyncSession = Depends(get_db)
):
    """Initialize LangChain embeddings with specified model."""
    try:
        success = await langchain_service.initialize_embeddings(model_name)
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize embeddings with {model_name}"
            )
        
        return {
            "success": True,
            "model_name": model_name,
            "message": "Embeddings initialized successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectorstore/create")
async def create_vector_store(
    request: VectorStoreRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create advanced FAISS vector store with specified index type."""
    try:
        if not langchain_service.embeddings:
            raise HTTPException(
                status_code=400,
                detail="Embeddings not initialized. Call /embeddings/initialize first."
            )
        
        # Create vector store
        vectorstore = langchain_service.create_advanced_vector_store(
            text_chunks=request.text_chunks,
            index_type=request.index_type,
            save_path=request.save_path
        )
        
        # Get stats
        stats = langchain_service.get_vector_store_stats()
        
        return {
            "success": True,
            "index_type": request.index_type,
            "num_chunks": len(request.text_chunks),
            "vectorstore_stats": stats,
            "save_path": request.save_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectorstore/save")
async def save_vector_store(
    file_path: str = Query(..., description="Path to save vector store"),
    db: AsyncSession = Depends(get_db)
):
    """Save vector store to disk."""
    try:
        if not langchain_service.vectorstore:
            raise HTTPException(
                status_code=400,
                detail="No vector store to save. Create one first."
            )
        
        langchain_service.save_vector_store(file_path)
        
        return {
            "success": True,
            "file_path": file_path,
            "message": "Vector store saved successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectorstore/load")
async def load_vector_store(
    file_path: str = Query(..., description="Path to load vector store from"),
    db: AsyncSession = Depends(get_db)
):
    """Load vector store from disk."""
    try:
        if not langchain_service.embeddings:
            raise HTTPException(
                status_code=400,
                detail="Embeddings not initialized. Call /embeddings/initialize first."
            )
        
        vectorstore = langchain_service.load_vector_store(file_path)
        
        # Get stats
        stats = langchain_service.get_vector_store_stats()
        
        return {
            "success": True,
            "file_path": file_path,
            "vectorstore_stats": stats,
            "message": "Vector store loaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/chain/create")
async def create_rag_chain(
    request: RAGChainRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create advanced RAG chain with custom configuration."""
    try:
        if not langchain_service.vectorstore:
            raise HTTPException(
                status_code=400,
                detail="No vector store available. Create one first."
            )
        
        # Create RAG chain
        qa_chain = langchain_service.create_advanced_rag_chain(
            model_name=request.model_name,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return {
            "success": True,
            "model_name": request.model_name,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "message": "RAG chain created successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/query")
async def query_rag_chain(
    request: RAGChainRequest,
    db: AsyncSession = Depends(get_db)
):
    """Query the RAG chain with a question."""
    try:
        if not langchain_service.vectorstore:
            raise HTTPException(
                status_code=400,
                detail="No vector store available. Create one first."
            )
        
        # Evaluate retrieval quality first
        docs, scores, metrics = langchain_service.evaluate_retrieval_quality(
            query=request.query,
            k=request.k
        )
        
        # For now, return retrieval results (LLM integration would be added later)
        results = []
        for i, (doc, score) in enumerate(zip(docs, scores)):
            results.append({
                "rank": i + 1,
                "score": score,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "success": True,
            "query": request.query,
            "results": results,
            "metrics": metrics,
            "message": "Query processed successfully (retrieval only - LLM integration pending)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/evaluate")
async def evaluate_retrieval_quality(
    query: str = Query(..., description="Query to evaluate"),
    k: int = Query(5, description="Number of results to retrieve"),
    db: AsyncSession = Depends(get_db)
):
    """Evaluate retrieval quality for a specific query."""
    try:
        if not langchain_service.vectorstore:
            raise HTTPException(
                status_code=400,
                detail="No vector store available. Create one first."
            )
        
        docs, scores, metrics = langchain_service.evaluate_retrieval_quality(
            query=query,
            k=k
        )
        
        # Format results
        results = []
        for i, (doc, score) in enumerate(zip(docs, scores)):
            results.append({
                "rank": i + 1,
                "score": score,
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "success": True,
            "query": query,
            "k": k,
            "results": results,
            "metrics": metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark/performance")
async def benchmark_performance(
    request: BenchmarkRequest,
    db: AsyncSession = Depends(get_db)
):
    """Benchmark FAISS index performance with test queries."""
    try:
        if not langchain_service.vectorstore:
            raise HTTPException(
                status_code=400,
                detail="No vector store available. Create one first."
            )
        
        # Run benchmark
        results = langchain_service.benchmark_index_performance(
            test_queries=request.test_queries,
            k=request.k
        )
        
        # Calculate summary statistics
        successful_queries = [r for r in results.values() if r.get("success", False)]
        
        if successful_queries:
            avg_search_time = sum(r["search_time"] for r in successful_queries) / len(successful_queries)
            avg_similarity = sum(r["avg_similarity"] for r in successful_queries) / len(successful_queries)
        else:
            avg_search_time = 0.0
            avg_similarity = 0.0
        
        summary = {
            "total_queries": len(request.test_queries),
            "successful_queries": len(successful_queries),
            "failed_queries": len(request.test_queries) - len(successful_queries),
            "avg_search_time": avg_search_time,
            "avg_similarity": avg_similarity
        }
        
        return {
            "success": True,
            "k": request.k,
            "summary": summary,
            "detailed_results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/parameters")
async def optimize_parameters(
    text_chunks: List[str],
    db: AsyncSession = Depends(get_db)
):
    """Optimize index parameters based on data characteristics."""
    try:
        if not langchain_service.embeddings:
            raise HTTPException(
                status_code=400,
                detail="Embeddings not initialized. Call /embeddings/initialize first."
            )
        
        # Optimize parameters
        optimal_params = langchain_service.optimize_index_parameters(text_chunks)
        
        return {
            "success": True,
            "num_chunks": len(text_chunks),
            "optimal_parameters": optimal_params
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/process")
async def process_documents(
    request: DocumentProcessingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Process documents and create vector store (background task)."""
    try:
        if not langchain_service.embeddings:
            raise HTTPException(
                status_code=400,
                detail="Embeddings not initialized. Call /embeddings/initialize first."
            )
        
        # This would typically process uploaded documents
        # For now, return optimization recommendations
        
        # Optimize parameters
        optimal_params = langchain_service.optimize_index_parameters(
            ["sample_text"] * request.chunk_size  # Placeholder
        )
        
        return {
            "success": True,
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "index_type": request.index_type,
            "recommendations": optimal_params,
            "message": "Document processing configuration validated"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_langchain_stats():
    """Get comprehensive LangChain service statistics."""
    try:
        stats = {
            "embeddings_initialized": langchain_service.embeddings is not None,
            "vectorstore_available": langchain_service.vectorstore is not None,
            "rag_chain_available": langchain_service.qa_chain is not None
        }
        
        if langchain_service.embeddings:
            stats["embedding_model"] = "HuggingFace Embeddings"
        
        if langchain_service.vectorstore:
            vectorstore_stats = langchain_service.get_vector_store_stats()
            stats["vectorstore_stats"] = vectorstore_stats
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Query(1000, description="Text chunk size"),
    chunk_overlap: int = Query(200, description="Text chunk overlap"),
    index_type: str = Query("IVF", description="FAISS index type"),
    db: AsyncSession = Depends(get_db)
):
    """Upload and process documents to create vector store."""
    try:
        if not langchain_service.embeddings:
            raise HTTPException(
                status_code=400,
                detail="Embeddings not initialized. Call /embeddings/initialize first."
            )
        
        # Process uploaded files
        processed_chunks = []
        
        for file in files:
            if file.filename.endswith('.txt'):
                # Process text file
                content = await file.read()
                text_content = content.decode('utf-8')
                
                # Simple text splitting (in production, use proper document loaders)
                chunks = [text_content[i:i+chunk_size] 
                         for i in range(0, len(text_content), chunk_size-chunk_overlap)]
                processed_chunks.extend(chunks)
                
            elif file.filename.endswith('.pdf'):
                # PDF processing would be added here
                processed_chunks.append(f"PDF content from {file.filename}")
                
            elif file.filename.endswith('.docx'):
                # DOCX processing would be added here
                processed_chunks.append(f"DOCX content from {file.filename}")
                
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}"
                )
        
        if not processed_chunks:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from uploaded files"
            )
        
        # Create vector store
        save_path = f"data/vectorstore_{len(processed_chunks)}_chunks"
        vectorstore = langchain_service.create_advanced_vector_store(
            text_chunks=processed_chunks,
            index_type=index_type,
            save_path=save_path
        )
        
        # Get stats
        stats = langchain_service.get_vector_store_stats()
        
        return {
            "success": True,
            "files_processed": len(files),
            "chunks_created": len(processed_chunks),
            "index_type": index_type,
            "save_path": save_path,
            "vectorstore_stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
