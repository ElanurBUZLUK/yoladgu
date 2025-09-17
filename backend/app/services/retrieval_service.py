"""
Retrieval Service with Comprehensive Logging
Tracks dense and sparse search results for ML backend selector training
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from pydantic import BaseModel

from app.db.session import get_db
from app.services.vector_service import VectorService
from app.services.recommenders.error_aware import ErrorAwareRecommender

logger = structlog.get_logger()

class RetrievalCandidate(BaseModel):
    """Single retrieval candidate result"""
    item_id: str
    score: float
    backend: str
    metadata: Dict[str, Any] = {}

class RetrievalLog(BaseModel):
    """Retrieval log entry"""
    user_id: str
    request_id: str
    query: str
    candidates: List[RetrievalCandidate]
    bm25_scores: Dict[str, float] = {}
    dense_scores: Dict[str, float] = {}
    durations: Dict[str, float] = {}
    selected_backend: str
    top_k: int
    filters: Dict[str, Any] = {}
    timestamp: datetime

class RetrievalService:
    """Service for retrieval operations with comprehensive logging"""
    
    def __init__(self):
        self.vector_service = VectorService()
        self.error_aware_recommender = ErrorAwareRecommender()
        self.logs_buffer: List[RetrievalLog] = []
        self.buffer_size = 100
        self.flush_interval = 30  # seconds
        
        # Background logging task will be started when first used
        
        logger.info("Retrieval service initialized with logging")
    
    async def search_with_logging(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        backend_preference: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        db: Optional[AsyncSession] = None
    ) -> Tuple[List[Dict[str, Any]], RetrievalLog]:
        """
        Perform retrieval with comprehensive logging
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Number of results to return
            backend_preference: Preferred backend (optional)
            filters: Search filters
            db: Database session
            
        Returns:
            Tuple of (results, retrieval_log)
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info("Starting retrieval with logging", 
                   user_id=user_id, request_id=request_id, query=query, top_k=top_k)
        
        # Initialize tracking
        candidates = []
        bm25_scores = {}
        dense_scores = {}
        durations = {}
        selected_backend = backend_preference or "auto"
        
        try:
            # Try multiple backends if no preference specified
            if not backend_preference:
                backends_to_try = ["faiss", "hnsw", "qdrant"]
            else:
                backends_to_try = [backend_preference]
            
            best_results = []
            best_backend = None
            best_score = -1
            
            for backend in backends_to_try:
                try:
                    backend_start = time.time()
                    
                    # Perform search with specific backend
                    if backend == "faiss":
                        results = await self._search_faiss(user_id, query, top_k, filters)
                    elif backend == "hnsw":
                        results = await self._search_hnsw(user_id, query, top_k, filters)
                    elif backend == "qdrant":
                        results = await self._search_qdrant(user_id, query, top_k, filters)
                    else:
                        continue
                    
                    backend_duration = time.time() - backend_start
                    durations[backend] = backend_duration
                    
                    # Calculate average score for backend selection
                    if results:
                        avg_score = sum(r.get("score", 0) for r in results) / len(results)
                        
                        # Store scores by type
                        if backend in ["faiss", "hnsw"]:
                            dense_scores[backend] = avg_score
                        else:
                            bm25_scores[backend] = avg_score
                        
                        # Create candidates
                        for i, result in enumerate(results):
                            candidate = RetrievalCandidate(
                                item_id=result.get("id", f"item_{i}"),
                                score=result.get("score", 0),
                                backend=backend,
                                metadata=result.get("metadata", {})
                            )
                            candidates.append(candidate)
                        
                        # Select best backend based on score
                        if avg_score > best_score:
                            best_score = avg_score
                            best_results = results
                            best_backend = backend
                    
                    logger.debug("Backend search completed", 
                               backend=backend, duration=backend_duration, 
                               results_count=len(results), avg_score=avg_score if results else 0)
                
                except Exception as e:
                    logger.warning("Backend search failed", backend=backend, error=str(e))
                    durations[backend] = -1  # Mark as failed
                    continue
            
            # Update selected backend
            selected_backend = best_backend or backends_to_try[0]
            
            # If no results from any backend, try error-aware recommendation
            if not best_results:
                logger.info("No results from vector backends, trying error-aware recommendation")
                error_start = time.time()
                
                best_results = await self.error_aware_recommender.recommend(
                    user_id=user_id,
                    n_recommendations=top_k,
                    db=db
                )
                
                durations["error_aware"] = time.time() - error_start
                selected_backend = "error_aware"
                
                # Create candidates for error-aware results
                for i, result in enumerate(best_results):
                    candidate = RetrievalCandidate(
                        item_id=result.get("id", f"item_{i}"),
                        score=result.get("score", 0.5),
                        backend="error_aware",
                        metadata=result.get("metadata", {})
                    )
                    candidates.append(candidate)
            
            total_duration = time.time() - start_time
            
            # Create retrieval log
            retrieval_log = RetrievalLog(
                user_id=user_id,
                request_id=request_id,
                query=query,
                candidates=candidates,
                bm25_scores=bm25_scores,
                dense_scores=dense_scores,
                durations=durations,
                selected_backend=selected_backend,
                top_k=top_k,
                filters=filters or {},
                timestamp=datetime.now()
            )
            
            # Add to buffer for batch writing
            await self._add_log_to_buffer(retrieval_log)
            
            logger.info("Retrieval completed with logging", 
                       user_id=user_id, request_id=request_id,
                       selected_backend=selected_backend, 
                       results_count=len(best_results),
                       total_duration=total_duration)
            
            return best_results, retrieval_log
            
        except Exception as e:
            logger.error("Retrieval failed", user_id=user_id, request_id=request_id, error=str(e))
            
            # Create error log
            error_log = RetrievalLog(
                user_id=user_id,
                request_id=request_id,
                query=query,
                candidates=[],
                durations=durations,
                selected_backend="error",
                top_k=top_k,
                filters=filters or {},
                timestamp=datetime.now()
            )
            
            await self._add_log_to_buffer(error_log)
            raise
    
    async def _search_faiss(self, user_id: str, query: str, top_k: int, 
                          filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search using FAISS backend"""
        try:
            # Get query embedding
            query_vector = await self.vector_service.get_query_embedding(query)
            
            # Search with FAISS
            results = await self.vector_service.search(
                query_vector=query_vector,
                k=top_k,
                backend_name="faiss",
                filters=filters
            )
            
            return results
            
        except Exception as e:
            logger.warning("FAISS search failed", error=str(e))
            return []
    
    async def _search_hnsw(self, user_id: str, query: str, top_k: int, 
                         filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search using HNSW backend"""
        try:
            # Get query embedding
            query_vector = await self.vector_service.get_query_embedding(query)
            
            # Search with HNSW
            results = await self.vector_service.search(
                query_vector=query_vector,
                k=top_k,
                backend_name="hnsw",
                filters=filters
            )
            
            return results
            
        except Exception as e:
            logger.warning("HNSW search failed", error=str(e))
            return []
    
    async def _search_qdrant(self, user_id: str, query: str, top_k: int, 
                           filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search using Qdrant backend"""
        try:
            # Get query embedding
            query_vector = await self.vector_service.get_query_embedding(query)
            
            # Search with Qdrant
            results = await self.vector_service.search(
                query_vector=query_vector,
                k=top_k,
                backend_name="qdrant",
                filters=filters
            )
            
            return results
            
        except Exception as e:
            logger.warning("Qdrant search failed", error=str(e))
            return []
    
    async def _add_log_to_buffer(self, log: RetrievalLog):
        """Add log to buffer for batch writing"""
        self.logs_buffer.append(log)
        
        # Flush if buffer is full
        if len(self.logs_buffer) >= self.buffer_size:
            await self._flush_logs_to_db()
    
    async def _periodic_flush_logs(self):
        """Periodically flush logs to database"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                if self.logs_buffer:
                    await self._flush_logs_to_db()
            except Exception as e:
                logger.error("Failed to flush logs periodically", error=str(e))
    
    async def _flush_logs_to_db(self):
        """Flush buffered logs to database"""
        if not self.logs_buffer:
            return
        
        try:
            async with get_db() as db:
                # Prepare batch insert data
                log_data = []
                for log in self.logs_buffer:
                    log_data.append({
                        "user_id": log.user_id,
                        "request_id": log.request_id,
                        "query": log.query,
                        "candidates": [c.dict() for c in log.candidates],
                        "bm25_scores": log.bm25_scores,
                        "dense_scores": log.dense_scores,
                        "durations": log.durations,
                        "selected_backend": log.selected_backend,
                        "top_k": log.top_k,
                        "filters": log.filters,
                        "timestamp": log.timestamp
                    })
                
                # Batch insert using raw SQL for performance
                insert_sql = text("""
                    INSERT INTO retrieval_logs (
                        user_id, request_id, query, candidates, 
                        bm25_scores, dense_scores, durations,
                        selected_backend, top_k, filters, timestamp
                    ) VALUES (
                        :user_id, :request_id, :query, :candidates,
                        :bm25_scores, :dense_scores, :durations,
                        :selected_backend, :top_k, :filters, :timestamp
                    )
                """)
                
                await db.execute(insert_sql, log_data)
                await db.commit()
                
                logger.info("Flushed retrieval logs to database", count=len(log_data))
                
                # Clear buffer
                self.logs_buffer.clear()
                
        except Exception as e:
            logger.error("Failed to flush logs to database", error=str(e))
            # Keep logs in buffer for retry
    
    async def get_retrieval_stats(self, user_id: Optional[str] = None, 
                                hours: int = 24) -> Dict[str, Any]:
        """Get retrieval statistics"""
        try:
            async with get_db() as db:
                # Build query
                where_clause = "WHERE timestamp >= NOW() - INTERVAL :hours HOUR"
                params = {"hours": hours}
                
                if user_id:
                    where_clause += " AND user_id = :user_id"
                    params["user_id"] = user_id
                
                # Get basic stats
                stats_sql = text(f"""
                    SELECT 
                        COUNT(*) as total_requests,
                        COUNT(DISTINCT user_id) as unique_users,
                        AVG(EXTRACT(EPOCH FROM durations->>'faiss')) as avg_faiss_duration,
                        AVG(EXTRACT(EPOCH FROM durations->>'hnsw')) as avg_hnsw_duration,
                        AVG(EXTRACT(EPOCH FROM durations->>'qdrant')) as avg_qdrant_duration,
                        COUNT(CASE WHEN selected_backend = 'faiss' THEN 1 END) as faiss_selections,
                        COUNT(CASE WHEN selected_backend = 'hnsw' THEN 1 END) as hnsw_selections,
                        COUNT(CASE WHEN selected_backend = 'qdrant' THEN 1 END) as qdrant_selections,
                        COUNT(CASE WHEN selected_backend = 'error_aware' THEN 1 END) as error_aware_selections
                    FROM retrieval_logs 
                    {where_clause}
                """)
                
                result = await db.execute(stats_sql, params)
                stats = result.fetchone()
                
                return {
                    "total_requests": stats.total_requests or 0,
                    "unique_users": stats.unique_users or 0,
                    "avg_durations": {
                        "faiss": stats.avg_faiss_duration or 0,
                        "hnsw": stats.avg_hnsw_duration or 0,
                        "qdrant": stats.avg_qdrant_duration or 0
                    },
                    "backend_selections": {
                        "faiss": stats.faiss_selections or 0,
                        "hnsw": stats.hnsw_selections or 0,
                        "qdrant": stats.qdrant_selections or 0,
                        "error_aware": stats.error_aware_selections or 0
                    }
                }
                
        except Exception as e:
            logger.error("Failed to get retrieval stats", error=str(e))
            return {}

# Global retrieval service instance
retrieval_service = RetrievalService()