"""Search Orchestration Service for E5 + RRF Integration"""

import asyncio
from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger()

class SearchOrchestrationService:
    """Orchestrates different search strategies including E5 + RRF"""
    
    def __init__(self):
        self.strategies = {
            "legacy": {"description": "Traditional vector search only"},
            "hybrid_e5": {"description": "E5-Large-v2 + BM25 + RRF"},
            "advanced_rag": {"description": "Advanced RAG with custom prompts"}
        }
        logger.info("Search orchestration service initialized")
    
    async def search(self, user_id: str, query: str, strategy: str = "legacy", 
                    top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform search using specified strategy"""
        try:
            logger.info("Starting search", strategy=strategy)
            
            if strategy == "legacy":
                return await self._legacy_search(query, top_k, filters)
            elif strategy == "hybrid_e5":
                return await self._hybrid_e5_search(query, top_k, filters)
            elif strategy == "advanced_rag":
                return await self._advanced_rag_search(query, top_k, filters)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
        except Exception as e:
            logger.error("Search failed", error=str(e))
            return {"results": [], "error": str(e)}
    
    async def _legacy_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform legacy vector search"""
        try:
            from app.services.vector_service import vector_service
            results = await vector_service.search(query=query, limit=top_k, filters=filters)
            return {"results": results, "strategy": "legacy"}
        except Exception as e:
            return {"results": [], "error": str(e)}
    
    async def _hybrid_e5_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform hybrid E5 + BM25 + RRF search"""
        try:
            from ml.nlp.hybrid import hybrid_search_engine
            hybrid_engine = await hybrid_search_engine()
            results = await hybrid_engine.search(query=query, top_k=top_k, filters=filters)
            return {"results": results, "strategy": "hybrid_e5"}
        except Exception as e:
            logger.error("Hybrid search failed", error=str(e))
            return await self._legacy_search(query, top_k, filters)
    
    async def _advanced_rag_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform advanced RAG search"""
        try:
            from app.services.advanced_rag_system import AdvancedRAGSystem
            rag_system = AdvancedRAGSystem()
            answer, context_results = await rag_system.query(question=query, k=top_k)
            return {"results": context_results, "answer": answer, "strategy": "advanced_rag"}
        except Exception as e:
            logger.error("RAG search failed", error=str(e))
            return await self._hybrid_e5_search(query, top_k, filters)

# Global instance
search_orchestration_service = SearchOrchestrationService()
