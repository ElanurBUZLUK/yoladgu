"""
Vector Index Manager for managing multiple vector index backends.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from enum import Enum

from app.core.config import settings
from app.services.index_backends.base import BaseIndexBackend
from app.services.index_backends.hnsw_index import HNSWIndexBackend
from app.services.index_backends.qdrant_index import QdrantIndexBackend
from app.services.index_backends.faiss_flat_index import FAISSFlatIndexBackend


class IndexType(Enum):
    """Supported index types."""
    QDRANT = "qdrant"
    HNSW = "hnsw"
    FAISS = "faiss"


class VectorIndexManager:
    """Manager for multiple vector index backends."""
    
    def __init__(self):
        self.backends: Dict[str, BaseIndexBackend] = {}
        self.default_backend = "qdrant"
        self.vector_size = 384  # sentence-transformers/all-MiniLM-L6-v2 size
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "backend_usage": {},
            "avg_response_time_ms": 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize all configured index backends."""
        try:
            # Initialize Qdrant backend
            qdrant_backend = QdrantIndexBackend(
                vector_size=self.vector_size,
                collection_name=settings.VECTOR_DB_COLLECTION,
                url=settings.VECTOR_DB_URL
            )
            
            if await qdrant_backend.initialize():
                self.backends["qdrant"] = qdrant_backend
                self.stats["backend_usage"]["qdrant"] = 0
                print("Qdrant backend initialized successfully")
            
            # Initialize HNSW backend (optional)
            try:
                # Remove old HNSW index file if exists
                import os
                hnsw_index_path = f"data/hnsw_index_{settings.VECTOR_DB_COLLECTION}.index"
                if os.path.exists(hnsw_index_path):
                    os.remove(hnsw_index_path)
                    print(f"Removed old HNSW index: {hnsw_index_path}")
                
                # Also remove metadata file if exists
                metadata_path = hnsw_index_path.replace('.index', '.metadata.pkl')
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    print(f"Removed old HNSW metadata: {metadata_path}")
                
                hnsw_backend = HNSWIndexBackend(
                    vector_size=self.vector_size,
                    space="ip",  # Inner product for cosine similarity
                    M=64,
                    ef_construction=800,
                    ef_search=512,
                    max_elements=100000,
                    index_path=hnsw_index_path
                )
                
                if await hnsw_backend.initialize():
                    self.backends["hnsw"] = hnsw_backend
                    self.stats["backend_usage"]["hnsw"] = 0
                    print("HNSW backend initialized successfully")
                    
            except ImportError:
                print("HNSW backend not available (hnswlib not installed)")
            except Exception as e:
                print(f"HNSW backend initialization failed: {e}")
            
            # Initialize FAISS Flat backend (optional)
            try:
                faiss_backend = FAISSFlatIndexBackend(
                    vector_size=self.vector_size,
                    metric="ip"
                )
                
                if await faiss_backend.initialize():
                    self.backends["faiss"] = faiss_backend
                    self.stats["backend_usage"]["faiss"] = 0
                    print("FAISS-Flat backend initialized successfully")
                    
            except ImportError:
                print("FAISS backend not available (faiss-cpu not installed)")
            except Exception as e:
                print(f"FAISS backend initialization failed: {e}")
            
            return len(self.backends) > 0
            
        except Exception as e:
            print(f"Error initializing vector index manager: {e}")
            return False
    
    def get_backend(self, backend_name: Optional[str] = None) -> BaseIndexBackend:
        """Get a specific backend or the default one."""
        backend_name = backend_name or self.default_backend
        
        if backend_name not in self.backends:
            raise ValueError(f"Backend '{backend_name}' not available. Available: {list(self.backends.keys())}")
        
        return self.backends[backend_name]
    
    def choose_backend(self, *, k: int, filters: Optional[Dict[str, Any]], prefer: str | None = None) -> str:
        """
        Intelligently choose the best backend based on query parameters.
        
        Args:
            k: Number of results to return
            filters: Optional filters to apply
            prefer: Preferred backend if specified
            
        Returns:
            Name of the chosen backend
        """
        if prefer and prefer in self.backends:
            return prefer
        
        total = 0
        for b in self.backends.values():
            try:
                st = asyncio.get_event_loop().run_until_complete(b.get_stats()) if asyncio.get_event_loop().is_running() is False else {}
            except:
                st = {}
            total = max(total, st.get("total_items", 0))

        has_filters = bool(filters)
        
        # Simple heuristics:
        if has_filters and "qdrant" in self.backends:
            return "qdrant"  # Qdrant handles filters better
        if total <= 100_000 and "faiss" in self.backends:
            # Small/medium data + large k means exact search is good
            if k >= 20 or "hnsw" not in self.backends:
                return "faiss"
        if "hnsw" in self.backends:
            return "hnsw"  # Default to HNSW for speed
        if "faiss" in self.backends:
            return "faiss"  # FAISS as fallback for exact search
        
        # Fallback
        return self.default_backend
    
    async def add_items(
        self,
        vectors: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        backend_name: Optional[str] = None
    ) -> bool:
        """Add items to the specified backend."""
        try:
            backend = self.get_backend(backend_name)
            name = backend_name or self.default_backend
            
            start_time = asyncio.get_event_loop().time()
            success = await backend.add_items(vectors, ids, metadata)
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if success:
                self._update_stats(name, response_time)
            
            return success
            
        except Exception as e:
            print(f"Error adding items: {e}")
            return False
    
    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        backend_name: Optional[str] = None,
        use_fallback: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using the specified backend."""
        try:
            # Auto-choose backend if none specified
            if backend_name is None:
                backend_name = self.choose_backend(k=k, filters=filters)
            
            backend = self.get_backend(backend_name)
            name = backend_name
            
            start_time = asyncio.get_event_loop().time()
            results = await backend.search(query_vector, k, filters)
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            self._update_stats(name, response_time)
            
            # If no results and fallback is enabled, try other backends
            if not results and use_fallback and len(self.backends) > 1:
                for other_backend_name in self.backends:
                    if other_backend_name != name:
                        try:
                            other_backend = self.backends[other_backend_name]
                            fallback_results = await other_backend.search(query_vector, k, filters)
                            if fallback_results:
                                print(f"Fallback to {other_backend_name} returned {len(fallback_results)} results")
                                return fallback_results
                        except Exception as e:
                            print(f"Fallback to {other_backend_name} failed: {e}")
            
            return results
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    async def hybrid_search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        backend_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search across multiple backends."""
        if len(self.backends) < 2:
            return await self.search(query_vector, k, filters)
        
        # Auto-choose backends based on query characteristics
        if backend_weights is None:
            primary_backend = self.choose_backend(k=k, filters=filters)
            
            # Choose secondary backend based on primary
            if primary_backend == "hnsw":
                secondary_backend = "qdrant" if "qdrant" in self.backends else "faiss"
            elif primary_backend == "qdrant":
                secondary_backend = "hnsw" if "hnsw" in self.backends else "faiss"
            elif primary_backend == "faiss":
                secondary_backend = "hnsw" if "hnsw" in self.backends else "qdrant"
            else:
                secondary_backend = "hnsw" if "hnsw" in self.backends else "qdrant"
            
            # Adjust weights based on backend choice
            if primary_backend == "hnsw":
                backend_weights = {"hnsw": 0.7, secondary_backend: 0.3}
            elif primary_backend == "qdrant":
                backend_weights = {"qdrant": 0.7, secondary_backend: 0.3}
            elif primary_backend == "faiss":
                backend_weights = {"faiss": 0.7, secondary_backend: 0.3}
            else:
                backend_weights = {"hnsw": 0.7, "qdrant": 0.3}
        
        try:
            # Create search tasks for parallel execution
            tasks = {
                name: self.search(query_vector, k, filters, name, use_fallback=False)
                for name in self.backends if name in backend_weights
            }
            
            # Execute searches in parallel
            results_list = await asyncio.gather(*tasks.values())
            results_by_backend = {name: res for name, res in zip(tasks.keys(), results_list)}
            
            # Combine results using weighted scoring
            combined_results = self._combine_search_results(
                results_by_backend, 
                backend_weights, 
                k
            )
            
            return combined_results
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return await self.search(query_vector, k, filters)
    
    def _combine_search_results(
        self,
        results_by_backend: Dict[str, List[Dict[str, Any]]],
        weights: Dict[str, float],
        k: int
    ) -> List[Dict[str, Any]]:
        """Combine results from multiple backends using weighted scoring."""
        # Create item score map
        item_scores = {}
        item_metadata = {}
        
        for backend_name, results in results_by_backend.items():
            weight = weights.get(backend_name, 1.0)
            
            for result in results:
                item_id = result["item_id"]
                score = result["score"] * weight
                
                if item_id not in item_scores:
                    item_scores[item_id] = 0.0
                    item_metadata[item_id] = result["metadata"]
                
                item_scores[item_id] += score
        
        # Sort by combined score
        sorted_items = sorted(
            item_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top-k results
        combined_results = []
        for item_id, combined_score in sorted_items[:k]:
            combined_results.append({
                "item_id": item_id,
                "score": combined_score,
                "metadata": item_metadata[item_id],
                "source": "hybrid"
            })
        
        return combined_results
    
    def _update_stats(self, backend_name: str, response_time: float):
        """Update performance statistics."""
        self.stats["total_requests"] += 1
        self.stats["backend_usage"][backend_name] = self.stats["backend_usage"].get(backend_name, 0) + 1
        
        # Update average response time
        current_avg = self.stats["avg_response_time_ms"]
        total_requests = self.stats["total_requests"]
        self.stats["avg_response_time_ms"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    async def get_backend_stats(self, backend_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a specific backend or all backends."""
        if backend_name:
            backend = self.get_backend(backend_name)
            return await backend.get_stats()
        
        # Get stats for all backends
        all_stats = {}
        for name, backend in self.backends.items():
            all_stats[name] = await backend.get_stats()
        
        return all_stats
    
    async def get_manager_stats(self) -> Dict[str, Any]:
        """Get overall manager statistics."""
        backend_stats = await self.get_backend_stats()
        
        return {
            **self.stats,
            "available_backends": list(self.backends.keys()),
            "default_backend": self.default_backend,
            "backend_details": backend_stats
        }
    
    async def switch_default_backend(self, backend_name: str) -> bool:
        """Switch the default backend."""
        if backend_name not in self.backends:
            return False
        
        self.default_backend = backend_name
        return True
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all backends."""
        health_status = {}
        
        for backend_name, backend in self.backends.items():
            try:
                # Try to get stats as a health check
                await backend.get_stats()
                health_status[backend_name] = True
            except Exception as e:
                print(f"Health check failed for {backend_name}: {e}")
                health_status[backend_name] = False
        
        return health_status


# Global instance
vector_index_manager = VectorIndexManager()
