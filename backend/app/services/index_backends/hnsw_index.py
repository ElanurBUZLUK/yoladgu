"""
HNSW (Hierarchical Navigable Small World) index backend for fast approximate nearest neighbor search.
"""

import asyncio
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import hnswlib
except ImportError:
    hnswlib = None

from .base import BaseIndexBackend


class HNSWIndexBackend(BaseIndexBackend):
    """HNSW index backend using hnswlib."""
    
    def __init__(
        self, 
        vector_size: int,
        space: str = "ip",  # "ip" for cosine similarity (after L2 normalization)
        M: int = 16,  # Max number of connections per element
        ef_construction: int = 200,  # Construction time/accuracy trade-off
        ef_search: int = 128,  # Search time/accuracy trade-off
        max_elements: int = 100000,
        index_path: Optional[str] = None
    ):
        super().__init__(vector_size)
        
        if hnswlib is None:
            raise ImportError("hnswlib not available. Install with: pip install hnswlib")
        
        self.space = space
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_elements = max_elements
        self.index_path = index_path
        
        # HNSW index
        self.index = None
        self.id_to_metadata = {}  # id -> metadata mapping
        self.string_id_to_int = {}  # string id -> integer index mapping
        self.int_to_string_id = {}  # integer index -> string id mapping
        self.next_int_id = 0
        
        # Performance tracking
        self.stats = {
            "total_items": 0,
            "search_count": 0,
            "avg_search_time_ms": 0.0,
            "build_time_ms": 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize HNSW index."""
        try:
            if self.index_path and os.path.exists(self.index_path):
                # Load existing index
                await self._load_index()
            else:
                # Create new index
                await self._create_index()
            
            self._built = True
            return True
            
        except Exception as e:
            print(f"Error initializing HNSW index: {e}")
            return False
    
    async def _create_index(self):
        """Create new HNSW index."""
        loop = asyncio.get_event_loop()
        
        def _create():
            index = hnswlib.Index(space=self.space, dim=self.vector_size)
            index.init_index(
                max_elements=self.max_elements,
                M=self.M,
                ef_construction=self.ef_construction
            )
            index.set_ef(self.ef_search)
            return index
        
        start_time = asyncio.get_event_loop().time()
        self.index = await loop.run_in_executor(None, _create)
        self.stats["build_time_ms"] = (asyncio.get_event_loop().time() - start_time) * 1000
    
    async def _load_index(self):
        """Load existing HNSW index from file."""
        loop = asyncio.get_event_loop()
        
        def _load():
            index = hnswlib.Index(space=self.space, dim=self.vector_size)
            index.load_index(self.index_path)
            index.set_ef(self.ef_search)
            return index
        
        self.index = await loop.run_in_executor(None, _load)
        
        # Load metadata and ID mappings if available
        metadata_path = self.index_path.replace('.index', '.metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.id_to_metadata = data.get('metadata', {})
                self.string_id_to_int = data.get('string_id_to_int', {})
                self.int_to_string_id = data.get('int_to_string_id', {})
                self.next_int_id = data.get('next_int_id', 0)
                self.stats["total_items"] = len(self.id_to_metadata)
    
    async def add_items(
        self, 
        vectors: np.ndarray, 
        ids: List[str], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Add items to HNSW index."""
        try:
            if not self._validate_vector(vectors):
                raise ValueError(f"Vector dimensions must be {self.vector_size}")
            
            # Normalize vectors for cosine similarity
            if self.space == "ip":
                vectors = self._normalize_vector(vectors)
            
            # Convert to float32 for HNSW
            vectors = vectors.astype(np.float32)
            
            loop = asyncio.get_event_loop()
            
            def _add():
                # Create integer IDs for HNSW
                int_ids = []
                for string_id in ids:
                    if string_id not in self.string_id_to_int:
                        self.string_id_to_int[string_id] = self.next_int_id
                        self.int_to_string_id[self.next_int_id] = string_id
                        self.next_int_id += 1
                    int_ids.append(self.string_id_to_int[string_id])
                
                # Add vectors to index
                self.index.add_items(vectors, int_ids)
                
                # Store metadata
                if metadata:
                    for i, item_id in enumerate(ids):
                        self.id_to_metadata[item_id] = metadata[i] if i < len(metadata) else {}
                        self.stats["total_items"] += 1
            
            await loop.run_in_executor(None, _add)
            
            # Periodic save (every 1000 items or if index_path is set)
            if self.index_path and self.stats["total_items"] % 1000 == 0:
                await self.save_index()
                print(f"HNSW index auto-saved after {self.stats['total_items']} items")
            
            return True
            
        except Exception as e:
            print(f"Error adding items to HNSW index: {e}")
            return False
    
    async def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using HNSW."""
        try:
            if not self._validate_vector(query_vector):
                raise ValueError(f"Query vector dimensions must be {self.vector_size}")
            
            # Normalize query vector for cosine similarity
            if self.space == "ip":
                query_vector = self._normalize_vector(query_vector)
            
            # Convert to float32
            query_vector = query_vector.astype(np.float32)
            
            loop = asyncio.get_event_loop()
            
            def _search():
                # Perform search
                indices, distances = self.index.knn_query(query_vector, k=k)
                return indices, distances
            
            start_time = asyncio.get_event_loop().time()
            indices, distances = await loop.run_in_executor(None, _search)
            search_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Update stats
            self.stats["search_count"] += 1
            self.stats["avg_search_time_ms"] = (
                (self.stats["avg_search_time_ms"] * (self.stats["search_count"] - 1) + search_time) 
                / self.stats["search_count"]
            )
            
            # Format results
            results = []
            for i in range(len(indices)):
                for j in range(len(indices[i])):
                    int_idx = indices[i][j]
                    distance = distances[i][j]
                    
                    # Convert integer index back to string ID
                    string_id = self.int_to_string_id.get(int_idx, str(int_idx))
                    
                    # Convert distance to similarity score (for cosine)
                    if self.space == "ip":
                        score = 1.0 - distance  # Convert distance to similarity
                    else:
                        score = 1.0 / (1.0 + distance)  # Convert L2 distance to similarity
                    
                    # Get metadata
                    metadata = self.id_to_metadata.get(string_id, {})
                    
                    results.append({
                        "item_id": string_id,
                        "score": float(score),
                        "distance": float(distance),
                        "metadata": metadata
                    })
            
            # Apply filters if specified
            if filters:
                results = self._apply_filters(results, filters)
            
            # Sort by score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results[:k]
            
        except Exception as e:
            print(f"Error searching HNSW index: {e}")
            return []
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply metadata filters to search results."""
        filtered_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            include = True
            
            for key, value in filters.items():
                if key in metadata:
                    if isinstance(value, list):
                        if not any(v in metadata[key] for v in value):
                            include = False
                            break
                    elif metadata[key] != value:
                        include = False
                        break
                else:
                    include = False
                    break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    async def delete_items(self, ids: List[str]) -> bool:
        """Delete items from HNSW index (not supported by hnswlib, would need rebuild)."""
        # HNSW doesn't support deletion, would need to rebuild index
        # For now, just remove from metadata and mappings
        for item_id in ids:
            if item_id in self.string_id_to_int:
                int_id = self.string_id_to_int[item_id]
                del self.int_to_string_id[int_id]
                del self.string_id_to_int[item_id]
            self.id_to_metadata.pop(item_id, None)
            self.stats["total_items"] = max(0, self.stats["total_items"] - 1)
        
        print("Warning: HNSW index doesn't support deletion. Items removed from metadata only.")
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            **self.stats,
            "index_type": "HNSW",
            "space": self.space,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "max_elements": self.max_elements,
            "current_elements": self.stats["total_items"]
        }
    
    async def save_index(self, path: Optional[str] = None) -> bool:
        """Save index to disk."""
        try:
            save_path = path or self.index_path
            if not save_path:
                raise ValueError("No save path specified")
            
            loop = asyncio.get_event_loop()
            
            def _save():
                # Save index
                self.index.save_index(save_path)
                
                # Save metadata and ID mappings
                metadata_path = save_path.replace('.index', '.metadata.pkl')
                data = {
                    'metadata': self.id_to_metadata,
                    'string_id_to_int': self.string_id_to_int,
                    'int_to_string_id': self.int_to_string_id,
                    'next_int_id': self.next_int_id
                }
                with open(metadata_path, 'wb') as f:
                    pickle.dump(data, f)
            
            await loop.run_in_executor(None, _save)
            return True
            
        except Exception as e:
            print(f"Error saving HNSW index: {e}")
            return False
