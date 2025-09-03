"""
Qdrant vector database backend implementation.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, PointIdsList

from .base import BaseIndexBackend


class QdrantIndexBackend(BaseIndexBackend):
    """Qdrant vector database backend."""
    
    def __init__(
        self, 
        vector_size: int,
        collection_name: str = "vectors",
        url: str = "http://localhost:6333",
        **kwargs
    ):
        super().__init__(vector_size)
        
        self.collection_name = collection_name
        self.url = url
        self.client = None
        
        # Performance tracking
        self.stats = {
            "total_items": 0,
            "search_count": 0,
            "avg_search_time_ms": 0.0,
            "collection_created": False
        }
    
    async def _get_client(self) -> QdrantClient:
        """Get Qdrant client (lazy initialization)."""
        if self.client is None:
            self.client = QdrantClient(url=self.url)
        return self.client
    
    async def initialize(self) -> bool:
        """Initialize Qdrant collection."""
        try:
            client = await self._get_client()
            
            # Check if collection exists
            collections = client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                self.stats["collection_created"] = True
                print(f"Created Qdrant collection: {self.collection_name}")
            
            # Get collection info for stats
            collection_info = client.get_collection(self.collection_name)
            self.stats["total_items"] = collection_info.points_count
            
            self._built = True
            return True
            
        except Exception as e:
            print(f"Error initializing Qdrant collection: {e}")
            return False
    
    async def add_items(
        self, 
        vectors: np.ndarray, 
        ids: List[str], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Add items to Qdrant collection."""
        try:
            if not self._validate_vector(vectors):
                raise ValueError(f"Vector dimensions must be {self.vector_size}")
            
            # Normalize vectors for cosine similarity
            vectors = self._normalize_vector(vectors)
            
            client = await self._get_client()
            
            # Prepare points
            points = []
            for i, (vector, item_id) in enumerate(zip(vectors, ids)):
                payload = metadata[i] if metadata and i < len(metadata) else {}
                payload["id"] = item_id  # Ensure ID is in payload
                
                point = PointStruct(
                    id=item_id,
                    vector=vector.tolist(),
                    payload=payload
                )
                points.append(point)
            
            # Add points to collection
            client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            self.stats["total_items"] += len(points)
            return True
            
        except Exception as e:
            print(f"Error adding items to Qdrant: {e}")
            return False
    
    async def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant."""
        try:
            if not self._validate_vector(query_vector):
                raise ValueError(f"Query vector dimensions must be {self.vector_size}")
            
            # Normalize query vector for cosine similarity
            query_vector = self._normalize_vector(query_vector)
            
            client = await self._get_client()
            
            # Build filter conditions with proper MatchAny/MatchValue usage
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Multiple values (OR condition) - use MatchAny
                        conditions.append(FieldCondition(
                            key=key, 
                            match=MatchAny(any=value)
                        ))
                    else:
                        # Single value - use MatchValue
                        conditions.append(FieldCondition(
                            key=key, 
                            match=MatchValue(value=value)
                        ))
                
                if conditions:
                    # Use 'must' for AND logic (more restrictive, usually what we want)
                    query_filter = Filter(must=conditions)
            
            # Perform search
            start_time = asyncio.get_event_loop().time()
            
            # Use executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(),
                    query_filter=query_filter,
                    limit=k
                )
            )
            
            search_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Update stats
            self.stats["search_count"] += 1
            self.stats["avg_search_time_ms"] = (
                (self.stats["avg_search_time_ms"] * (self.stats["search_count"] - 1) + search_time) 
                / self.stats["search_count"]
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "item_id": result.id,
                    "score": result.score,
                    "distance": 1.0 - result.score,  # Convert similarity to distance
                    "metadata": result.payload
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return []
    
    async def delete_items(self, ids: List[str]) -> bool:
        """Delete items from Qdrant collection."""
        try:
            client = await self._get_client()
            
            # Delete points using proper PointIdsList selector
            client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids)
            )
            
            self.stats["total_items"] = max(0, self.stats["total_items"] - len(ids))
            return True
            
        except Exception as e:
            print(f"Error deleting items from Qdrant: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Qdrant collection statistics."""
        try:
            if self.client:
                collection_info = self.client.get_collection(self.collection_name)
                self.stats["total_items"] = collection_info.points_count
        except:
            pass
        
        return {
            **self.stats,
            "index_type": "Qdrant",
            "collection_name": self.collection_name,
            "url": self.url
        }
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get detailed collection information."""
        try:
            client = await self._get_client()
            collection_info = client.get_collection(self.collection_name)
            
            return {
                "name": collection_info.name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "segments_count": collection_info.segments_count,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance)
                }
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
