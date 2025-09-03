"""
Base abstract class for vector index backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class BaseIndexBackend(ABC):
    """Abstract base class for vector index backends."""
    
    def __init__(self, vector_size: int, **kwargs):
        self.vector_size = vector_size
        self._built = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the index backend."""
        pass
    
    @abstractmethod
    async def add_items(
        self, 
        vectors: np.ndarray, 
        ids: List[str], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Add items to the index."""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete_items(self, ids: List[str]) -> bool:
        """Delete items from the index."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        pass
    
    @property
    def is_built(self) -> bool:
        """Check if index is built and ready."""
        return self._built
    
    def _validate_vector(self, vector: np.ndarray) -> bool:
        """Validate vector dimensions."""
        if vector.ndim == 1:
            return vector.shape[0] == self.vector_size
        elif vector.ndim == 2:
            return vector.shape[1] == self.vector_size
        return False
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        if vector.ndim == 1:
            norm = np.linalg.norm(vector)
            if norm > 0:
                return vector / norm
            return vector
        elif vector.ndim == 2:
            norms = np.linalg.norm(vector, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            return vector / norms
        return vector
