"""
Base Embedding Service
Temel embedding işlemleri için base class
"""

import structlog
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

logger = structlog.get_logger()


class EmbeddingService:
    """Base embedding service class"""
    
    def __init__(self):
        self.model = None
        self.model_name = "paraphrase-MiniLM-L6-v2"
        self.embedding_dim = 384
        
    def _load_model(self):
        """Model yükle"""
        try:
            if self.model is None:
                logger.info("loading_embedding_model", model=self.model_name)
                self.model = SentenceTransformer(self.model_name)
                logger.info("embedding_model_loaded", model=self.model_name)
        except Exception as e:
            logger.error("embedding_model_load_error", error=str(e))
            raise
    
    def compute_embedding(self, text: str) -> List[float]:
        """Tek metin için embedding hesapla"""
        if self.model is None:
            self._load_model()
        
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error("compute_embedding_error", error=str(e))
            raise
    
    def compute_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding hesapla"""
        if self.model is None:
            self._load_model()
        
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error("compute_embeddings_batch_error", error=str(e))
            raise
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """İki metin arasındaki semantic benzerlik"""
        try:
            embedding1 = self.compute_embedding(text1)
            embedding2 = self.compute_embedding(text2)
            
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)
        except Exception as e:
            logger.error("semantic_similarity_error", error=str(e))
            raise


# Global instance
embedding_service = EmbeddingService() 