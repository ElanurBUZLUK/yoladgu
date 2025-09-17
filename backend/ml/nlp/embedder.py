"""
E5-Large-v2 Embedding Service for Hybrid Search
Supports query: and passage: prefixes for optimal retrieval
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union
import structlog

logger = structlog.get_logger()

class E5EmbeddingService:
    """E5-Large-v2 embedding service with query/passage prefixes"""
    
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the E5 model and tokenizer"""
        try:
            logger.info("Loading E5 model: %s", self.model_name)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self._initialized = True
            logger.info("E5 model loaded successfully on device: %s", self.device)
            return True
            
        except Exception as e:
            logger.error("Failed to load E5 model: %s", e)
            return False
    
    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling with attention mask"""
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(1) / attention_mask.sum(1)[..., None]
    
    @torch.no_grad()
    def encode_queries(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode queries with 'query:' prefix"""
        if not self._initialized:
            raise RuntimeError("E5 model not initialized")
        
        # Add query prefix
        prefixed_texts = [f"query: {text}" for text in texts]
        
        embeddings = []
        for i in range(0, len(prefixed_texts), batch_size):
            batch_texts = prefixed_texts[i:i + batch_size]
            
            # Tokenize
            batch = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            outputs = self.model(**batch)
            emb = self._pool(outputs.last_hidden_state, batch["attention_mask"])
            
            # L2 normalize
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb.cpu().numpy())
        
        return np.vstack(embeddings)
    
    @torch.no_grad()
    def encode_passages(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode passages with 'passage:' prefix"""
        if not self._initialized:
            raise RuntimeError("E5 model not initialized")
        
        # Add passage prefix
        prefixed_texts = [f"passage: {text}" for text in texts]
        
        embeddings = []
        for i in range(0, len(prefixed_texts), batch_size):
            batch_texts = prefixed_texts[i:i + batch_size]
            
            # Tokenize
            batch = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            outputs = self.model(**batch)
            emb = self._pool(outputs.last_hidden_state, batch["attention_mask"])
            
            # L2 normalize
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode_single_query(self, text: str) -> np.ndarray:
        """Encode a single query"""
        return self.encode_queries([text])[0]
    
    def encode_single_passage(self, text: str) -> np.ndarray:
        """Encode a single passage"""
        return self.encode_passages([text])[0]
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension"""
        return 1024  # E5-Large-v2 dimension
    
    @property
    def is_initialized(self) -> bool:
        """Check if model is initialized"""
        return self._initialized

# Global instance
e5_embedder = E5EmbeddingService()
