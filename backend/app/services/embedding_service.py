from typing import List, Dict, Any, Optional, Union
import logging
import asyncio
import time
from datetime import datetime
import hashlib

import openai
from openai import AsyncOpenAI
import numpy as np

from app.core.config import settings
from app.core.cache import cache_service

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Production-ready embedding service with OpenAI API integration"""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model_name = settings.embedding_model
        self.embedding_dimension = settings.pgvector_dim
        self.max_retries = settings.max_retry_attempts
        self.retry_delay = 1.0
        self.cache_ttl = settings.embedding_cache_ttl
        self.batch_size = settings.embedding_batch_size
        self.rate_limit_delay = settings.embedding_rate_limit_delay
        
        # Cost tracking
        self.cost_per_1k_tokens = 0.00002  # $0.00002 per 1K tokens for text-embedding-3-small
        
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.embedding_dimension
        
        # Check cache first
        cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
        cached_embedding = await cache_service.get(cache_key)
        if cached_embedding:
            logger.debug(f"Cache hit for embedding: {text[:50]}...")
            return cached_embedding
        
        try:
            # Get embedding from OpenAI
            embedding = await self._get_openai_embedding(text)
            
            # Cache the result
            await cache_service.set(cache_key, embedding, self.cache_ttl)
            
            logger.debug(f"Generated embedding for: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding for text: {e}")
            # Return fallback embedding
            return await self._get_fallback_embedding(text)
    
    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently"""
        
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return []
        
        # Check cache for all texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(valid_texts):
            cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
            cached_embedding = await cache_service.get(cache_key)
            
            if cached_embedding:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Get embeddings for uncached texts
        if uncached_texts:
            try:
                new_embeddings = await self._get_openai_batch_embeddings(uncached_texts)
                
                # Update embeddings list and cache
                for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                    original_index = uncached_indices[i]
                    embeddings[original_index] = embedding
                    
                    # Cache the new embedding
                    cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
                    await cache_service.set(cache_key, embedding, self.cache_ttl)
                    
            except Exception as e:
                logger.error(f"Error getting batch embeddings: {e}")
                # Use fallback for failed texts
                for i, text in enumerate(uncached_texts):
                    original_index = uncached_indices[i]
                    fallback_embedding = await self._get_fallback_embedding(text)
                    embeddings[original_index] = fallback_embedding
        
        return embeddings
    
    async def _get_openai_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API with retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                response = await self.openai_client.embeddings.create(
                    model=self.model_name,
                    input=text,
                    encoding_format="float"
                )
                
                embedding = response.data[0].embedding
                
                # Log cost
                usage = response.usage
                cost = (usage.total_tokens / 1000) * self.cost_per_1k_tokens
                logger.debug(f"OpenAI embedding cost: ${cost:.6f} for {usage.total_tokens} tokens")
                
                return embedding
                
            except openai.RateLimitError:
                wait_time = (2 ** attempt) * self.retry_delay
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                await asyncio.sleep(wait_time)
                
            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                logger.error(f"Unexpected error getting embedding: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)
        
        raise Exception("Failed to get embedding after all retries")
    
    async def _get_openai_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts from OpenAI API"""
        
        embeddings = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = await self.openai_client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float"
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                # Log cost
                usage = response.usage
                cost = (usage.total_tokens / 1000) * self.cost_per_1k_tokens
                logger.debug(f"OpenAI batch embedding cost: ${cost:.6f} for {usage.total_tokens} tokens")
                
                # Rate limiting
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                # Use fallback for this batch
                for text in batch:
                    fallback_embedding = await self._get_fallback_embedding(text)
                    embeddings.append(fallback_embedding)
        
        return embeddings
    
    async def _get_fallback_embedding(self, text: str) -> List[float]:
        """Generate fallback embedding when OpenAI fails"""
        
        # Try local embedding model if configured
        if settings.local_embedding_model:
            try:
                return await self._get_local_embedding(text)
            except Exception as e:
                logger.warning(f"Local embedding failed, using hash fallback: {e}")
        
        # Simple hash-based embedding as fallback
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to embedding-like vector
        embedding = []
        for i in range(0, len(text_hash), 2):
            if len(embedding) >= self.embedding_dimension:
                break
            hex_val = text_hash[i:i+2]
            embedding.append(float(int(hex_val, 16)) / 255.0)
        
        # Pad or truncate to required dimension
        while len(embedding) < self.embedding_dimension:
            embedding.append(0.0)
        
        embedding = embedding[:self.embedding_dimension]
        
        logger.warning(f"Using hash fallback embedding for: {text[:50]}...")
        return embedding
    
    async def _get_local_embedding(self, text: str) -> List[float]:
        """Generate embedding using local model (SentenceTransformer)"""
        
        try:
            # Lazy import to avoid dependency issues
            from sentence_transformers import SentenceTransformer
            
            # Initialize model if not already done
            if not hasattr(self, '_local_model'):
                self._local_model = SentenceTransformer(settings.local_embedding_model)
                logger.info(f"Initialized local embedding model: {settings.local_embedding_model}")
            
            # Generate embedding
            embedding = self._local_model.encode(text, convert_to_tensor=False)
            
            # Ensure correct dimension
            if len(embedding) != self.embedding_dimension:
                logger.warning(f"Local model dimension mismatch: {len(embedding)} vs {self.embedding_dimension}")
                # Pad or truncate to match pgvector dimension
                if len(embedding) < self.embedding_dimension:
                    embedding = list(embedding) + [0.0] * (self.embedding_dimension - len(embedding))
                else:
                    embedding = list(embedding[:self.embedding_dimension])
            
            logger.debug(f"Generated local embedding for: {text[:50]}...")
            return list(embedding)
            
        except ImportError:
            logger.error("SentenceTransformer not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error generating local embedding: {e}")
            raise
    
    async def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        
        if not embedding1 or not embedding2:
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    async def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find most similar embeddings to query"""
        
        if not query_embedding or not candidate_embeddings:
            return []
        
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = await self.calculate_similarity(query_embedding, candidate)
            similarities.append({
                "index": i,
                "similarity": similarity,
                "embedding": candidate
            })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:top_k]
    
    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get embedding service statistics"""
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_retries": self.max_retries,
            "batch_size": self.batch_size,
            "cache_ttl": self.cache_ttl,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "rate_limit_delay": self.rate_limit_delay
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check embedding service health"""
        
        try:
            # Test with a simple text
            test_text = "This is a test for embedding service health check."
            embedding = await self.get_embedding(test_text)
            
            return {
                "status": "healthy",
                "embedding_dimension": len(embedding),
                "test_successful": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "test_successful": False,
                "timestamp": datetime.now().isoformat()
            }


# Global instance
embedding_service = EmbeddingService()
