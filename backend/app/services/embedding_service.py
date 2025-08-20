from __future__ import annotations
from typing import List, Optional, Dict, Any
import os, asyncio, logging
import httpx

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Enhanced async embedding service with domain-specific model selection.
    ENV:
      - OPENAI_API_KEY (zorunlu)
      - EMBEDDING_MODEL (default: text-embedding-3-small)
      - EMBEDDING_DIM (default: 1536)
      - LOCAL_EMBEDDING_MODEL (optional: bge-small-en)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        dim: Optional[int] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.provider = provider.lower()
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.dim = int(dim or os.getenv("EMBEDDING_DIM", "1536"))
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Domain-specific model mapping
        self.domain_models = {
            "english": {
                "default": "text-embedding-3-small",
                "grammar": "text-embedding-3-small",
                "vocabulary": "text-embedding-3-small",
                "cloze": "text-embedding-3-small"
            },
            "math": {
                "default": "text-embedding-3-large",  # Higher dimension for math concepts
                "concepts": "text-embedding-3-large",
                "problems": "text-embedding-3-large",
                "solutions": "text-embedding-3-large"
            },
            "cefr": {
                "default": "text-embedding-3-small",
                "rubrics": "text-embedding-3-small",
                "assessments": "text-embedding-3-small"
            }
        }
        
        # Model dimension mapping
        self.model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "bge-small-en": 384,
            "bge-base-en": 768,
            "bge-large-en": 1024
        }

    async def get_embedding(
        self, 
        text: str, 
        domain: str = "general",
        content_type: str = "default"
    ) -> List[float]:
        """Get embedding for text with domain-specific model selection"""
        try:
            # Select appropriate model for domain and content type
            model = self._select_model_for_domain(domain, content_type)
            
            # Update dimension based on selected model
            self.dim = self.model_dimensions.get(model, self.dim)
            
            logger.info(f"Generating embedding for {domain}/{content_type} using model {model}")
            
            if self.provider == "openai":
                return await self._embed_openai([text], model)
            else:
                raise RuntimeError(f"Unknown embedding provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    async def embed_texts(
        self, 
        texts: List[str],
        domain: str = "general",
        content_type: str = "default"
    ) -> List[List[float]]:
        """Embed multiple texts with domain-specific model selection"""
        if not texts:
            return []
            
        try:
            # Select appropriate model for domain and content type
            model = self._select_model_for_domain(domain, content_type)
            
            # Update dimension based on selected model
            self.dim = self.model_dimensions.get(model, self.dim)
            
            logger.info(f"Generating embeddings for {len(texts)} texts in {domain}/{content_type} using model {model}")
            
            if self.provider == "openai":
                return await self._embed_openai(texts, model)
            else:
                raise RuntimeError(f"Unknown embedding provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            raise

    def _select_model_for_domain(self, domain: str, content_type: str) -> str:
        """Select the most appropriate embedding model for domain and content type"""
        try:
            # Check if domain exists in mapping
            if domain in self.domain_models:
                domain_models = self.domain_models[domain]
                
                # Check if content type exists
                if content_type in domain_models:
                    selected_model = domain_models[content_type]
                else:
                    # Use default for domain
                    selected_model = domain_models.get("default", self.model)
            else:
                # Use global default
                selected_model = self.model
            
            # Validate model availability
            if selected_model not in self.model_dimensions:
                logger.warning(f"Model {selected_model} not found in dimensions mapping, using default")
                selected_model = self.model
            
            return selected_model
            
        except Exception as e:
            logger.error(f"Error selecting model for domain: {e}")
            return self.model

    async def _embed_openai(self, texts: List[str], model: str) -> List[List[float]]:
        """Generate embeddings using OpenAI API with enhanced error handling"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "input": texts}

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                
                # Handle rate limiting and server errors
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait_time = min(1.0 * attempt, 4.0)
                    logger.warning(f"Rate limit/server error (attempt {attempt}/{self.max_retries}), waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    if attempt < self.max_retries:
                        continue
                
                resp.raise_for_status()
                data = resp.json()
                
                # Extract embeddings
                vectors = [item["embedding"] for item in data.get("data", [])]
                
                # Validate dimensions
                expected_dim = self.model_dimensions.get(model, self.dim)
                for i, v in enumerate(vectors):
                    if len(v) != expected_dim:
                        raise RuntimeError(
                            f"Embedding dimension mismatch for text {i}. "
                            f"Expected: {expected_dim}, Got: {len(v)}"
                        )
                
                logger.info(f"Successfully generated {len(vectors)} embeddings using {model}")
                return vectors
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise RuntimeError("Invalid OpenAI API key")
                elif e.response.status_code == 400:
                    raise RuntimeError(f"Invalid request to OpenAI API: {e.response.text}")
                elif e.response.status_code >= 500:
                    logger.warning(f"OpenAI server error (attempt {attempt}/{self.max_retries}): {e}")
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(0.5 * attempt, 2.0))
                        continue
                    raise
                else:
                    raise RuntimeError(f"OpenAI API error: {e}")
                    
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(min(0.5 * attempt, 2.0))
                    continue
                raise
        
        raise RuntimeError(f"Failed to generate embeddings after {self.max_retries} attempts")

    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get embedding service statistics and model information"""
        try:
            stats = {
                "provider": self.provider,
                "current_model": self.model,
                "embedding_dimension": self.dim,
                "max_retries": self.max_retries,
                "timeout": self.timeout,
                "available_models": list(self.model_dimensions.keys()),
                "domain_models": self.domain_models,
                "model_dimensions": self.model_dimensions
            }
            
            # Add cost information for OpenAI models
            if self.provider == "openai":
                stats["cost_per_1k_tokens"] = {
                    "text-embedding-3-small": 0.00002,
                    "text-embedding-3-large": 0.00013
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting embedding statistics: {e}")
            return {"error": str(e)}

    async def validate_embedding_quality(
        self, 
        embedding: List[float], 
        expected_dim: Optional[int] = None
    ) -> Dict[str, Any]:
        """Validate embedding quality and characteristics"""
        try:
            if not embedding:
                return {"valid": False, "error": "Empty embedding"}
            
            # Check dimension
            actual_dim = len(embedding)
            expected_dim = expected_dim or self.dim
            
            if actual_dim != expected_dim:
                return {
                    "valid": False, 
                    "error": f"Dimension mismatch: expected {expected_dim}, got {actual_dim}"
                }
            
            # Check for all-zero vectors
            if all(x == 0.0 for x in embedding):
                return {"valid": False, "error": "All-zero embedding vector"}
            
            # Check for NaN or infinite values
            if any(not (x == x) or not (x != float('inf') and x != float('-inf')) for x in embedding):
                return {"valid": False, "error": "Invalid values in embedding (NaN or inf)"}
            
            # Calculate statistics
            min_val = min(embedding)
            max_val = max(embedding)
            mean_val = sum(embedding) / len(embedding)
            variance = sum((x - mean_val) ** 2 for x in embedding) / len(embedding)
            
            # Quality indicators
            quality_score = 1.0
            
            # Penalize very small variance (indicates poor differentiation)
            if variance < 0.01:
                quality_score *= 0.8
            
            # Penalize extreme values
            if abs(min_val) > 10 or abs(max_val) > 10:
                quality_score *= 0.9
            
            return {
                "valid": True,
                "dimension": actual_dim,
                "min_value": min_val,
                "max_value": max_val,
                "mean_value": mean_val,
                "variance": variance,
                "quality_score": quality_score,
                "recommendations": self._get_quality_recommendations(quality_score, variance, min_val, max_val)
            }
            
        except Exception as e:
            logger.error(f"Error validating embedding quality: {e}")
            return {"valid": False, "error": str(e)}

    def _get_quality_recommendations(
        self, 
        quality_score: float, 
        variance: float, 
        min_val: float, 
        max_val: float
    ) -> List[str]:
        """Get recommendations for improving embedding quality"""
        recommendations = []
        
        if quality_score < 0.8:
            recommendations.append("Consider using a different embedding model")
        
        if variance < 0.01:
            recommendations.append("Text may be too similar or generic - consider more diverse input")
        
        if abs(min_val) > 10 or abs(max_val) > 10:
            recommendations.append("Consider normalizing embeddings to [-1, 1] range")
        
        if quality_score >= 0.9:
            recommendations.append("Embedding quality is good")
        
        return recommendations

    async def batch_validate_embeddings(
        self, 
        embeddings: List[List[float]], 
        expected_dim: Optional[int] = None
    ) -> Dict[str, Any]:
        """Validate a batch of embeddings"""
        try:
            if not embeddings:
                return {"valid": False, "error": "No embeddings provided"}
            
            validation_results = []
            valid_count = 0
            total_dim = 0
            
            for i, embedding in enumerate(embeddings):
                result = await self.validate_embedding_quality(embedding, expected_dim)
                validation_results.append({
                    "index": i,
                    "result": result
                })
                
                if result.get("valid", False):
                    valid_count += 1
                    total_dim += result.get("dimension", 0)
            
            # Calculate batch statistics
            batch_stats = {
                "total_embeddings": len(embeddings),
                "valid_embeddings": valid_count,
                "invalid_embeddings": len(embeddings) - valid_count,
                "success_rate": valid_count / len(embeddings) if embeddings else 0,
                "average_dimension": total_dim / valid_count if valid_count > 0 else 0
            }
            
            return {
                "batch_valid": valid_count == len(embeddings),
                "batch_statistics": batch_stats,
                "individual_results": validation_results
            }
            
        except Exception as e:
            logger.error(f"Error in batch validation: {e}")
            return {"error": str(e)}

embedding_service = EmbeddingService()
