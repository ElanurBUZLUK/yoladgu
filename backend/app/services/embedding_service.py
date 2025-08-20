from __future__ import annotations
from typing import List, Optional, Dict, Any, Union
import os, asyncio, logging
import httpx
import time
from datetime import datetime
import json

import openai
from openai import AsyncOpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class MonitoringEmbeddingService:
    """Monitoring wrapper for embedding service"""
    
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_used": 0,
            "average_response_time": 0.0,
            "total_response_time": 0.0,
            "last_request": None,
            "error_types": {}
        }
    
    async def get_embedding(self, text: str, domain: str = "general", content_type: str = "text") -> List[float]:
        start_time = time.time()
        try:
            result = await self.embedding_service.get_embedding(text, domain, content_type)
            response_time = time.time() - start_time
            
            # Update stats
            self.stats["total_requests"] += 1
            self.stats["successful_requests"] += 1
            self.stats["total_response_time"] += response_time
            self.stats["average_response_time"] = (
                self.stats["total_response_time"] / self.stats["total_requests"]
            )
            self.stats["last_request"] = datetime.utcnow().isoformat()
            
            logger.debug(f"✅ Embedding generated successfully in {response_time:.2f}s")
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.stats["total_requests"] += 1
            self.stats["failed_requests"] += 1
            self.stats["total_response_time"] += response_time
            
            error_type = type(e).__name__
            self.stats["error_types"][error_type] = self.stats["error_types"].get(error_type, 0) + 1
            
            logger.error(f"❌ Embedding failed: {e}")
            raise
    
    async def embed_texts(self, texts: List[str], domain: str = "general") -> List[List[float]]:
        start_time = time.time()
        try:
            result = await self.embedding_service.embed_texts(texts, domain)
            response_time = time.time() - start_time
            
            # Update stats
            self.stats["total_requests"] += 1
            self.stats["successful_requests"] += 1
            self.stats["total_response_time"] += response_time
            self.stats["average_response_time"] = (
                self.stats["total_response_time"] / self.stats["total_requests"]
            )
            self.stats["last_request"] = datetime.utcnow().isoformat()
            
            logger.info(f"✅ Batch embedding generated for {len(texts)} texts in {response_time:.2f}s")
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.stats["total_requests"] += 1
            self.stats["failed_requests"] += 1
            self.stats["total_response_time"] += response_time
            
            error_type = type(e).__name__
            self.stats["error_types"][error_type] = self.stats["error_types"].get(error_type, 0) + 1
            
            logger.error(f"❌ Batch embedding failed: {e}")
            raise


class EmbeddingService:
    """Enhanced embedding service with fallback mechanisms and monitoring"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Domain-specific models
        self.domain_models = {
            "english": "text-embedding-3-small",  # Fast and efficient for English
            "math": "text-embedding-3-large",     # High quality for Math
            "cefr": "text-embedding-3-small",     # Efficient for CEFR assessment
            "general": "text-embedding-3-small"   # Default model
        }
        
        # Model dimensions
        self.model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        
        # Fallback configuration
        self.fallback_config = {
            "use_local_fallback": True,
            "local_model": "all-MiniLM-L6-v2",
            "max_retries": 3,
            "retry_delay": 1.0
        }
        
        # Performance tracking
        self.performance_metrics = {
            "total_embeddings_generated": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "fallback_used": 0,
            "average_generation_time": 0.0,
            "total_generation_time": 0.0,
            "model_usage": {},
            "error_counts": {}
        }
    
    async def get_embedding(
        self, 
        text: str, 
        domain: str = "general", 
        content_type: str = "text"
    ) -> List[float]:
        """Get embedding with fallback mechanisms"""
        
        try:
            return await self._get_embedding_with_fallback(text, domain, content_type)
        except Exception as e:
            logger.error(f"❌ All embedding methods failed for text: {e}")
            # Return zero vector as last resort
            model = self.domain_models.get(domain, "text-embedding-3-small")
            dimension = self.model_dimensions.get(model, 1536)
            return [0.0] * dimension
    
    async def _get_embedding_with_fallback(
        self, 
        text: str, 
        domain: str, 
        content_type: str
    ) -> List[float]:
        """Get embedding with multiple fallback strategies"""
        
        start_time = time.time()
        
        # Method 1: Primary OpenAI API
        try:
            embedding = await self._embed_openai(text, domain)
            self._update_performance_metrics("openai", time.time() - start_time, True)
            return embedding
        except Exception as e:
            logger.warning(f"⚠️ OpenAI embedding failed: {e}")
        
        # Method 2: Retry with exponential backoff
        for attempt in range(self.fallback_config["max_retries"]):
            try:
                await asyncio.sleep(self.fallback_config["retry_delay"] * (2 ** attempt))
                embedding = await self._embed_openai(text, domain)
                self._update_performance_metrics("openai_retry", time.time() - start_time, True)
                logger.info(f"✅ OpenAI embedding succeeded on retry {attempt + 1}")
                return embedding
            except Exception as e:
                logger.warning(f"⚠️ OpenAI retry {attempt + 1} failed: {e}")
        
        # Method 3: Local fallback model
        if self.fallback_config["use_local_fallback"]:
            try:
                embedding = await self._get_local_embedding(text)
                self._update_performance_metrics("local_fallback", time.time() - start_time, True)
                self.performance_metrics["fallback_used"] += 1
                logger.info("✅ Used local fallback embedding")
                return embedding
            except Exception as e:
                logger.error(f"❌ Local fallback failed: {e}")
        
        # Method 4: Simple hash-based embedding (last resort)
        try:
            embedding = self._get_hash_based_embedding(text)
            self._update_performance_metrics("hash_fallback", time.time() - start_time, True)
            logger.warning("⚠️ Using hash-based fallback embedding")
            return embedding
        except Exception as e:
            logger.error(f"❌ Hash-based fallback failed: {e}")
            raise
    
    async def _embed_openai(self, text: str, domain: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        
        try:
            # Select appropriate model for domain
            model = self._select_model_for_domain(domain)
            
            # Prepare text for embedding
            processed_text = self._preprocess_text_for_embedding(text, domain)
            
            # Generate embedding
            response = await self.client.embeddings.create(
                model=model,
                input=processed_text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Validate embedding quality
            if not self._validate_embedding_quality(embedding):
                raise ValueError("Generated embedding failed quality validation")
            
            logger.debug(f"✅ OpenAI embedding generated with model {model}")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ OpenAI embedding failed: {e}")
            raise
    
    def _select_model_for_domain(self, domain: str) -> str:
        """Select appropriate embedding model for domain"""
        
        return self.domain_models.get(domain, self.domain_models["general"])
    
    def _preprocess_text_for_embedding(self, text: str, domain: str) -> str:
        """Preprocess text for better embedding quality"""
        
        try:
            # Basic preprocessing
            processed_text = text.strip()
            
            # Domain-specific preprocessing
            if domain == "math":
                # Preserve mathematical symbols and structure
                processed_text = processed_text.replace('\n', ' ').replace('  ', ' ')
            elif domain == "english":
                # Clean up for English text
                processed_text = processed_text.replace('\n', ' ').replace('  ', ' ')
                # Remove excessive punctuation
                import re
                processed_text = re.sub(r'[^\w\s\?\.]', '', processed_text)
            
            # Limit text length
            max_length = 8000  # OpenAI limit
            if len(processed_text) > max_length:
                processed_text = processed_text[:max_length]
            
            return processed_text
            
        except Exception as e:
            logger.warning(f"⚠️ Text preprocessing failed: {e}")
            return text[:8000]  # Simple truncation
    
    def _validate_embedding_quality(self, embedding: List[float]) -> bool:
        """Validate embedding quality"""
        
        try:
            if not embedding or len(embedding) == 0:
                return False
            
            # Check for zero vectors
            if all(x == 0.0 for x in embedding):
                return False
            
            # Check for NaN or infinite values
            if any(not (x == x) or not (x != float('inf')) for x in embedding):
                return False
            
            # Check dimension consistency
            expected_dim = 1536  # Default dimension
            if len(embedding) != expected_dim:
                logger.warning(f"⚠️ Unexpected embedding dimension: {len(embedding)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Embedding validation failed: {e}")
            return False
    
    async def _get_local_embedding(self, text: str) -> List[float]:
        """Generate embedding using local model"""
        
        try:
            # Use sentence-transformers for local embedding
            from sentence_transformers import SentenceTransformer
            
            model_name = self.fallback_config["local_model"]
            model = SentenceTransformer(model_name)
            
            # Generate embedding
            embedding = model.encode(text)
            
            # Convert to list of floats
            embedding_list = embedding.tolist()
            
            logger.info(f"✅ Local embedding generated with {model_name}")
            return embedding_list
            
        except Exception as e:
            logger.error(f"❌ Local embedding failed: {e}")
            raise
    
    def _get_hash_based_embedding(self, text: str) -> List[float]:
        """Generate simple hash-based embedding as last resort"""
        
        try:
            import hashlib
            import struct
            
            # Create hash of text
            text_hash = hashlib.sha256(text.encode()).digest()
            
            # Convert hash to list of floats
            embedding = []
            for i in range(0, len(text_hash), 4):
                if i + 4 <= len(text_hash):
                    # Convert 4 bytes to float
                    float_val = struct.unpack('f', text_hash[i:i+4])[0]
                    embedding.append(float_val)
            
            # Pad or truncate to standard dimension
            target_dim = 1536
            while len(embedding) < target_dim:
                embedding.extend(embedding[:min(len(embedding), target_dim - len(embedding))])
            
            embedding = embedding[:target_dim]
            
            logger.warning("⚠️ Generated hash-based embedding")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Hash-based embedding failed: {e}")
            raise
    
    async def embed_texts(self, texts: List[str], domain: str = "general") -> List[List[float]]:
        """Generate embeddings for multiple texts with batching"""
        
        try:
            if not texts:
                return []
            
            # Use batch processing for efficiency
            batch_size = 100  # OpenAI batch limit
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    # Process batch
                    batch_embeddings = await self._embed_batch_openai(batch_texts, domain)
                    embeddings.extend(batch_embeddings)
                    
                except Exception as e:
                    logger.error(f"❌ Batch embedding failed: {e}")
                    # Fallback to individual processing
                    for text in batch_texts:
                        try:
                            embedding = await self.get_embedding(text, domain)
                            embeddings.append(embedding)
                        except Exception as e2:
                            logger.error(f"❌ Individual embedding failed: {e2}")
                            # Add zero vector as fallback
                            model = self.domain_models.get(domain, "text-embedding-3-small")
                            dimension = self.model_dimensions.get(model, 1536)
                            embeddings.append([0.0] * dimension)
            
            logger.info(f"✅ Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Batch embedding failed: {e}")
            raise
    
    async def _embed_batch_openai(self, texts: List[str], domain: str) -> List[List[float]]:
        """Generate embeddings for a batch of texts using OpenAI"""
        
        try:
            model = self._select_model_for_domain(domain)
            
            # Preprocess texts
            processed_texts = [self._preprocess_text_for_embedding(text, domain) for text in texts]
            
            # Generate embeddings
            response = await self.client.embeddings.create(
                model=model,
                input=processed_texts,
                encoding_format="float"
            )
            
            embeddings = [data.embedding for data in response.data]
            
            # Validate embeddings
            valid_embeddings = []
            for i, embedding in enumerate(embeddings):
                if self._validate_embedding_quality(embedding):
                    valid_embeddings.append(embedding)
                else:
                    logger.warning(f"⚠️ Invalid embedding for text {i}, using fallback")
                    # Use local fallback for this text
                    try:
                        fallback_embedding = await self._get_local_embedding(texts[i])
                        valid_embeddings.append(fallback_embedding)
                    except Exception:
                        # Use zero vector as last resort
                        dimension = self.model_dimensions.get(model, 1536)
                        valid_embeddings.append([0.0] * dimension)
            
            return valid_embeddings
            
        except Exception as e:
            logger.error(f"❌ Batch OpenAI embedding failed: {e}")
            raise
    
    def _update_performance_metrics(self, method: str, generation_time: float, success: bool):
        """Update performance tracking metrics"""
        
        try:
            self.performance_metrics["total_embeddings_generated"] += 1
            self.performance_metrics["total_generation_time"] += generation_time
            self.performance_metrics["average_generation_time"] = (
                self.performance_metrics["total_generation_time"] / 
                self.performance_metrics["total_embeddings_generated"]
            )
            
            if success:
                self.performance_metrics["successful_generations"] += 1
            else:
                self.performance_metrics["failed_generations"] += 1
            
            # Track method usage
            self.performance_metrics["model_usage"][method] = (
                self.performance_metrics["model_usage"].get(method, 0) + 1
            )
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get comprehensive embedding service statistics"""
        
        try:
            # Calculate success rate
            total_requests = self.performance_metrics["total_embeddings_generated"]
            success_rate = (
                self.performance_metrics["successful_generations"] / total_requests 
                if total_requests > 0 else 0.0
            )
            
            # Calculate fallback rate
            fallback_rate = (
                self.performance_metrics["fallback_used"] / total_requests 
                if total_requests > 0 else 0.0
            )
            
            return {
                "service": "EmbeddingService",
                "performance": {
                    "total_embeddings_generated": total_requests,
                    "successful_generations": self.performance_metrics["successful_generations"],
                    "failed_generations": self.performance_metrics["failed_generations"],
                    "success_rate": round(success_rate, 4),
                    "fallback_used": self.performance_metrics["fallback_used"],
                    "fallback_rate": round(fallback_rate, 4),
                    "average_generation_time": round(self.performance_metrics["average_generation_time"], 3)
                },
                "model_usage": self.performance_metrics["model_usage"],
                "error_counts": self.performance_metrics["error_counts"],
                "configuration": {
                    "domain_models": self.domain_models,
                    "fallback_config": self.fallback_config,
                    "model_dimensions": self.model_dimensions
                },
                "recommendations": self._generate_recommendations()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting embedding statistics: {e}")
            return {}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on performance metrics"""
        
        recommendations = []
        
        # Success rate recommendations
        success_rate = (
            self.performance_metrics["successful_generations"] / 
            self.performance_metrics["total_embeddings_generated"]
            if self.performance_metrics["total_embeddings_generated"] > 0 else 0.0
        )
        
        if success_rate < 0.8:
            recommendations.append("Low success rate - consider improving error handling and fallback mechanisms")
        
        # Fallback rate recommendations
        fallback_rate = (
            self.performance_metrics["fallback_used"] / 
            self.performance_metrics["total_embeddings_generated"]
            if self.performance_metrics["total_embeddings_generated"] > 0 else 0.0
        )
        
        if fallback_rate > 0.2:
            recommendations.append("High fallback usage - consider optimizing primary embedding method")
        
        # Performance recommendations
        avg_time = self.performance_metrics["average_generation_time"]
        if avg_time > 2.0:
            recommendations.append("Slow embedding generation - consider using faster models or caching")
        
        if not recommendations:
            recommendations.append("Performance is optimal with current configuration")
        
        return recommendations
    
    async def validate_embedding_quality(self, embedding: List[float]) -> Dict[str, Any]:
        """Validate embedding quality and provide detailed analysis"""
        
        try:
            analysis = {
                "is_valid": True,
                "dimension": len(embedding),
                "zero_vectors": 0,
                "nan_values": 0,
                "inf_values": 0,
                "min_value": 0.0,
                "max_value": 0.0,
                "mean_value": 0.0,
                "std_deviation": 0.0,
                "quality_score": 0.0,
                "recommendations": []
            }
            
            if not embedding:
                analysis["is_valid"] = False
                analysis["recommendations"].append("Empty embedding")
                return analysis
            
            # Check for zero vectors
            zero_count = sum(1 for x in embedding if x == 0.0)
            analysis["zero_vectors"] = zero_count
            
            # Check for NaN and infinite values
            nan_count = sum(1 for x in embedding if x != x)
            inf_count = sum(1 for x in embedding if x == float('inf') or x == float('-inf'))
            analysis["nan_values"] = nan_count
            analysis["inf_values"] = inf_count
            
            # Calculate statistics
            valid_values = [x for x in embedding if x == x and x != float('inf') and x != float('-inf')]
            if valid_values:
                analysis["min_value"] = min(valid_values)
                analysis["max_value"] = max(valid_values)
                analysis["mean_value"] = sum(valid_values) / len(valid_values)
                
                # Calculate standard deviation
                variance = sum((x - analysis["mean_value"]) ** 2 for x in valid_values) / len(valid_values)
                analysis["std_deviation"] = variance ** 0.5
            
            # Calculate quality score
            quality_score = 1.0
            if zero_count > len(embedding) * 0.5:
                quality_score -= 0.3
            if nan_count > 0:
                quality_score -= 0.5
            if inf_count > 0:
                quality_score -= 0.5
            if analysis["std_deviation"] < 0.01:
                quality_score -= 0.2
            
            analysis["quality_score"] = max(0.0, quality_score)
            
            # Generate recommendations
            if zero_count > len(embedding) * 0.3:
                analysis["recommendations"].append("High number of zero values - consider regenerating embedding")
            if nan_count > 0:
                analysis["recommendations"].append("NaN values detected - embedding is invalid")
            if inf_count > 0:
                analysis["recommendations"].append("Infinite values detected - embedding is invalid")
            if analysis["std_deviation"] < 0.01:
                analysis["recommendations"].append("Low variance - embedding may not be discriminative")
            
            if not analysis["recommendations"]:
                analysis["recommendations"].append("Embedding quality is good")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error validating embedding quality: {e}")
            return {"is_valid": False, "error": str(e)}


# Global instances
embedding_service = EmbeddingService()
monitoring_embedding_service = MonitoringEmbeddingService(embedding_service)
