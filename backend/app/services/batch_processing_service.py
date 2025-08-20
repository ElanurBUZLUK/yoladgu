import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import time

from app.services.embedding_service import embedding_service
from app.services.vector_index_manager import vector_index_manager
from app.services.metadata_schema_service import metadata_schema_service, ContentType, Domain
from app.core.config import settings

logger = logging.getLogger(__name__)


class BatchProcessingService:
    """High-performance batch processing service for embeddings and vector operations"""
    
    def __init__(self):
        self.batch_size = settings.vector_batch_size or 100
        self.max_concurrent_batches = 5
        self.retry_attempts = 3
        self.retry_delay = 1.0
        
        # Performance tracking
        self.performance_metrics = {
            "total_batches_processed": 0,
            "total_items_processed": 0,
            "average_batch_time": 0.0,
            "total_processing_time": 0.0,
            "success_rate": 1.0,
            "last_processed": None
        }
    
    async def batch_process_questions(
        self,
        session: AsyncSession,
        questions: List[Dict[str, Any]],
        domain: str,
        content_type: str = "question"
    ) -> Dict[str, Any]:
        """Batch process questions with embeddings and vector storage"""
        
        try:
            start_time = time.time()
            logger.info(f"🚀 Starting batch processing for {len(questions)} questions in domain: {domain}")
            
            # Split questions into batches
            batches = self._split_into_batches(questions, self.batch_size)
            logger.info(f"📦 Split into {len(batches)} batches of size {self.batch_size}")
            
            # Process batches concurrently
            results = await self._process_batches_concurrently(
                batches, domain, content_type
            )
            
            # Aggregate results
            total_processed = sum(result.get("items_processed", 0) for result in results)
            total_embeddings = sum(result.get("embeddings_generated", 0) for result in results)
            total_vector_stored = sum(result.get("vector_stored", 0) for result in results)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(len(batches), total_processed, processing_time)
            
            logger.info(f"✅ Batch processing completed: {total_processed}/{len(questions)} items processed")
            
            return {
                "success": True,
                "total_questions": len(questions),
                "total_processed": total_processed,
                "embeddings_generated": total_embeddings,
                "vector_stored": total_vector_stored,
                "batches_processed": len(batches),
                "processing_time": processing_time,
                "performance_metrics": self.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "total_questions": len(questions),
                "total_processed": 0
            }
    
    async def _process_batches_concurrently(
        self,
        batches: List[List[Dict[str, Any]]],
        domain: str,
        content_type: str
    ) -> List[Dict[str, Any]]:
        """Process multiple batches concurrently for better performance"""
        
        try:
            # Limit concurrent batches to avoid overwhelming the system
            semaphore = asyncio.Semaphore(self.max_concurrent_batches)
            
            async def process_batch_with_semaphore(batch):
                async with semaphore:
                    return await self._process_single_batch(batch, domain, content_type)
            
            # Process batches concurrently
            tasks = [process_batch_with_semaphore(batch) for batch in batches]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Batch {i} failed: {result}")
                    processed_results.append({
                        "success": False,
                        "error": str(result),
                        "items_processed": 0,
                        "embeddings_generated": 0,
                        "vector_stored": 0
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Concurrent batch processing failed: {e}")
            return []
    
    async def _process_single_batch(
        self,
        batch: List[Dict[str, Any]],
        domain: str,
        content_type: str
    ) -> Dict[str, Any]:
        """Process a single batch of questions"""
        
        try:
            batch_start_time = time.time()
            logger.debug(f"🔄 Processing batch of {len(batch)} items")
            
            # 1. Generate embeddings in batch
            embedding_results = await self._batch_generate_embeddings(batch, domain)
            
            # 2. Prepare vector DB data
            vector_data = self._prepare_vector_data(batch, embedding_results, domain, content_type)
            
            # 3. Store in vector DB
            vector_stored = await self._batch_store_vectors(vector_data, domain, content_type)
            
            batch_time = time.time() - batch_start_time
            logger.debug(f"✅ Batch processed in {batch_time:.2f}s")
            
            return {
                "success": True,
                "items_processed": len(batch),
                "embeddings_generated": len(embedding_results),
                "vector_stored": vector_stored,
                "batch_time": batch_time
            }
            
        except Exception as e:
            logger.error(f"❌ Single batch processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "items_processed": 0,
                "embeddings_generated": 0,
                "vector_stored": 0
            }
    
    async def _batch_generate_embeddings(
        self,
        items: List[Dict[str, Any]],
        domain: str
    ) -> List[Tuple[Dict[str, Any], List[float]]]:
        """Generate embeddings for multiple items in a single API call"""
        
        try:
            # Extract text content for embedding
            texts = []
            for item in items:
                text = self._extract_text_for_embedding(item)
                if text:
                    texts.append(text)
            
            if not texts:
                logger.warning("⚠️ No valid texts found for embedding generation")
                return []
            
            # Generate embeddings in batch
            embeddings = await embedding_service.embed_texts(
                texts=texts,
                domain=domain
            )
            
            # Match embeddings with items
            results = []
            for i, (item, embedding) in enumerate(zip(items, embeddings)):
                if embedding:
                    results.append((item, embedding))
                else:
                    logger.warning(f"⚠️ Failed to generate embedding for item {i}")
            
            logger.info(f"✅ Generated {len(results)} embeddings from {len(texts)} texts")
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch embedding generation failed: {e}")
            return []
    
    def _extract_text_for_embedding(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract text content from item for embedding generation"""
        
        try:
            # Try different possible text fields
            text_fields = ["content", "text", "question", "error_text", "description"]
            
            for field in text_fields:
                if field in item and item[field]:
                    text = str(item[field]).strip()
                    if len(text) > 10:  # Minimum text length
                        return text
            
            # If no text field found, try to construct from available data
            if "error_type" in item and "topic_category" in item:
                return f"{item['error_type']}: {item['topic_category']}"
            
            logger.warning(f"⚠️ No suitable text found for embedding in item: {item.get('id', 'unknown')}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting text for embedding: {e}")
            return None
    
    def _prepare_vector_data(
        self,
        items: List[Dict[str, Any]],
        embedding_results: List[Tuple[Dict[str, Any], List[float]]],
        domain: str,
        content_type: str
    ) -> List[Dict[str, Any]]:
        """Prepare data for vector database storage"""
        
        try:
            vector_data = []
            
            for item, embedding in embedding_results:
                try:
                    # Build standardized metadata
                    metadata = self._build_standardized_metadata(
                        item, domain, content_type
                    )
                    
                    # Prepare vector item
                    vector_item = {
                        "obj_ref": str(item.get("id", f"generated_{time.time()}")),
                        "content": self._extract_text_for_embedding(item),
                        "embedding": embedding,
                        "metadata": metadata
                    }
                    
                    vector_data.append(vector_item)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to prepare vector data for item: {e}")
                    continue
            
            logger.info(f"✅ Prepared {len(vector_data)} items for vector storage")
            return vector_data
            
        except Exception as e:
            logger.error(f"❌ Error preparing vector data: {e}")
            return []
    
    def _build_standardized_metadata(
        self,
        item: Dict[str, Any],
        domain: str,
        content_type: str
    ) -> Dict[str, Any]:
        """Build standardized metadata for vector storage"""
        
        try:
            # Use metadata schema service for standardization
            if content_type == "error_pattern":
                metadata = metadata_schema_service.build_error_pattern_metadata(
                    domain=domain,
                    error_type=item.get("error_type", "unknown"),
                    obj_ref=str(item.get("id", "unknown")),
                    user_id=item.get("user_id"),
                    topic_category=item.get("topic_category"),
                    skill_tag=item.get("skill_tag"),
                    error_count=item.get("error_count", 1)
                )
            elif content_type == "question":
                metadata = metadata_schema_service.build_question_metadata(
                    domain=domain,
                    question_type=item.get("question_type", "multiple_choice"),
                    obj_ref=str(item.get("id", "unknown")),
                    difficulty_level=item.get("difficulty_level", 2.5),
                    topic_category=item.get("topic_category"),
                    source_type=item.get("source_type", "generated")
                )
            else:
                metadata = metadata_schema_service.build_standard_metadata(
                    domain=domain,
                    content_type=content_type,
                    obj_ref=str(item.get("id", "unknown")),
                    difficulty_level=item.get("difficulty_level", 2.5),
                    topic_category=item.get("topic_category", "general")
                )
            
            # Add any additional item-specific metadata
            if "created_at" in item:
                metadata["created_at"] = item["created_at"]
            if "user_id" in item:
                metadata["user_id"] = item["user_id"]
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Error building standardized metadata: {e}")
            # Return basic metadata as fallback
            return {
                "domain": domain,
                "content_type": content_type,
                "obj_ref": str(item.get("id", "unknown")),
                "created_at": datetime.utcnow().isoformat(),
                "metadata_version": "1.0.0"
            }
    
    async def _batch_store_vectors(
        self,
        vector_data: List[Dict[str, Any]],
        domain: str,
        content_type: str
    ) -> int:
        """Store multiple vectors in vector database using batch operations"""
        
        try:
            if not vector_data:
                logger.warning("⚠️ No vector data to store")
                return 0
            
            # Use enhanced batch upsert for better performance
            result = await vector_index_manager.batch_upsert_domain_embeddings_enhanced(
                domain=domain,
                content_type=content_type,
                items=vector_data,
                batch_size=self.batch_size
            )
            
            stored_count = len(vector_data)
            logger.info(f"✅ Stored {stored_count} vectors in vector database")
            
            return stored_count
            
        except Exception as e:
            logger.error(f"❌ Batch vector storage failed: {e}")
            return 0
    
    def _split_into_batches(
        self,
        items: List[Dict[str, Any]],
        batch_size: int
    ) -> List[List[Dict[str, Any]]]:
        """Split items into batches of specified size"""
        
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _update_performance_metrics(
        self,
        batches_processed: int,
        items_processed: int,
        processing_time: float
    ):
        """Update performance tracking metrics"""
        
        try:
            self.performance_metrics["total_batches_processed"] += batches_processed
            self.performance_metrics["total_items_processed"] += items_processed
            self.performance_metrics["total_processing_time"] += processing_time
            
            # Calculate average batch time
            if self.performance_metrics["total_batches_processed"] > 0:
                self.performance_metrics["average_batch_time"] = (
                    self.performance_metrics["total_processing_time"] / 
                    self.performance_metrics["total_batches_processed"]
                )
            
            # Update success rate (simplified calculation)
            if self.performance_metrics["total_items_processed"] > 0:
                self.performance_metrics["success_rate"] = 0.95  # Placeholder
            
            self.performance_metrics["last_processed"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        return {
            "service": "BatchProcessingService",
            "metrics": self.performance_metrics,
            "configuration": {
                "batch_size": self.batch_size,
                "max_concurrent_batches": self.max_concurrent_batches,
                "retry_attempts": self.retry_attempts
            },
            "recommendations": self._generate_performance_recommendations()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        
        recommendations = []
        
        # Batch size optimization
        if self.performance_metrics["average_batch_time"] > 5.0:
            recommendations.append("Consider reducing batch size for faster processing")
        elif self.performance_metrics["average_batch_time"] < 1.0:
            recommendations.append("Consider increasing batch size for better throughput")
        
        # Concurrency optimization
        if self.performance_metrics["success_rate"] < 0.9:
            recommendations.append("Reduce concurrent batches to improve stability")
        elif self.performance_metrics["success_rate"] > 0.98:
            recommendations.append("Consider increasing concurrent batches for better performance")
        
        # General recommendations
        if self.performance_metrics["total_items_processed"] > 1000:
            recommendations.append("High volume detected - consider implementing streaming processing")
        
        if not recommendations:
            recommendations.append("Performance is optimal with current configuration")
        
        return recommendations
    
    async def clear_performance_metrics(self):
        """Clear performance metrics (useful for testing)"""
        
        self.performance_metrics = {
            "total_batches_processed": 0,
            "total_items_processed": 0,
            "average_batch_time": 0.0,
            "total_processing_time": 0.0,
            "success_rate": 1.0,
            "last_processed": None
        }
        
        logger.info("🧹 Performance metrics cleared")


# Global instance
batch_processing_service = BatchProcessingService()
