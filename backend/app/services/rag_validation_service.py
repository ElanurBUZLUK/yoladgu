from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime

from app.core.config import settings
from app.services.llm_gateway import llm_gateway
from app.services.vector_index_manager import VectorIndexManager

logger = logging.getLogger(__name__)


class RAGValidationService:
    """RAG validation service with source grounding and answer validation"""
    
    def __init__(self):
        self.vector_manager = VectorIndexManager()
        self.min_similarity_threshold = settings.vector_similarity_threshold
        self.max_citations = 5
        
    async def validate_retrieval_quality(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate the quality of retrieved documents"""
        
        if not retrieved_docs:
            return {
                "answerable": False,
                "reason": "No relevant documents found",
                "coverage_score": 0.0,
                "max_similarity": 0.0,
                "mean_similarity": 0.0
            }
        
        # Calculate similarity statistics
        similarities = [doc.get('similarity', 0.0) for doc in retrieved_docs]
        max_similarity = max(similarities) if similarities else 0.0
        mean_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Determine if query is answerable
        answerable = max_similarity >= self.min_similarity_threshold
        
        # Calculate coverage score (how well the documents cover the query)
        coverage_score = min(1.0, len(retrieved_docs) / 3.0)  # Normalize to 0-1
        
        return {
            "answerable": answerable,
            "reason": "Sufficient relevant documents found" if answerable else "Insufficient similarity",
            "coverage_score": coverage_score,
            "max_similarity": max_similarity,
            "mean_similarity": mean_similarity,
            "doc_count": len(retrieved_docs)
        }
    
    async def generate_grounded_response(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]],
        response_schema: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate a response grounded in retrieved documents"""
        
        # Validate retrieval quality first
        validation = await self.validate_retrieval_quality(query, retrieved_docs)
        
        if not validation["answerable"]:
            return {
                "answer": "I don't have enough relevant information to answer this question accurately.",
                "citations": [],
                "answerable": False,
                "confidence": 0.0,
                "validation": validation
            }
        
        # Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)
        
        # Default response schema if not provided
        if not response_schema:
            response_schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1
                    },
                    "answerable": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["answer", "citations", "answerable", "confidence"]
            }
        
        # Generate grounded response
        try:
            prompt = self._create_grounding_prompt(query, context)
            
            llm_response = await llm_gateway.chat_completion_json(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                json_schema=response_schema
            )
            
            # Add validation info
            llm_response["validation"] = validation
            
            logger.info(f"✅ Generated grounded response for query: {query[:50]}...")
            return llm_response
            
        except Exception as e:
            logger.error(f"❌ Error generating grounded response: {e}")
            return {
                "answer": "I encountered an error while processing your question.",
                "citations": [],
                "answerable": False,
                "confidence": 0.0,
                "validation": validation,
                "error": str(e)
            }
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs[:self.max_citations], 1):
            content = doc.get('content', '')
            similarity = doc.get('similarity', 0.0)
            
            context_parts.append(f"Document {i} (similarity: {similarity:.3f}):\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _create_grounding_prompt(self, query: str, context: str) -> str:
        """Create a prompt that enforces source grounding"""
        
        return f"""You are a helpful assistant that provides accurate, well-sourced answers. 
You must base your response ONLY on the provided context documents. If the context doesn't contain enough information to answer the question accurately, you must say so.

Question: {query}

Context Documents:
{context}

Instructions:
1. Answer the question based ONLY on the provided context documents
2. Include specific citations from the documents in your answer
3. If the context doesn't provide enough information, clearly state this
4. Be concise but thorough
5. Maintain high accuracy and avoid speculation

Please provide your response in the required JSON format with the following fields:
- answer: Your response to the question
- citations: List of specific references from the context documents
- answerable: Whether the question can be answered with the provided context
- confidence: Your confidence level (0.0 to 1.0) in the answer based on the context quality"""
    
    async def validate_answer_quality(
        self, 
        question: str, 
        answer: str, 
        citations: List[str]
    ) -> Dict[str, Any]:
        """Validate the quality of a generated answer"""
        
        validation_schema = {
            "type": "object",
            "properties": {
                "is_grounded": {"type": "boolean"},
                "citation_coverage": {"type": "number", "minimum": 0, "maximum": 1},
                "answer_completeness": {"type": "number", "minimum": 0, "maximum": 1},
                "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
                "issues": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["is_grounded", "citation_coverage", "answer_completeness", "relevance_score", "issues"]
        }
        
        try:
            prompt = f"""Evaluate the quality of this answer:

Question: {question}
Answer: {answer}
Citations: {citations}

Rate the following aspects:
1. is_grounded: Does the answer rely on the provided citations?
2. citation_coverage: How well do the citations support the answer (0-1)?
3. answer_completeness: How complete is the answer (0-1)?
4. relevance_score: How relevant is the answer to the question (0-1)?
5. issues: List any problems with the answer

Provide your evaluation in JSON format."""

            evaluation = await llm_gateway.chat_completion_json(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                json_schema=validation_schema
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"❌ Error validating answer quality: {e}")
            return {
                "is_grounded": False,
                "citation_coverage": 0.0,
                "answer_completeness": 0.0,
                "relevance_score": 0.0,
                "issues": [f"Validation error: {str(e)}"]
            }
    
    async def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get statistics about retrieval performance"""
        
        try:
            # Get index statistics
            index_stats = await self.vector_manager.get_index_statistics()
            
            # Get recent retrieval metrics (you might want to store these in a separate table)
            stats = {
                "index_statistics": index_stats,
                "validation_settings": {
                    "min_similarity_threshold": self.min_similarity_threshold,
                    "max_citations": self.max_citations
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting retrieval statistics: {e}")
            return {"error": str(e)}


# Global instance
rag_validation_service = RAGValidationService()
