"""
RAG API schemas
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class SearchType(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class RAGSearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    subject: str = Field(..., description="Subject (math/english)")
    min_difficulty: Optional[float] = Field(None, description="Minimum difficulty level")
    max_difficulty: Optional[float] = Field(None, description="Maximum difficulty level")
    limit: Optional[int] = Field(10, description="Maximum number of results")


class QuestionResult(BaseModel):
    id: str = Field(..., description="Question ID")
    content: str = Field(..., description="Question content")
    subject: str = Field(..., description="Subject")
    difficulty_level: int = Field(..., description="Difficulty level")
    topic_category: str = Field(..., description="Topic category")
    question_type: str = Field(..., description="Question type")
    similarity_score: float = Field(..., description="Similarity score")
    search_type: str = Field(..., description="Type of search that found this result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RAGSearchResponse(BaseModel):
    query: str = Field(..., description="Original search query")
    subject: str = Field(..., description="Subject searched")
    results: List[QuestionResult] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of results")
    search_type: str = Field(..., description="Type of search performed")


class RAGStatisticsResponse(BaseModel):
    total_questions: int = Field(..., description="Total number of questions")
    questions_with_embeddings: int = Field(..., description="Questions with embeddings")
    embedding_coverage: float = Field(..., description="Percentage of questions with embeddings")
    subject_distribution: Dict[str, int] = Field(..., description="Questions by subject")
    similarity_threshold: float = Field(..., description="Current similarity threshold")
    max_results: int = Field(..., description="Maximum results per search")


class EmbeddingGenerationRequest(BaseModel):
    question_ids: List[str] = Field(..., description="List of question IDs to generate embeddings for")


class EmbeddingGenerationResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    processed: int = Field(..., description="Number of questions processed")
    total: int = Field(..., description="Total number of questions")
    error: Optional[str] = Field(None, description="Error message if failed")
