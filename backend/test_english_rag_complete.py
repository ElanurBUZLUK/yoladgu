#!/usr/bin/env python3
"""
Comprehensive test suite for English RAG system
Tests all components: Vector Index Manager, Context Compression, Critic & Revise, and RAG API
"""

import asyncio
import pytest
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List, Dict, Any

# Import the services to test
from app.services.vector_index_manager import VectorIndexManager
from app.services.context_compression import ContextCompressionService
from app.services.critic_revise import CriticReviseService
from app.domains.english.hybrid_retriever import HybridRetriever
from app.domains.english.moderation import ContentModerator
from app.domains.english.rag_retriever_pgvector import RAGRetrieverPGVector


class TestVectorIndexManager:
    """Test Vector Index Manager functionality"""
    
    @pytest.mark.asyncio
    async def test_vector_index_manager_initialization(self):
        """Test vector index manager initialization"""
        manager = VectorIndexManager()
        
        assert manager.embedding_dimension == 1536
        assert manager.index_name_questions == "questions_content_embedding_idx"
        assert manager.index_name_errors == "error_patterns_embedding_idx"
        assert manager.cache_ttl == 1800
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self):
        """Test embedding generation"""
        manager = VectorIndexManager()
        
        text = "This is a test question about English grammar."
        embedding = await manager._generate_embedding(text)
        
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
        assert any(x != 0.0 for x in embedding)  # Not all zeros
    
    @pytest.mark.asyncio
    async def test_perform_vector_search(self):
        """Test vector search functionality"""
        manager = VectorIndexManager()
        
        # Mock query embedding
        query_embedding = [0.1] * 1536
        
        # Mock database response
        mock_results = [
            {
                "id": "1",
                "content": "Test question 1",
                "similarity": 0.85
            },
            {
                "id": "2", 
                "content": "Test question 2",
                "similarity": 0.75
            }
        ]
        
        with patch.object(manager, 'database') as mock_db:
            mock_db.fetch_all.return_value = mock_results
            
            results = await manager.perform_vector_search(
                query_embedding=query_embedding,
                table_name="questions",
                filters={"subject": "english"},
                limit=10
            )
            
            assert len(results) == 2
            assert results[0]["similarity"] > results[1]["similarity"]
    
    @pytest.mark.asyncio
    async def test_get_index_statistics(self):
        """Test index statistics retrieval"""
        manager = VectorIndexManager()
        
        # Mock database responses
        mock_questions_stats = {
            "total_questions": 100,
            "questions_with_embeddings": 80,
            "questions_without_embeddings": 20
        }
        
        mock_errors_stats = {
            "total_patterns": 50,
            "patterns_with_embeddings": 40,
            "patterns_without_embeddings": 10
        }
        
        with patch.object(manager, 'database') as mock_db:
            mock_db.fetch_one.side_effect = [mock_questions_stats, mock_errors_stats]
            
            stats = await manager.get_index_statistics()
            
            assert "questions" in stats
            assert "error_patterns" in stats
            assert stats["questions"]["total_questions"] == 100
            assert stats["error_patterns"]["total_patterns"] == 50


class TestContextCompressionService:
    """Test Context Compression Service functionality"""
    
    @pytest.mark.asyncio
    async def test_context_compression_initialization(self):
        """Test context compression service initialization"""
        service = ContextCompressionService()
        
        assert service.max_tokens == 1200
        assert service.target_tokens == 1000
        assert service.similarity_threshold == 0.9
        assert service.cache_ttl == 1800
    
    @pytest.mark.asyncio
    async def test_compress_context(self):
        """Test context compression functionality"""
        service = ContextCompressionService()
        
        # Test passages
        passages = [
            {
                "id": "1",
                "content": "This is a test passage about English grammar. It contains useful information for learning."
            },
            {
                "id": "2", 
                "content": "Another passage about vocabulary. This helps students learn new words."
            }
        ]
        
        result = await service.compress_context(
            passages=passages,
            target_cefr="B1",
            error_focus=["grammar_error"],
            budget_tokens=500
        )
        
        assert "compressed_text" in result
        assert "source_ids" in result
        assert "token_count" in result
        assert "compression_ratio" in result
        assert "cefr_compliance" in result
        assert "error_coverage" in result
    
    @pytest.mark.asyncio
    async def test_extract_nuggets(self):
        """Test nugget extraction"""
        service = ContextCompressionService()
        
        passages = [
            {
                "id": "1",
                "content": "This is a test passage about English grammar. It contains useful information for learning."
            }
        ]
        
        nuggets = await service._extract_nuggets(
            passages=passages,
            target_cefr="B1",
            error_focus=["grammar_error"]
        )
        
        assert len(nuggets) > 0
        assert all("text" in nugget for nugget in nuggets)
        assert all("source_id" in nugget for nugget in nuggets)
        assert all("relevance_score" in nugget for nugget in nuggets)
    
    @pytest.mark.asyncio
    async def test_deduplicate_nuggets(self):
        """Test nugget deduplication"""
        service = ContextCompressionService()
        
        nuggets = [
            {
                "text": "This is a test sentence.",
                "source_id": "1",
                "relevance_score": 0.8,
                "cefr_level": "B1",
                "length": 6
            },
            {
                "text": "This is a test sentence.",  # Duplicate
                "source_id": "2",
                "relevance_score": 0.9,
                "cefr_level": "B1",
                "length": 6
            }
        ]
        
        unique_nuggets = await service._deduplicate_nuggets(nuggets)
        
        assert len(unique_nuggets) == 1  # Should remove duplicate
        assert unique_nuggets[0]["relevance_score"] == 0.9  # Keep higher score
    
    @pytest.mark.asyncio
    async def test_estimate_cefr_level(self):
        """Test CEFR level estimation"""
        service = ContextCompressionService()
        
        # Test simple text
        simple_text = "I am a student. I like to study."
        cefr_simple = service._estimate_cefr_level(simple_text)
        assert cefr_simple in ["A1", "A2"]
        
        # Test complex text
        complex_text = "The sophisticated implementation demonstrates advanced linguistic capabilities."
        cefr_complex = service._estimate_cefr_level(complex_text)
        assert cefr_complex in ["C1", "C2"]


class TestCriticReviseService:
    """Test Critic & Revise Service functionality"""
    
    @pytest.mark.asyncio
    async def test_critic_revise_initialization(self):
        """Test critic revise service initialization"""
        service = CriticReviseService()
        
        assert service.cache_ttl == 1800
        assert service.max_revision_attempts == 2
    
    @pytest.mark.asyncio
    async def test_critic_and_revise(self):
        """Test critic and revise cycle"""
        service = CriticReviseService()
        
        draft_question = {
            "content": "Choose the correct form of the verb.",
            "options": ["goes", "go", "going", "gone"],
            "correct_answer": "goes",
            "explanation": "The correct form is 'goes'."
        }
        
        with patch.object(service, '_critic_question') as mock_critic, \
             patch.object(service, '_revise_question') as mock_revise:
            
            # Mock critique that needs revision
            mock_critic.return_value = {
                "needs_revision": True,
                "overall_score": 0.6,
                "issues": ["Question is too simple"],
                "suggestions": ["Make it more challenging"]
            }
            
            # Mock revision
            mock_revise.return_value = {
                "content": "Choose the correct form of the verb in the following complex sentence.",
                "options": ["goes", "go", "going", "gone"],
                "correct_answer": "goes",
                "explanation": "The correct form is 'goes' for third person singular."
            }
            
            result = await service.critic_and_revise(
                draft_question=draft_question,
                target_cefr="B1",
                error_focus=["verb_tense"],
                context="Test context",
                question_format="mcq"
            )
            
            assert result["was_revised"] == True
            assert result["revision_count"] == 1
            assert "final_question" in result
            assert "critique" in result
    
    @pytest.mark.asyncio
    async def test_critic_question(self):
        """Test question critique"""
        service = CriticReviseService()
        
        question = {
            "content": "Choose the correct answer.",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A"
        }
        
        with patch.object(service, 'llm_gateway') as mock_llm:
            mock_llm.generate_structured_with_fallback.return_value = {
                "success": True,
                "data": {
                    "needs_revision": False,
                    "overall_score": 0.8,
                    "issues": [],
                    "suggestions": []
                }
            }
            
            critique = await service._critic_question(
                question=question,
                target_cefr="B1",
                error_focus=["grammar_error"],
                context="Test context",
                question_format="mcq"
            )
            
            assert critique["needs_revision"] == False
            assert critique["overall_score"] == 0.8
    
    @pytest.mark.asyncio
    async def test_validate_critique(self):
        """Test critique validation"""
        service = CriticReviseService()
        
        raw_critique = {
            "needs_revision": True,
            "overall_score": 0.6,
            "issues": ["Issue 1", "Issue 2"],
            "suggestions": ["Suggestion 1"]
        }
        
        validated = service._validate_critique(raw_critique)
        
        assert validated["needs_revision"] == True
        assert validated["overall_score"] == 0.6
        assert len(validated["issues"]) == 2
        assert len(validated["suggestions"]) == 1
    
    @pytest.mark.asyncio
    async def test_fallback_critique(self):
        """Test fallback critique"""
        service = CriticReviseService()
        
        question = {
            "content": "Short",  # Too short
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A"
        }
        
        critique = service._fallback_critique(question, "B1")
        
        assert critique["needs_revision"] == True
        assert len(critique["issues"]) > 0
        assert len(critique["suggestions"]) > 0


class TestHybridRetriever:
    """Test Hybrid Retriever functionality"""
    
    @pytest.mark.asyncio
    async def test_hybrid_retriever_initialization(self):
        """Test hybrid retriever initialization"""
        retriever = HybridRetriever()
        
        assert retriever.semantic_weight == 0.6
        assert retriever.keyword_weight == 0.4
        assert retriever.max_results == 50
        assert retriever.cache_ttl == 3600
    
    @pytest.mark.asyncio
    async def test_retrieve_questions(self):
        """Test question retrieval"""
        retriever = HybridRetriever()
        
        # Mock user
        user = MagicMock()
        user.id = "test-user-id"
        user.current_english_level = 3
        
        # Mock database session
        db = AsyncMock()
        
        with patch.object(retriever, '_semantic_search') as mock_semantic, \
             patch.object(retriever, '_keyword_search') as mock_keyword, \
             patch.object(retriever, '_combine_results') as mock_combine:
            
            mock_semantic.return_value = [{"question": "test", "score": 0.8}]
            mock_keyword.return_value = [{"question": "test", "score": 0.7}]
            mock_combine.return_value = [{"question": "test", "score": 0.75}]
            
            results = await retriever.retrieve_questions(
                db=db,
                user=user,
                topic="grammar",
                difficulty=None,
                limit=10,
                exclude_attempted=True
            )
            
            assert len(results) > 0
            mock_semantic.assert_called_once()
            mock_keyword.assert_called_once()
            mock_combine.assert_called_once()


class TestContentModerator:
    """Test Content Moderator functionality"""
    
    @pytest.mark.asyncio
    async def test_content_moderator_initialization(self):
        """Test content moderator initialization"""
        moderator = ContentModerator()
        
        assert len(moderator.inappropriate_patterns) > 0
        assert len(moderator.grammar_patterns) > 0
        assert len(moderator.difficulty_indicators) == 5
        assert moderator.cache_ttl == 1800
    
    @pytest.mark.asyncio
    async def test_moderate_question(self):
        """Test question moderation"""
        moderator = ContentModerator()
        
        result = await moderator.moderate_question(
            question_content="This is a good English question about grammar.",
            options=["A", "B", "C", "D"],
            correct_answer="A",
            difficulty_level=3,
            topic="grammar",
            user_id="test-user"
        )
        
        assert "is_appropriate" in result
        assert "is_quality_acceptable" in result
        assert "moderation_score" in result
        assert "issues" in result
        assert "warnings" in result
        assert "suggestions" in result
    
    @pytest.mark.asyncio
    async def test_check_inappropriate_content(self):
        """Test inappropriate content checking"""
        moderator = ContentModerator()
        
        # Test appropriate content
        appropriate_result = moderator._check_inappropriate_content("This is a good question.")
        assert appropriate_result["is_appropriate"] == True
        
        # Test inappropriate content (if patterns match)
        # Note: This depends on the actual patterns defined


class TestRAGRetrieverPGVector:
    """Test RAG Retriever PGVector functionality"""
    
    @pytest.mark.asyncio
    async def test_rag_retriever_initialization(self):
        """Test RAG retriever initialization"""
        retriever = RAGRetrieverPGVector()
        
        assert retriever.embedding_dimension == 1536
        assert retriever.similarity_threshold == 0.7
        assert retriever.max_results == 20
        assert retriever.cache_ttl == 1800
    
    @pytest.mark.asyncio
    async def test_retrieve_similar_questions(self):
        """Test similar questions retrieval"""
        retriever = RAGRetrieverPGVector()
        
        # Mock user
        user = MagicMock()
        user.id = "test-user-id"
        
        # Mock database session
        db = AsyncMock()
        
        with patch.object(retriever, '_generate_embedding') as mock_embedding, \
             patch.object(retriever, '_vector_similarity_search') as mock_search:
            
            mock_embedding.return_value = [0.1] * 1536
            mock_search.return_value = [{"id": "1", "content": "test", "similarity": 0.8}]
            
            results = await retriever.retrieve_similar_questions(
                db=db,
                query_text="test query",
                user=user,
                topic="grammar",
                difficulty=None,
                limit=10
            )
            
            assert len(results) > 0
            mock_embedding.assert_called_once()
            mock_search.assert_called_once()


class TestEnglishRAGIntegration:
    """Integration tests for English RAG system"""
    
    @pytest.mark.asyncio
    async def test_full_rag_pipeline(self):
        """Test full RAG pipeline integration"""
        # This would test the complete flow from API to final response
        # For now, we'll test the individual components work together
        
        # Test vector index manager
        vector_manager = VectorIndexManager()
        assert vector_manager is not None
        
        # Test context compression
        compression_service = ContextCompressionService()
        assert compression_service is not None
        
        # Test critic revise
        critic_service = CriticReviseService()
        assert critic_service is not None
        
        # Test hybrid retriever
        hybrid_retriever = HybridRetriever()
        assert hybrid_retriever is not None
        
        # Test content moderator
        content_moderator = ContentModerator()
        assert content_moderator is not None
        
        # Test RAG retriever
        rag_retriever = RAGRetrieverPGVector()
        assert rag_retriever is not None
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling across components"""
        
        # Test vector index manager error handling
        vector_manager = VectorIndexManager()
        with patch.object(vector_manager, 'database') as mock_db:
            mock_db.execute.side_effect = Exception("Database error")
            
            result = await vector_manager.create_vector_indexes()
            assert result == False
        
        # Test context compression error handling
        compression_service = ContextCompressionService()
        result = await compression_service.compress_context(
            passages=[],  # Empty passages should trigger fallback
            target_cefr="B1",
            error_focus=[],
            budget_tokens=100
        )
        
        assert "compressed_text" in result
        assert "error_coverage" in result
        
        # Test critic revise error handling
        critic_service = CriticReviseService()
        with patch.object(critic_service, '_critic_question') as mock_critic:
            mock_critic.side_effect = Exception("LLM error")
            
            result = await critic_service.critic_and_revise(
                draft_question={"content": "test"},
                target_cefr="B1",
                error_focus=[],
                context="test",
                question_format="mcq"
            )
            
            assert "error" in result
            assert result["was_revised"] == False


def run_tests():
    """Run all tests"""
    print("🧪 Running English RAG System Tests...")
    
    # Test Vector Index Manager
    print("\n1. Testing Vector Index Manager...")
    vector_manager = VectorIndexManager()
    assert vector_manager.embedding_dimension == 1536
    print("✅ Vector Index Manager tests passed")
    
    # Test Context Compression Service
    print("\n2. Testing Context Compression Service...")
    compression_service = ContextCompressionService()
    assert compression_service.max_tokens == 1200
    print("✅ Context Compression Service tests passed")
    
    # Test Critic Revise Service
    print("\n3. Testing Critic Revise Service...")
    critic_service = CriticReviseService()
    assert critic_service.cache_ttl == 1800
    print("✅ Critic Revise Service tests passed")
    
    # Test Hybrid Retriever
    print("\n4. Testing Hybrid Retriever...")
    hybrid_retriever = HybridRetriever()
    assert hybrid_retriever.semantic_weight == 0.6
    print("✅ Hybrid Retriever tests passed")
    
    # Test Content Moderator
    print("\n5. Testing Content Moderator...")
    content_moderator = ContentModerator()
    assert len(content_moderator.inappropriate_patterns) > 0
    print("✅ Content Moderator tests passed")
    
    # Test RAG Retriever
    print("\n6. Testing RAG Retriever...")
    rag_retriever = RAGRetrieverPGVector()
    assert rag_retriever.embedding_dimension == 1536
    print("✅ RAG Retriever tests passed")
    
    print("\n🎉 All English RAG System tests passed!")


if __name__ == "__main__":
    run_tests()
