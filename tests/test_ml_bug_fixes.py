"""
Test fixes for ML service bugs
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.services.enhanced_embedding_service import EnhancedEmbeddingService
from app.services.ensemble_service import EnhancedEnsembleScoringService


class TestEmbeddingServiceBugFixes:
    """Test embedding service parameter mismatch fixes"""
    
    @pytest.fixture
    def embedding_service(self):
        return EnhancedEmbeddingService()
    
    @pytest.fixture
    def mock_db(self):
        db = Mock(spec=Session)
        return db
    
    def test_find_similar_questions_by_id_helper(self, embedding_service, mock_db):
        """Test the new helper method works correctly"""
        # Mock question
        mock_question = Mock()
        mock_question.content = "What is 2+2?"
        mock_db.query.return_value.filter.return_value.first.return_value = mock_question
        
        # Mock embedding computation
        with patch.object(embedding_service, 'compute_embedding') as mock_compute:
            mock_compute.return_value = [0.1, 0.2, 0.3]
            
            with patch.object(embedding_service, 'find_similar_questions') as mock_find:
                mock_find.return_value = [{"question_id": 1, "similarity": 0.9}]
                
                # Test the helper method
                result = embedding_service.find_similar_questions_by_id(
                    question_id=123, 
                    db_session=mock_db,
                    threshold=0.8,
                    limit=5
                )
                
                # Verify correct calls
                mock_compute.assert_called_once_with("What is 2+2?")
                mock_find.assert_called_once_with([0.1, 0.2, 0.3], threshold=0.8, limit=5)
                assert result == [{"question_id": 1, "similarity": 0.9}]
    
    def test_find_similar_questions_by_id_question_not_found(self, embedding_service, mock_db):
        """Test helper method handles missing question"""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = embedding_service.find_similar_questions_by_id(
            question_id=999, 
            db_session=mock_db
        )
        
        assert result == []
    
    def test_find_similar_questions_by_id_embedding_fails(self, embedding_service, mock_db):
        """Test helper method handles embedding computation failure"""
        # Mock question
        mock_question = Mock()
        mock_question.content = "Test question"
        mock_db.query.return_value.filter.return_value.first.return_value = mock_question
        
        # Mock embedding failure
        with patch.object(embedding_service, 'compute_embedding') as mock_compute:
            mock_compute.return_value = None  # Embedding failed
            
            result = embedding_service.find_similar_questions_by_id(
                question_id=123,
                db_session=mock_db
            )
            
            assert result == []


class TestNeo4jSimilarityBugFixes:
    """Test Neo4j similarity score calculation fixes"""
    
    @pytest.fixture
    def ensemble_service(self):
        return EnhancedEnsembleScoringService()
    
    def test_calculate_neo4j_similarity_score_with_dict_return(self, ensemble_service):
        """Test similarity score calculation with Dict return from Neo4j"""
        # Mock get_similar_questions_from_neo4j to return Dict format
        mock_neo4j_data = [
            {"question_id": 101, "shared_skills": 3},
            {"question_id": 102, "shared_skills": 2},
            {"question_id": 101, "shared_skills": 1},  # Duplicate for frequency test
        ]
        
        with patch('app.services.ensemble_service.get_similar_questions_from_neo4j') as mock_neo4j:
            mock_neo4j.return_value = mock_neo4j_data
            
            # Test when question is found (should get frequency-based score)
            score = ensemble_service.calculate_neo4j_similarity_score(
                question_id=101,
                student_recent_question_ids=[1, 2, 3]
            )
            
            # Question 101 appears twice, so frequency = 2, score = min(1.0, 2 * 0.3) = 0.6
            assert score == 0.6
    
    def test_calculate_neo4j_similarity_score_question_not_found(self, ensemble_service):
        """Test similarity score when question not in similar results"""
        mock_neo4j_data = [
            {"question_id": 201, "shared_skills": 3},
            {"question_id": 202, "shared_skills": 2},
        ]
        
        with patch('app.services.ensemble_service.get_similar_questions_from_neo4j') as mock_neo4j:
            mock_neo4j.return_value = mock_neo4j_data
            
            # Test when question is NOT found (should get default score)
            score = ensemble_service.calculate_neo4j_similarity_score(
                question_id=999,  # Not in mock data
                student_recent_question_ids=[1, 2, 3]
            )
            
            assert score == 0.3  # Default score
    
    def test_calculate_neo4j_similarity_score_mixed_formats(self, ensemble_service):
        """Test similarity score with mixed Dict/int return formats (backward compatibility)"""
        mock_neo4j_data = [
            {"question_id": 301, "shared_skills": 3},  # Dict format
            302,  # Int format (backward compatibility)
            {"question_id": 301, "shared_skills": 1},  # Dict format duplicate
        ]
        
        with patch('app.services.ensemble_service.get_similar_questions_from_neo4j') as mock_neo4j:
            mock_neo4j.return_value = mock_neo4j_data
            
            score = ensemble_service.calculate_neo4j_similarity_score(
                question_id=301,
                student_recent_question_ids=[1, 2, 3]
            )
            
            # Question 301 appears twice, score = min(1.0, 2 * 0.3) = 0.6
            assert score == 0.6
    
    def test_calculate_neo4j_similarity_score_no_recent_questions(self, ensemble_service):
        """Test similarity score with no recent questions"""
        score = ensemble_service.calculate_neo4j_similarity_score(
            question_id=123,
            student_recent_question_ids=[]
        )
        
        assert score == 0.5  # Default when no recent questions
    
    def test_calculate_neo4j_similarity_score_exception_handling(self, ensemble_service):
        """Test similarity score handles exceptions gracefully"""
        with patch('app.services.ensemble_service.get_similar_questions_from_neo4j') as mock_neo4j:
            mock_neo4j.side_effect = Exception("Neo4j connection failed")
            
            score = ensemble_service.calculate_neo4j_similarity_score(
                question_id=123,
                student_recent_question_ids=[1, 2, 3]
            )
            
            assert score == 0.5  # Fallback score on exception