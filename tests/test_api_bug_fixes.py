"""
Test API endpoint bug fixes
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from app.main import app
from app.db.models import User, Question


client = TestClient(app)


class TestSimilarQuestionsEndpointFix:
    """Test the fixed similar questions endpoint"""
    
    @pytest.fixture
    def mock_user(self):
        user = Mock(spec=User)
        user.id = 1
        user.username = "test_user"
        return user
    
    @pytest.fixture  
    def mock_question(self):
        question = Mock(spec=Question)
        question.id = 123
        question.content = "What is 2+2?"
        return question
    
    def test_similar_questions_endpoint_fixed_parameters(self, mock_user, mock_question):
        """Test that similar questions endpoint uses correct parameters"""
        
        # Mock database and authentication
        with patch('app.api.v1.endpoints.recommendations.get_db') as mock_get_db, \
             patch('app.api.v1.endpoints.recommendations.get_current_user') as mock_get_user, \
             patch('app.services.enhanced_embedding_service.enhanced_embedding_service') as mock_embedding:
            
            # Setup mocks
            mock_db = Mock(spec=Session)
            mock_get_db.return_value = mock_db
            mock_get_user.return_value = mock_user
            
            # Mock the fixed method call
            mock_embedding.find_similar_questions_by_id.return_value = [
                {"question_id": 124, "similarity_score": 0.9, "source": "embedding"},
                {"question_id": 125, "similarity_score": 0.8, "source": "embedding"}
            ]
            
            # Make request
            response = client.get(
                "/api/v1/similar/123?count=5&similarity_threshold=0.7",
                headers={"Authorization": "Bearer fake-token"}
            )
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["source_question_id"] == 123
            assert len(data["similar_questions"]) == 2
            assert data["similarity_threshold"] == 0.7
            
            # Verify correct method was called with correct parameters
            mock_embedding.find_similar_questions_by_id.assert_called_once_with(
                123, mock_db, threshold=0.7, limit=5
            )
    
    def test_similar_questions_endpoint_fallback_to_neo4j(self, mock_user):
        """Test fallback to Neo4j when embedding fails"""
        
        with patch('app.api.v1.endpoints.recommendations.get_db') as mock_get_db, \
             patch('app.api.v1.endpoints.recommendations.get_current_user') as mock_get_user, \
             patch('app.services.enhanced_embedding_service.enhanced_embedding_service') as mock_embedding, \
             patch('app.api.v1.endpoints.recommendations.get_similar_questions_from_neo4j') as mock_neo4j:
            
            # Setup mocks
            mock_db = Mock(spec=Session)
            mock_get_db.return_value = mock_db
            mock_get_user.return_value = mock_user
            
            # Mock embedding service failure (returns empty list)
            mock_embedding.find_similar_questions_by_id.return_value = []
            
            # Mock Neo4j fallback
            mock_neo4j.return_value = [
                {"question_id": 124, "shared_skills": 3},
                {"question_id": 125, "shared_skills": 2}
            ]
            
            # Make request
            response = client.get(
                "/api/v1/similar/123?count=5",
                headers={"Authorization": "Bearer fake-token"}
            )
            
            # Verify response uses Neo4j fallback
            assert response.status_code == 200
            data = response.json()
            
            # Check that Neo4j data was properly transformed
            similar_questions = data["similar_questions"]
            assert len(similar_questions) == 2
            assert similar_questions[0]["source"] == "neo4j"
            assert similar_questions[0]["question_id"] == 124
            assert similar_questions[0]["similarity_score"] == 0.3  # 3/10 normalized
            
            # Verify Neo4j was called as fallback
            mock_neo4j.assert_called_once_with(123, limit=5)
    
    def test_similar_questions_endpoint_parameter_validation(self):
        """Test endpoint parameter validation"""
        
        with patch('app.api.v1.endpoints.recommendations.get_current_user') as mock_get_user:
            mock_user = Mock(spec=User)
            mock_get_user.return_value = mock_user
            
            # Test invalid similarity threshold (too low)
            response = client.get(
                "/api/v1/similar/123?similarity_threshold=0.3",
                headers={"Authorization": "Bearer fake-token"}
            )
            assert response.status_code == 422  # Validation error
            
            # Test invalid similarity threshold (too high)
            response = client.get(
                "/api/v1/similar/123?similarity_threshold=1.5",
                headers={"Authorization": "Bearer fake-token"}
            )
            assert response.status_code == 422  # Validation error
            
            # Test invalid count (too high)
            response = client.get(
                "/api/v1/similar/123?count=25",
                headers={"Authorization": "Bearer fake-token"}
            )
            assert response.status_code == 422  # Validation error


class TestEmbeddingServiceIntegration:
    """Integration tests for embedding service fixes"""
    
    def test_original_find_similar_questions_still_works(self):
        """Test that original find_similar_questions method still works with correct parameters"""
        from app.services.enhanced_embedding_service import EnhancedEmbeddingService
        
        service = EnhancedEmbeddingService()
        
        # Test with proper embedding parameter
        query_embedding = [0.1, 0.2, 0.3]
        
        with patch.object(service, 'get_postgres_connection') as mock_conn:
            # Mock database response
            mock_cursor = Mock()
            mock_cursor.fetchall.return_value = []
            mock_conn.return_value.cursor.return_value = mock_cursor
            
            # This should work without errors
            result = service.find_similar_questions(
                query_embedding=query_embedding,
                threshold=0.8,
                limit=10
            )
            
            assert isinstance(result, list)
            # Verify cursor was called (method executed)
            mock_cursor.execute.assert_called_once()
    
    def test_helper_method_integration(self):
        """Test the new helper method integration"""
        from app.services.enhanced_embedding_service import EnhancedEmbeddingService
        from app.db.models import Question
        
        service = EnhancedEmbeddingService()
        
        # Mock database session
        mock_db = Mock(spec=Session)
        mock_question = Mock(spec=Question)
        mock_question.content = "Test question content"
        mock_db.query.return_value.filter.return_value.first.return_value = mock_question
        
        with patch.object(service, 'compute_embedding') as mock_compute, \
             patch.object(service, 'find_similar_questions') as mock_find:
            
            mock_compute.return_value = [0.1, 0.2, 0.3]
            mock_find.return_value = [{"question_id": 1, "similarity": 0.9}]
            
            # Test the helper method
            result = service.find_similar_questions_by_id(
                question_id=123,
                db_session=mock_db,
                threshold=0.8,
                limit=5
            )
            
            # Verify the flow
            assert len(result) == 1
            assert result[0]["question_id"] == 1
            mock_compute.assert_called_once_with("Test question content")
            mock_find.assert_called_once_with([0.1, 0.2, 0.3], threshold=0.8, limit=5)