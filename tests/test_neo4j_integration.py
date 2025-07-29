import pytest
from unittest.mock import Mock, patch
from app.crud.question import get_similar_questions_from_neo4j, get_question_skill_centrality
from app.crud.student_response import get_student_skill_mastery_from_neo4j, get_student_learning_path_from_neo4j
from app.core.config import settings

@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver"""
    mock_driver = Mock()
    mock_session = Mock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_driver.session.return_value.__exit__ = Mock(return_value=False)
    return mock_driver, mock_session

def test_get_similar_questions_from_neo4j(mock_neo4j_driver):
    """Test getting similar questions from Neo4j"""
    mock_driver, mock_session = mock_neo4j_driver
    
    # Mock Neo4j response
    mock_result = Mock()
    mock_result.__iter__ = lambda self: iter([
        Mock(**{"__getitem__": lambda x, key: {"question_id": 2}.get(key)}),
        Mock(**{"__getitem__": lambda x, key: {"question_id": 3}.get(key)})
    ])
    mock_session.run.return_value = mock_result
    
    with patch('app.crud.question._get_neo4j_driver', return_value=mock_driver), \
         patch.object(settings, 'USE_NEO4J', True):
        
        similar_ids = get_similar_questions_from_neo4j(1, limit=10)
        
        assert similar_ids == [2, 3]
        mock_session.run.assert_called_once()

def test_get_similar_questions_neo4j_disabled():
    """Test that similar questions returns empty list when Neo4j is disabled"""
    with patch.object(settings, 'USE_NEO4J', False):
        similar_ids = get_similar_questions_from_neo4j(1)
        assert similar_ids == []

def test_get_question_skill_centrality(mock_neo4j_driver):
    """Test getting skill centrality for a question"""
    mock_driver, mock_session = mock_neo4j_driver
    
    # Mock Neo4j response
    mock_result = Mock()
    mock_result.__iter__ = lambda self: iter([
        Mock(**{"__getitem__": lambda x, key: {"skill_id": 1, "centrality": 5}.get(key)}),
        Mock(**{"__getitem__": lambda x, key: {"skill_id": 2, "centrality": 3}.get(key)})
    ])
    mock_session.run.return_value = mock_result
    
    with patch('app.crud.question._get_neo4j_driver', return_value=mock_driver), \
         patch.object(settings, 'USE_NEO4J', True):
        
        centrality = get_question_skill_centrality(1)
        
        assert centrality == {1: 5, 2: 3}
        mock_session.run.assert_called_once()

def test_get_student_skill_mastery_from_neo4j(mock_neo4j_driver):
    """Test getting student skill mastery from Neo4j"""
    mock_driver, mock_session = mock_neo4j_driver
    
    # Mock Neo4j response
    mock_result = Mock()
    mock_result.__iter__ = lambda self: iter([
        Mock(**{"__getitem__": lambda x, key: {
            "skill_id": 1, 
            "skill_name": "Algebra", 
            "mastery_score": 0.8, 
            "attempt_count": 10
        }.get(key)})
    ])
    mock_session.run.return_value = mock_result
    
    with patch('app.crud.student_response._get_neo4j_driver', return_value=mock_driver), \
         patch.object(settings, 'USE_NEO4J', True):
        
        mastery = get_student_skill_mastery_from_neo4j(1)
        
        expected = {
            1: {
                "name": "Algebra",
                "mastery": 0.8,
                "attempts": 10
            }
        }
        assert mastery == expected
        mock_session.run.assert_called_once()

def test_get_student_learning_path_from_neo4j(mock_neo4j_driver):
    """Test getting student learning path from Neo4j"""
    mock_driver, mock_session = mock_neo4j_driver
    
    # Mock Neo4j response
    mock_result = Mock()
    mock_result.__iter__ = lambda self: iter([
        Mock(**{"__getitem__": lambda x, key: {
            "question_id": 1,
            "question_text": "Test question",
            "is_correct": True,
            "response_time": 45.2,
            "timestamp": "2023-01-01T10:00:00"
        }.get(key)})
    ])
    mock_session.run.return_value = mock_result
    
    with patch('app.crud.student_response._get_neo4j_driver', return_value=mock_driver), \
         patch.object(settings, 'USE_NEO4J', True):
        
        learning_path = get_student_learning_path_from_neo4j(1, limit=10)
        
        expected = [{
            "question_id": 1,
            "question_text": "Test question",
            "is_correct": True,
            "response_time": 45.2,
            "timestamp": "2023-01-01T10:00:00"
        }]
        assert learning_path == expected
        mock_session.run.assert_called_once()

def test_neo4j_connection_error_handling():
    """Test error handling when Neo4j connection fails"""
    with patch('app.crud.question._get_neo4j_driver', side_effect=Exception("Connection failed")), \
         patch.object(settings, 'USE_NEO4J', True):
        
        # Should return empty list/dict on error
        similar_ids = get_similar_questions_from_neo4j(1)
        assert similar_ids == []
        
        centrality = get_question_skill_centrality(1)
        assert centrality == {}

def test_neo4j_driver_none_handling():
    """Test handling when Neo4j driver returns None"""
    with patch('app.crud.question._get_neo4j_driver', return_value=None), \
         patch.object(settings, 'USE_NEO4J', True):
        
        # Should return empty list/dict when driver is None
        similar_ids = get_similar_questions_from_neo4j(1)
        assert similar_ids == []
        
        centrality = get_question_skill_centrality(1)
        assert centrality == {}

@pytest.mark.asyncio
async def test_question_crud_neo4j_integration():
    """Test Neo4j integration in question CRUD operations"""
    from app.crud.question import create_question
    from app.schemas.question import QuestionCreate
    
    mock_db = Mock()
    mock_question = Mock()
    mock_question.id = 1
    mock_db.add.return_value = None
    mock_db.commit.return_value = None
    mock_db.refresh.return_value = None
    
    question_data = QuestionCreate(
        content="Test question",
        question_type="multiple_choice",
        difficulty_level=2,
        subject_id=1,
        options=["A", "B", "C", "D"],
        correct_answer="A",
        skill_ids={1: 0.8, 2: 0.6}
    )
    
    with patch('app.crud.question._sync_question_to_neo4j') as mock_sync:
        result = create_question(mock_db, question_data, user_id=1)
        
        # Should call Neo4j sync
        mock_sync.assert_called_once_with(1, {1: 0.8, 2: 0.6})

@pytest.mark.asyncio
async def test_student_response_crud_neo4j_integration():
    """Test Neo4j integration in student response CRUD operations"""
    from app.crud.student_response import create_response
    
    mock_db = Mock()
    mock_response = Mock()
    mock_db.add.return_value = None
    mock_db.commit.return_value = None
    mock_db.refresh.return_value = None
    
    with patch('app.crud.student_response._sync_student_response_to_neo4j') as mock_sync:
        result = create_response(
            mock_db,
            student_id=1,
            question_id=1,
            answer="A",
            is_correct=True,
            response_time=45.2
        )
        
        # Should call Neo4j sync
        mock_sync.assert_called_once_with(1, 1, True, 45.2) 