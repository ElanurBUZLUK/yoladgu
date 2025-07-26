import pytest
from unittest.mock import Mock, patch
from app.services.recommendation_service import RecommendationService
from app.services.level_service import update_student_level

@pytest.fixture
def mock_db():
    return Mock()

@pytest.fixture
def mock_redis():
    return Mock()

@pytest.fixture
def recommendation_service(mock_redis):
    with patch('app.services.recommendation_service.redis.Redis.from_url', return_value=mock_redis):
        service = RecommendationService()
        return service

def test_recommendation_service_initialization(recommendation_service):
    assert recommendation_service.model_type == 'river'
    assert recommendation_service.river_model is not None
    assert recommendation_service.linucb_bandit is not None

def test_get_student_features(recommendation_service, mock_db):
    # Mock student responses
    mock_response = Mock()
    mock_response.is_correct = True
    mock_response.response_time = 5000
    mock_response.confidence_level = 0.8
    mock_response.created_at.hour = 14
    
    mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [mock_response]
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    features = recommendation_service.get_student_features(mock_db, student_id=1)
    
    assert 'total_questions' in features
    assert 'correct_answers' in features
    assert 'avg_response_time' in features
    assert features['total_questions'] == 1
    assert features['correct_answers'] == 1

def test_get_question_features(recommendation_service, mock_db):
    # Mock question
    mock_question = Mock()
    mock_question.id = 1
    mock_question.difficulty_level = 2
    mock_question.question_type = "multiple_choice"
    mock_question.subject_id = 1
    mock_question.content = "Test question"
    mock_question.options = ["A", "B", "C", "D"]
    mock_question.correct_answer = "A"
    mock_question.explanation = "Test explanation"
    mock_question.tags = ["math", "algebra"]
    mock_question.created_by = 1
    mock_question.is_active = True
    
    # Mock question skills
    mock_question_skill = Mock()
    mock_question_skill.skill_id = 1
    mock_question_skill.weight = 0.5
    mock_question_skill.skill.difficulty_level = 2
    
    mock_db.query.return_value.filter.return_value.first.return_value = mock_question
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_question_skill]
    
    features = recommendation_service.get_question_features(mock_db, question_id=1)
    
    assert 'difficulty_level' in features
    assert 'question_type' in features
    assert 'subject_id' in features
    assert features['difficulty_level'] == 2
    assert features['question_type'] == "multiple_choice"

def test_update_student_level():
    # Test level update logic
    current_level = 1
    is_correct = True
    consecutive_correct = 3
    
    new_level = update_student_level(current_level, is_correct, consecutive_correct)
    assert new_level >= current_level

def test_recommendation_service_with_linucb(recommendation_service, mock_db):
    # Test LinUCB model
    recommendation_service.model_type = 'linucb'
    
    # Mock questions
    mock_question = Mock()
    mock_question.id = 1
    mock_question.content = "Test question"
    mock_question.question_type = "multiple_choice"
    mock_question.difficulty_level = 2
    mock_question.subject_id = 1
    mock_question.options = ["A", "B", "C", "D"]
    mock_question.correct_answer = "A"
    mock_question.explanation = "Test explanation"
    mock_question.tags = ["math"]
    mock_question.created_by = 1
    mock_question.is_active = True
    
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_question]
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    # Mock student features
    with patch.object(recommendation_service, 'get_student_features', return_value={'accuracy_rate_overall': 0.8}):
        with patch.object(recommendation_service, 'get_question_features', return_value={'difficulty_level': 2}):
            recommendations = recommendation_service.get_recommendations(mock_db, student_id=1, n_recommendations=1)
            
            assert len(recommendations) > 0
            assert 'question_id' in recommendations[0]
            assert 'score' in recommendations[0] 