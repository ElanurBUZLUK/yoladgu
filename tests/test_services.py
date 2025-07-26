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

@pytest.mark.asyncio
async def test_recommendation_service_with_linucb(recommendation_service, mock_db):
    # Test LinUCB model with hybrid approach
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
    with patch.object(recommendation_service, 'get_student_features', return_value={'accuracy_rate_overall': 0.8, 'level': 2}):
        with patch.object(recommendation_service, 'get_question_features', return_value={'difficulty_level': 2}):
            with patch.object(recommendation_service, '_adjust_question_difficulty_runtime', return_value=2):
                recommendations = await recommendation_service.get_recommendations(mock_db, student_id=1, n_recommendations=1)
                
                assert len(recommendations) > 0
                assert 'question_id' in recommendations[0]
                assert 'score' in recommendations[0]
                assert 'adjusted_difficulty' in recommendations[0]
                assert 'original_difficulty' in recommendations[0]

@pytest.mark.asyncio
async def test_adaptive_hint_generation(recommendation_service, mock_db):
    """Test adaptive hint generation"""
    # Mock question
    mock_question = Mock()
    mock_question.id = 1
    mock_question.content = "Test question"
    mock_question.hint = "Basic hint"
    
    mock_db.query.return_value.filter.return_value.first.return_value = mock_question
    
    # Mock student features
    with patch.object(recommendation_service, 'get_student_features', return_value={'level': 2, 'last20_accuracy': 0.3, 'avg_response_time': 70000}):
        with patch('app.services.recommendation_service.llm_service') as mock_llm:
            mock_llm.generate_adaptive_hint.return_value = "Adaptive hint for struggling student"
            
            hint = await recommendation_service.get_adaptive_hint(1, 1, mock_db)
            
            assert hint == "Adaptive hint for struggling student"
            mock_llm.generate_adaptive_hint.assert_called_once()

@pytest.mark.asyncio
async def test_contextual_explanation_generation(recommendation_service, mock_db):
    """Test contextual explanation generation"""
    # Mock question
    mock_question = Mock()
    mock_question.id = 1
    mock_question.content = "Test question"
    mock_question.correct_answer = "Correct answer"
    mock_question.explanation = "Basic explanation"
    
    mock_db.query.return_value.filter.return_value.first.return_value = mock_question
    
    # Mock student features
    with patch.object(recommendation_service, 'get_student_features', return_value={'level': 2}):
        with patch('app.services.recommendation_service.llm_service') as mock_llm:
            mock_llm.generate_contextual_explanation.return_value = "Contextual explanation"
            
            explanation = await recommendation_service.get_contextual_explanation(1, 1, "Wrong answer", mock_db)
            
            assert explanation == "Contextual explanation"
            mock_llm.generate_contextual_explanation.assert_called_once()

@pytest.mark.asyncio
async def test_difficulty_adjustment_runtime(recommendation_service, mock_db):
    """Test runtime difficulty adjustment"""
    # Mock question
    mock_question = Mock()
    mock_question.content = "Test question"
    mock_question.difficulty_level = 3
    
    # Mock student features
    student_features = {'level': 2, 'last20_accuracy': 0.9, 'avg_response_time': 30000}
    
    with patch('app.services.recommendation_service.llm_service') as mock_llm:
        mock_llm.adjust_difficulty_runtime.return_value = {'adjusted_difficulty': 4, 'reason': 'High performance'}
        
        adjusted = await recommendation_service._adjust_question_difficulty_runtime(
            mock_question, 2, student_features
        )
        
        assert adjusted == 4
        mock_llm.adjust_difficulty_runtime.assert_called_once()

def test_question_ingestion_fetch_from_api(mock_db):
    from app.services.question_ingestion_service import QuestionIngestionService
    service = QuestionIngestionService()
    sample_data = [
        {
            "content": "Sample question",
            "question_type": "multiple_choice",
            "difficulty_level": 1,
            "subject_id": 1,
            "options": ["a", "b"],
            "correct_answer": "a",
        }
    ]
    with patch('app.services.question_ingestion_service.requests.get') as mock_get, \
         patch('app.services.question_ingestion_service.create_question') as mock_create:
        mock_get.return_value.json.return_value = sample_data
        mock_get.return_value.raise_for_status.return_value = None
        mock_create.return_value = object()
        imported = service.fetch_from_api(mock_db, 'http://test', created_by=1)
        assert len(imported) == 1
        mock_create.assert_called()

def test_question_ingestion_scrape_from_url(mock_db):
    from app.services.question_ingestion_service import QuestionIngestionService
    service = QuestionIngestionService()
    html = "<div class='question'>What is 2+2?</div>"
    with patch('app.services.question_ingestion_service.requests.get') as mock_get, \
         patch('app.services.question_ingestion_service.create_question') as mock_create:
        mock_get.return_value.text = html
        mock_get.return_value.raise_for_status.return_value = None
        mock_create.return_value = object()
        imported = service.scrape_from_url(mock_db, 'http://test', subject_id=1, created_by=1)
        assert len(imported) == 1
        mock_create.assert_called() 