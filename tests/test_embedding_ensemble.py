import pytest
from unittest.mock import Mock, patch
import numpy as np
from app.services.embedding_service import EmbeddingService, compute_embedding
from app.services.ensemble_service import EnsembleScoringService, calculate_ensemble_score
from app.core.config import settings

@pytest.fixture
def embedding_service():
    return EmbeddingService()

@pytest.fixture
def ensemble_service():
    return EnsembleScoringService()

def test_embedding_service_initialization(embedding_service):
    """Embedding service başlatma testi"""
    assert embedding_service.model_name == "paraphrase-MiniLM-L6-v2"
    assert embedding_service.embedding_dim == 384
    assert embedding_service.model is None  # Lazy loading

def test_compute_embedding_basic(embedding_service):
    """Temel embedding hesaplama testi"""
    with patch('app.services.embedding_service.SentenceTransformer') as mock_transformer:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3] * 128)  # 384 boyutlu
        mock_transformer.return_value = mock_model
        
        result = embedding_service.compute_embedding("Test question")
        
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)
        mock_model.encode.assert_called_once()

def test_compute_embedding_empty_text(embedding_service):
    """Boş metin için embedding testi"""
    result = embedding_service.compute_embedding("")
    assert len(result) == 384
    assert all(x == 0.0 for x in result)

def test_compute_embeddings_batch(embedding_service):
    """Toplu embedding hesaplama testi"""
    texts = ["Question 1", "Question 2", ""]
    
    with patch('app.services.embedding_service.SentenceTransformer') as mock_transformer:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3] * 128, [0.4, 0.5, 0.6] * 128])
        mock_transformer.return_value = mock_model
        
        results = embedding_service.compute_embeddings_batch(texts)
        
        assert len(results) == 3
        assert len(results[0]) == 384
        assert len(results[1]) == 384
        assert all(x == 0.0 for x in results[2])  # Boş metin için

def test_embedding_similarity_calculation(ensemble_service):
    """Embedding benzerlik hesaplama testi"""
    recent_questions = ["What is 2+2?", "Solve for x: 3x = 6"]
    
    with patch('app.services.ensemble_service.compute_embedding') as mock_compute:
        mock_compute.side_effect = [
            [0.1, 0.2, 0.3] * 128,  # İlk soru embedding'i
            [0.4, 0.5, 0.6] * 128,  # İkinci soru embedding'i
            [0.2, 0.3, 0.4] * 128   # Mevcut soru embedding'i
        ]
        
        score = ensemble_service.calculate_embedding_similarity_score(
            "What is 3+3?", recent_questions
        )
        
        assert 0 <= score <= 1
        assert isinstance(score, float)

def test_skill_mastery_score_calculation(ensemble_service):
    """Skill mastery skor hesaplama testi"""
    with patch('app.services.ensemble_service.get_question_skill_centrality') as mock_centrality, \
         patch('app.services.ensemble_service.get_student_skill_mastery_from_neo4j') as mock_mastery:
        
        mock_centrality.return_value = {1: 5, 2: 3}
        mock_mastery.return_value = {
            1: {"name": "Algebra", "mastery": 0.3, "attempts": 10},
            2: {"name": "Geometry", "mastery": 0.7, "attempts": 8}
        }
        
        score = ensemble_service.calculate_skill_mastery_score(1, 1)
        
        assert 0 <= score <= 1
        assert isinstance(score, float)

def test_difficulty_match_score_calculation(ensemble_service):
    """Zorluk uyum skoru hesaplama testi"""
    # Yüksek performans - zorluk artmalı
    score_high = ensemble_service.calculate_difficulty_match_score(
        question_difficulty=4,
        student_level=3.0,
        student_recent_performance=0.9
    )
    
    # Düşük performans - zorluk azalmalı
    score_low = ensemble_service.calculate_difficulty_match_score(
        question_difficulty=2,
        student_level=3.0,
        student_recent_performance=0.2
    )
    
    assert 0 <= score_high <= 1
    assert 0 <= score_low <= 1
    assert isinstance(score_high, float)
    assert isinstance(score_low, float)

def test_neo4j_similarity_score_calculation(ensemble_service):
    """Neo4j benzerlik skoru hesaplama testi"""
    with patch('app.services.ensemble_service.get_similar_questions_from_neo4j') as mock_similar:
        mock_similar.return_value = [2, 3, 4]  # Benzer soru ID'leri
        
        score = ensemble_service.calculate_neo4j_similarity_score(
            question_id=1,
            student_recent_question_ids=[2, 3]
        )
        
        assert 0 <= score <= 1
        assert isinstance(score, float)

def test_ensemble_score_calculation(ensemble_service):
    """Ensemble skor hesaplama testi"""
    with patch.object(ensemble_service, 'calculate_embedding_similarity_score', return_value=0.8), \
         patch.object(ensemble_service, 'calculate_skill_mastery_score', return_value=0.6), \
         patch.object(ensemble_service, 'calculate_difficulty_match_score', return_value=0.9), \
         patch.object(ensemble_service, 'calculate_neo4j_similarity_score', return_value=0.4):
        
        scores = ensemble_service.calculate_ensemble_score(
            river_score=0.7,
            question_content="Test question",
            question_id=1,
            question_difficulty=3,
            student_id=1,
            student_level=2.5,
            student_recent_performance=0.6,
            student_recent_questions=["Previous question"],
            student_recent_question_ids=[2, 3]
        )
        
        assert 'ensemble_score' in scores
        assert 'river_score' in scores
        assert 'embedding_similarity' in scores
        assert 'skill_mastery' in scores
        assert 'difficulty_match' in scores
        assert 'neo4j_similarity' in scores
        
        assert 0 <= scores['ensemble_score'] <= 1
        assert scores['river_score'] == 0.7

def test_ensemble_score_fallback(ensemble_service):
    """Ensemble skor fallback testi"""
    with patch.object(ensemble_service, 'calculate_embedding_similarity_score', side_effect=Exception("Error")):
        scores = ensemble_service.calculate_ensemble_score(
            river_score=0.8,
            question_content="Test question",
            question_id=1,
            question_difficulty=3,
            student_id=1,
            student_level=2.5,
            student_recent_performance=0.6,
            student_recent_questions=[],
            student_recent_question_ids=[]
        )
        
        # Hata durumunda fallback değerler kullanılmalı
        assert scores['ensemble_score'] == 0.8  # River score'a fallback
        assert scores['embedding_similarity'] == 0.5  # Varsayılan değer

def test_filter_questions_by_thresholds(ensemble_service):
    """Threshold filtreleme testi"""
    questions = [
        {'difficulty_level': 3, 'embedding_similarity': 0.7, 'skill_mastery': 0.4},
        {'difficulty_level': 5, 'embedding_similarity': 0.8, 'skill_mastery': 0.6},  # Zorluk çok yüksek
        {'difficulty_level': 2, 'embedding_similarity': 0.5, 'skill_mastery': 0.3},  # Benzerlik düşük
        {'difficulty_level': 3, 'embedding_similarity': 0.8, 'skill_mastery': 0.2}   # Mastery düşük
    ]
    
    filtered = ensemble_service.filter_questions_by_thresholds(
        questions, student_level=3.0, student_recent_performance=0.6
    )
    
    # Sadece ilk soru geçmeli (diğerleri threshold'ları geçemez)
    assert len(filtered) == 1
    assert filtered[0]['difficulty_level'] == 3

def test_dynamic_weight_adjustment(ensemble_service):
    """Dinamik ağırlık ayarlama testi"""
    # Düşük performans - embedding ağırlığı artmalı
    ensemble_service.adjust_weights_dynamically(student_performance=0.3, question_count=5)
    assert ensemble_service.weights['embedding_similarity'] == 0.35
    assert ensemble_service.weights['river_score'] == 0.25
    
    # Yüksek performans - skill mastery ağırlığı artmalı
    ensemble_service.adjust_weights_dynamically(student_performance=0.9, question_count=20)
    assert ensemble_service.weights['skill_mastery'] == 0.30
    assert ensemble_service.weights['river_score'] == 0.30

@pytest.mark.asyncio
async def test_embedding_service_integration():
    """Embedding service entegrasyon testi"""
    with patch('app.services.embedding_service.psycopg2.connect') as mock_connect:
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock embedding hesaplama
        with patch('app.services.embedding_service.SentenceTransformer') as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3] * 128)
            mock_transformer.return_value = mock_model
            
            service = EmbeddingService()
            success = await service.update_question_embedding(1, "Test question")
            
            assert success is True
            mock_cursor.execute.assert_called()
            mock_conn.commit.assert_called()

def test_embedding_stats(embedding_service):
    """Embedding istatistikleri testi"""
    with patch('app.services.embedding_service.psycopg2.connect') as mock_connect:
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock istatistik verileri
        mock_cursor.fetchone.side_effect = [10, 7, 3]  # total, with_embedding, without_embedding
        
        stats = embedding_service.get_embedding_stats()
        
        assert 'total_questions' in stats
        assert 'questions_with_embedding' in stats
        assert 'questions_without_embedding' in stats
        assert 'embedding_coverage' in stats
        assert stats['embedding_coverage'] == 70.0  # 7/10 * 100 