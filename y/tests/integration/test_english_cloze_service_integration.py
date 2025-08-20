import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.english_cloze_service import EnglishClozeService
from app.services.llm_context.builders.cloze_question_context_builder import ClozeQuestionContextBuilder
from app.services.llm_context.schemas.cloze_question_context import ClozeQuestionGenerationContext


class TestEnglishClozeServiceIntegration:
    """EnglishClozeService integration testleri"""
    
    @pytest.fixture
    def mock_llm_gateway(self):
        """Mock LLM gateway"""
        gateway = AsyncMock()
        gateway.generate_json.return_value = {
            "success": True,
            "parsed_json": [
                {
                    "original_sentence": "I go to school every day.",
                    "cloze_sentence": "I ___ to school every day.",
                    "correct_answer": "go",
                    "distractors": ["goes", "going", "went"],
                    "explanation": "Present tense with 'I'",
                    "difficulty_level": 3,
                    "error_type_addressed": "present_tense"
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }
        return gateway
    
    @pytest.fixture
    def mock_context_builder(self):
        """Mock context builder"""
        builder = AsyncMock(spec=ClozeQuestionContextBuilder)
        
        # Mock context
        mock_context = MagicMock(spec=ClozeQuestionGenerationContext)
        mock_context.to_prompt.return_value = "Test prompt"
        mock_context.to_system_prompt.return_value = "Test system prompt"
        
        builder.build.return_value = mock_context
        return builder
    
    @pytest.fixture
    def mock_question_repo(self):
        """Mock question repository"""
        repo = AsyncMock()
        
        # Mock question
        mock_question = MagicMock()
        mock_question.id = "test_question_id"
        mock_question.content = "I ___ to school every day."
        mock_question.correct_answer = "go"
        
        repo.create.return_value = mock_question
        return repo
    
    @pytest.fixture
    def mock_session(self):
        """Mock database session"""
        return AsyncMock(spec=AsyncSession)
    
    @pytest.fixture
    def service(self, mock_llm_gateway, mock_context_builder, mock_question_repo):
        """Service instance"""
        return EnglishClozeService(
            llm_gateway_service=mock_llm_gateway,
            context_builder=mock_context_builder,
            question_repo=mock_question_repo
        )
    
    async def test_generate_cloze_questions_success(
        self, 
        service, 
        mock_session, 
        mock_context_builder,
        mock_llm_gateway,
        mock_question_repo
    ):
        """Başarılı cloze soru üretimi testi"""
        
        # Test parametreleri
        user_id = "test_user_123"
        num_questions = 2
        last_n_errors = 5
        
        # Service'i çağır
        result = await service.generate_cloze_questions(
            session=mock_session,
            user_id=user_id,
            num_questions=num_questions,
            last_n_errors=last_n_errors
        )
        
        # Context builder'ın çağrıldığını kontrol et
        mock_context_builder.build.assert_called_once_with(
            session=mock_session,
            user_id=user_id,
            num_questions=num_questions,
            last_n_errors=last_n_errors
        )
        
        # LLM gateway'in çağrıldığını kontrol et
        mock_llm_gateway.generate_json.assert_called_once()
        call_args = mock_llm_gateway.generate_json.call_args
        
        assert call_args[1]["prompt"] == "Test prompt"
        assert call_args[1]["system_prompt"] == "Test system prompt"
        assert call_args[1]["schema"] is not None
        
        # Sonuçları kontrol et
        assert len(result) == 1
        assert result[0].content == "I ___ to school every day."
        assert result[0].correct_answer == "go"
    
    async def test_generate_cloze_questions_llm_failure(
        self, 
        service, 
        mock_session, 
        mock_llm_gateway
    ):
        """LLM hatası durumunda test"""
        
        # LLM gateway'i hata verecek şekilde ayarla
        mock_llm_gateway.generate_json.return_value = {
            "success": False,
            "error": "LLM service unavailable"
        }
        
        # Service'i çağır
        result = await service.generate_cloze_questions(
            session=mock_session,
            user_id="test_user_123"
        )
        
        # Boş liste döndüğünü kontrol et
        assert result == []
    
    async def test_generate_cloze_questions_validation_error(
        self, 
        service, 
        mock_session, 
        mock_llm_gateway
    ):
        """Validation hatası durumunda test"""
        
        # Geçersiz JSON döndür
        mock_llm_gateway.generate_json.return_value = {
            "success": True,
            "parsed_json": [
                {
                    "original_sentence": "Test sentence",
                    # Eksik required field'lar
                    "cloze_sentence": "Test ___ sentence"
                }
            ]
        }
        
        # Service'i çağır
        result = await service.generate_cloze_questions(
            session=mock_session,
            user_id="test_user_123"
        )
        
        # Boş liste döndüğünü kontrol et (validation hatası nedeniyle)
        assert result == []
    
    async def test_generate_cloze_questions_correct_answer_in_distractors(
        self, 
        service, 
        mock_session, 
        mock_llm_gateway
    ):
        """Doğru cevabın şıklarda olması durumunda test"""
        
        # Doğru cevabı şıklarda olan JSON döndür
        mock_llm_gateway.generate_json.return_value = {
            "success": True,
            "parsed_json": [
                {
                    "original_sentence": "I go to school every day.",
                    "cloze_sentence": "I ___ to school every day.",
                    "correct_answer": "go",
                    "distractors": ["go", "going", "went"],  # Doğru cevap şıklarda
                    "explanation": "Test explanation",
                    "difficulty_level": 3,
                    "error_type_addressed": "present_tense"
                }
            ]
        }
        
        # Service'i çağır
        result = await service.generate_cloze_questions(
            session=mock_session,
            user_id="test_user_123"
        )
        
        # Boş liste döndüğünü kontrol et (doğru cevap şıklarda olduğu için)
        assert result == []
    
    async def test_generate_cloze_questions_exception_handling(
        self, 
        service, 
        mock_session
    ):
        """Genel exception durumunda test"""
        
        # Context builder'ı hata verecek şekilde ayarla
        service.context_builder.build.side_effect = Exception("Database error")
        
        # Service'i çağır
        result = await service.generate_cloze_questions(
            session=mock_session,
            user_id="test_user_123"
        )
        
        # Boş liste döndüğünü kontrol et
        assert result == []
