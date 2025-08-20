import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.llm_context.builders.cloze_question_context_builder import ClozeQuestionContextBuilder
from app.services.llm_context.schemas.cloze_question_context import ClozeQuestionGenerationContext


class TestClozeQuestionContextBuilder:
    """ClozeQuestionContextBuilder testleri"""
    
    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever"""
        retriever = AsyncMock()
        retriever.retrieve_grammar_rules.return_value = ["Rule 1", "Rule 2"]
        retriever.retrieve_vocabulary_context.return_value = "Vocabulary context"
        retriever.retrieve_topic_context.return_value = "Topic context"
        return retriever
    
    @pytest.fixture
    def mock_error_pattern_repo(self):
        """Mock error pattern repository"""
        return AsyncMock()
    
    @pytest.fixture
    def mock_session(self):
        """Mock database session"""
        session = AsyncMock(spec=AsyncSession)
        
        # Mock student attempts
        mock_attempts = [
            MagicMock(
                is_correct=False,
                error_analysis={"patterns": ["past_tense", "articles"]}
            ),
            MagicMock(
                is_correct=True,
                error_analysis=None
            ),
            MagicMock(
                is_correct=False,
                error_analysis={"patterns": ["present_perfect"]}
            )
        ]
        
        session.execute.return_value.scalars.return_value.all.return_value = mock_attempts
        return session
    
    @pytest.fixture
    def builder(self, mock_retriever, mock_error_pattern_repo):
        """Context builder instance"""
        return ClozeQuestionContextBuilder(
            retriever=mock_retriever,
            error_pattern_repo=mock_error_pattern_repo
        )
    
    async def test_build_context_success(self, builder, mock_session):
        """Context oluşturma başarı testi"""
        context = await builder.build(
            session=mock_session,
            user_id="test_user_123",
            num_questions=3,
            last_n_errors=5,
            difficulty_level=4
        )
        
        assert isinstance(context, ClozeQuestionGenerationContext)
        assert context.user_id == "test_user_123"
        assert context.num_questions == 3
        assert context.difficulty_level == 4
        assert "past_tense" in context.user_error_patterns
        assert "articles" in context.user_error_patterns
        assert "present_perfect" in context.user_error_patterns
    
    async def test_get_user_context(self, builder, mock_session):
        """Kullanıcı bağlamı alma testi"""
        user_context = await builder._get_user_context(
            session=mock_session,
            user_id="test_user_123"
        )
        
        assert user_context["user_id"] == "test_user_123"
        assert "user_level" in user_context
        assert "user_preferences" in user_context
        assert user_context["user_preferences"]["recent_attempts_count"] == 3
    
    async def test_get_user_error_patterns(self, builder, mock_session):
        """Kullanıcı hata pattern'ları alma testi"""
        error_patterns = await builder._get_user_error_patterns(
            session=mock_session,
            user_id="test_user_123",
            last_n_errors=5
        )
        
        assert "past_tense" in error_patterns
        assert "articles" in error_patterns
        assert "present_perfect" in error_patterns
        assert len(error_patterns) == 3
    
    async def test_get_knowledge_context(self, builder, mock_session):
        """Bilgi bağlamı alma testi"""
        error_patterns = ["past_tense", "articles"]
        
        knowledge_context = await builder._get_knowledge_context(
            session=mock_session,
            error_patterns=error_patterns
        )
        
        assert "grammar_rules" in knowledge_context
        assert "vocabulary_context" in knowledge_context
        assert "topic_context" in knowledge_context
        assert len(knowledge_context["grammar_rules"]) == 4  # 2 pattern * 2 rules each
    
    async def test_get_output_format_context(self, builder):
        """Çıktı format bağlamı alma testi"""
        output_context = await builder._get_output_format_context(
            num_questions=2
        )
        
        assert "output_schema" in output_context
        assert "format_instructions" in output_context
        assert "2" in output_context["format_instructions"]
    
    def test_create_task_definition(self, builder):
        """Task definition oluşturma testi"""
        task_def = builder._create_task_definition(
            num_questions=3,
            difficulty_level=4
        )
        
        assert "3 adet cloze sorusu" in task_def
        assert "Zorluk seviyesi: 4/5" in task_def
        assert "İngilizce" in task_def
    
    async def test_build_context_with_defaults(self, builder, mock_session):
        """Varsayılan değerlerle context oluşturma testi"""
        context = await builder.build(
            session=mock_session,
            user_id="test_user_123"
        )
        
        assert context.num_questions == 1  # Default
        assert context.difficulty_level == 3  # Default from user level calculation
        assert context.user_id == "test_user_123"
    
    async def test_build_context_error_handling(self, builder, mock_session):
        """Hata durumunda context oluşturma testi"""
        # Session'ı hata verecek şekilde ayarla
        mock_session.execute.side_effect = Exception("Database error")
        
        with pytest.raises(Exception):
            await builder.build(
                session=mock_session,
                user_id="test_user_123"
            )
