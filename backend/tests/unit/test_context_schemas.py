import pytest
from pydantic import ValidationError
from app.services.llm_context.schemas.cloze_question_context import ClozeQuestionGenerationContext


class TestClozeQuestionGenerationContext:
    """ClozeQuestionGenerationContext testleri"""
    
    def test_valid_context_creation(self):
        """Geçerli context oluşturma testi"""
        context = ClozeQuestionGenerationContext(
            task_definition="Test task",
            num_questions=3,
            difficulty_level=4,
            user_id="test_user_123",
            user_error_patterns=["past_tense", "articles"],
            user_level=3,
            grammar_rules=["Rule 1", "Rule 2"],
            vocabulary_context="Test vocabulary",
            topic_context="Test topic"
        )
        
        assert context.task_definition == "Test task"
        assert context.num_questions == 3
        assert context.difficulty_level == 4
        assert context.user_id == "test_user_123"
        assert len(context.user_error_patterns) == 2
        assert context.user_level == 3
    
    def test_invalid_difficulty_level(self):
        """Geçersiz zorluk seviyesi testi"""
        with pytest.raises(ValidationError):
            ClozeQuestionGenerationContext(
                task_definition="Test task",
                num_questions=1,
                difficulty_level=6,  # Geçersiz - 1-5 arası olmalı
                user_id="test_user_123"
            )
    
    def test_to_prompt_method(self):
        """to_prompt metodu testi"""
        context = ClozeQuestionGenerationContext(
            task_definition="Create cloze questions",
            num_questions=2,
            difficulty_level=3,
            user_id="test_user_123",
            user_error_patterns=["past_tense"],
            grammar_rules=["Past tense rule"]
        )
        
        prompt = context.to_prompt()
        
        assert "Create cloze questions" in prompt
        assert "test_user_123" in prompt
        assert "past_tense" in prompt
        assert "Past tense rule" in prompt
        assert "2" in prompt
        assert "3" in prompt
    
    def test_to_system_prompt_method(self):
        """to_system_prompt metodu testi"""
        context = ClozeQuestionGenerationContext(
            task_definition="Test task",
            num_questions=1,
            difficulty_level=3,
            user_id="test_user_123"
        )
        
        system_prompt = context.to_system_prompt()
        
        assert "İngilizce öğretmeni" in system_prompt
        assert "cloze soruları" in system_prompt
        assert "hata yaptığı alanlara odaklan" in system_prompt
    
    def test_to_dict_method(self):
        """to_dict metodu testi"""
        context = ClozeQuestionGenerationContext(
            task_definition="Test task",
            num_questions=1,
            difficulty_level=3,
            user_id="test_user_123"
        )
        
        context_dict = context.to_dict()
        
        assert isinstance(context_dict, dict)
        assert context_dict["task_definition"] == "Test task"
        assert context_dict["num_questions"] == 1
        assert context_dict["difficulty_level"] == 3
        assert context_dict["user_id"] == "test_user_123"
    
    def test_get_template_variables(self):
        """get_template_variables metodu testi"""
        context = ClozeQuestionGenerationContext(
            task_definition="Test task",
            num_questions=1,
            difficulty_level=3,
            user_id="test_user_123",
            user_error_patterns=["test_pattern"]
        )
        
        variables = context.get_template_variables()
        
        assert "task_definition" in variables
        assert "user_error_patterns" in variables
        assert variables["user_error_patterns"] == ["test_pattern"]
    
    def test_render_template(self):
        """render_template metodu testi"""
        context = ClozeQuestionGenerationContext(
            task_definition="Create questions",
            num_questions=2,
            difficulty_level=4,
            user_id="test_user_123"
        )
        
        template_content = "Task: {{ task_definition }}, User: {{ user_id }}, Count: {{ num_questions }}"
        rendered = context.render_template(template_content)
        
        assert "Task: Create questions" in rendered
        assert "User: test_user_123" in rendered
        assert "Count: 2" in rendered
