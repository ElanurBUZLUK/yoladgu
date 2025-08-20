# Context Schemas Module
# LLM'e gönderilecek bağlamın farklı parçalarını temsil eden Pydantic modelleri

from .base_context import BaseContext
from .cloze_question_context import ClozeQuestionGenerationContext

__all__ = [
    "BaseContext",
    "ClozeQuestionGenerationContext"
]
