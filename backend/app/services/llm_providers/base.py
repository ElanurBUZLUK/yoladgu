from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from enum import Enum


class LLMProvider(str, Enum):
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE_OPUS = "anthropic_claude_opus"
    ANTHROPIC_CLAUDE_SONNET = "anthropic_claude_sonnet"
    ANTHROPIC_CLAUDE_HAIKU = "anthropic_claude_haiku"
    LOCAL_MODEL = "local_model"


class LLMConfig(BaseModel):
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    cost_per_1k_tokens: float = 0.0
    supports_turkish: bool = True
    use_cases: List[str] = []
    timeout: int = 30
    retry_attempts: int = 3


class LLMResponse(BaseModel):
    content: str
    usage: Dict[str, int]
    model: str
    provider: str
    cost: float = 0.0
    response_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the LLM client"""
        pass
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using the LLM"""
        pass
    
    @abstractmethod
    async def generate_structured_output(
        self, 
        prompt: str, 
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate structured output (JSON) using the LLM"""
        pass
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost based on token usage"""
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * self.config.cost_per_1k_tokens
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "provider": self.config.provider,
            "model": self.config.model_name,
            "supports_turkish": self.config.supports_turkish,
            "use_cases": self.config.use_cases,
            "cost_per_1k_tokens": self.config.cost_per_1k_tokens
        }