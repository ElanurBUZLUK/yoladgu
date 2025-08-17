import json
import time
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from .base import BaseLLMProvider, LLMResponse, LLMConfig


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider"""
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            timeout=self.config.timeout
        )
    
    async def generate_text(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using OpenAI"""
        
        start_time = time.time()
        
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            # Override config with kwargs
            max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
            temperature = kwargs.get("temperature", self.config.temperature)
            
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=self.config.timeout
            )
            
            response_time = time.time() - start_time
            
            # Extract response data
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            cost = self.calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            
            return LLMResponse(
                content=content,
                usage=usage,
                model=self.config.model_name,
                provider=self.config.provider,
                cost=cost,
                response_time=response_time,
                success=True
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            
            return LLMResponse(
                content="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self.config.model_name,
                provider=self.config.provider,
                cost=0.0,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    async def generate_structured_output(
        self, 
        prompt: str, 
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate structured JSON output using OpenAI"""
        
        # Add JSON formatting instruction to prompt
        json_prompt = f"""
{prompt}

Please respond with a valid JSON object that matches this schema:
{json.dumps(schema, indent=2)}

Ensure your response is valid JSON and follows the schema exactly.
"""
        
        # Add JSON instruction to system prompt
        json_system_prompt = system_prompt or ""
        json_system_prompt += "\n\nYou must respond with valid JSON only. Do not include any other text or formatting."
        
        response = await self.generate_text(
            prompt=json_prompt,
            system_prompt=json_system_prompt,
            **kwargs
        )
        
        if response.success:
            try:
                # Validate JSON
                json.loads(response.content)
            except json.JSONDecodeError as e:
                response.success = False
                response.error = f"Invalid JSON response: {str(e)}"
        
        return response


class GPT4Provider(OpenAIProvider):
    """GPT-4 specific provider"""
    
    def __init__(self, api_key: str):
        config = LLMConfig(
            provider="openai_gpt4",
            model_name="gpt-4o",
            api_key=api_key,
            max_tokens=1000,
            temperature=0.7,
            cost_per_1k_tokens=0.03,
            supports_turkish=True,
            use_cases=["question_generation", "complex_evaluation", "content_creation"],
            timeout=30,
            retry_attempts=3
        )
        super().__init__(config)


class GPT35Provider(OpenAIProvider):
    """GPT-3.5 specific provider"""
    
    def __init__(self, api_key: str):
        config = LLMConfig(
            provider="openai_gpt35",
            model_name="gpt-3.5-turbo",
            api_key=api_key,
            max_tokens=800,
            temperature=0.5,
            cost_per_1k_tokens=0.002,
            supports_turkish=True,
            use_cases=["quick_evaluation", "simple_generation", "classification"],
            timeout=20,
            retry_attempts=2
        )
        super().__init__(config)