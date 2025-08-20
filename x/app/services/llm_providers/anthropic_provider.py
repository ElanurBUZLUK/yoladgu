import json
import time
from typing import Dict, Any, Optional
from anthropic import AsyncAnthropic
from .base import BaseLLMProvider, LLMResponse, LLMConfig


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM Provider"""
    
    def _initialize_client(self):
        """Initialize Anthropic client"""
        if not self.config.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.client = AsyncAnthropic(
            api_key=self.config.api_key,
            timeout=self.config.timeout
        )
    
    async def generate_text(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using Anthropic Claude"""
        
        start_time = time.time()
        
        try:
            # Override config with kwargs
            max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
            temperature = kwargs.get("temperature", self.config.temperature)
            
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Create request parameters
            request_params = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt
            
            response = await self.client.messages.create(**request_params)
            
            response_time = time.time() - start_time
            
            # Extract response data
            content = response.content[0].text
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
            cost = self.calculate_cost(
                response.usage.input_tokens,
                response.usage.output_tokens
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
        """Generate structured JSON output using Anthropic Claude"""
        
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


class ClaudeOpusProvider(AnthropicProvider):
    """Claude Opus specific provider"""
    
    def __init__(self, api_key: str):
        config = LLMConfig(
            provider="anthropic_claude_opus",
            model_name="claude-3-opus-20240229",
            api_key=api_key,
            max_tokens=1000,
            temperature=0.7,
            cost_per_1k_tokens=0.075,  # Higher cost for Opus
            supports_turkish=True,
            use_cases=["complex_reasoning", "detailed_analysis", "creative_writing"],
            timeout=45,
            retry_attempts=3
        )
        super().__init__(config)


class ClaudeSonnetProvider(AnthropicProvider):
    """Claude Sonnet specific provider"""
    
    def __init__(self, api_key: str):
        config = LLMConfig(
            provider="anthropic_claude_sonnet",
            model_name="claude-3-sonnet-20240229",
            api_key=api_key,
            max_tokens=800,
            temperature=0.6,
            cost_per_1k_tokens=0.015,
            supports_turkish=True,
            use_cases=["balanced_tasks", "question_generation", "content_analysis"],
            timeout=30,
            retry_attempts=3
        )
        super().__init__(config)


class ClaudeHaikuProvider(AnthropicProvider):
    """Claude Haiku specific provider"""
    
    def __init__(self, api_key: str):
        config = LLMConfig(
            provider="anthropic_claude_haiku",
            model_name="claude-3-haiku-20240307",
            api_key=api_key,
            max_tokens=500,
            temperature=0.3,
            cost_per_1k_tokens=0.0025,  # Most cost-effective
            supports_turkish=True,
            use_cases=["quick_evaluation", "simple_classification", "fast_responses"],
            timeout=15,
            retry_attempts=2
        )
        super().__init__(config)