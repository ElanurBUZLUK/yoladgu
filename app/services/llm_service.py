"""
LLM Service
OpenAI, Anthropic ve local model entegrasyonları
"""

import asyncio
import json
import structlog
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import aiohttp
from pydantic import BaseModel
from app.core.config import settings

logger = structlog.get_logger()


class ModelProvider(Enum):
    """Model sağlayıcıları"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"


class ModelType(Enum):
    """Model türleri"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"


class LLMRequest(BaseModel):
    """LLM isteği"""
    prompt: str
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None


class LLMResponse(BaseModel):
    """LLM yanıtı"""
    text: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    created_at: datetime


class LLMService:
    """LLM servisi"""

    def __init__(self):
        self.current_model = "gpt-3.5-turbo"
        self.current_provider = ModelProvider.OPENAI
        self.api_keys = {}
        self.session = None
        self.initialized = False
        
        # Model configurations
        self.models = {
            "openai": {
                "gpt-3.5-turbo": {"max_tokens": 4096, "cost_per_1k": 0.002},
                "gpt-4": {"max_tokens": 8192, "cost_per_1k": 0.03},
                "gpt-4-turbo": {"max_tokens": 128000, "cost_per_1k": 0.01},
            },
            "anthropic": {
                "claude-3-sonnet": {"max_tokens": 200000, "cost_per_1k": 0.015},
                "claude-3-haiku": {"max_tokens": 200000, "cost_per_1k": 0.00025},
                "claude-3-opus": {"max_tokens": 200000, "cost_per_1k": 0.15},
            },
            "local": {
                "llama-2-7b": {"max_tokens": 4096, "cost_per_1k": 0.0},
                "mistral-7b": {"max_tokens": 8192, "cost_per_1k": 0.0},
            }
        }
    
    async def initialize(self):
        """LLM servisini başlat"""
        try:
            # API keys'i yükle
            self.api_keys = {
                "openai": getattr(settings, 'OPENAI_API_KEY', None),
                "anthropic": getattr(settings, 'ANTHROPIC_API_KEY', None),
            }
            
            # HTTP session oluştur
            self.session = aiohttp.ClientSession()
            
            self.initialized = True
            logger.info("llm_service_initialized")
            
        except Exception as e:
            logger.error("llm_service_initialization_error", error=str(e))
            raise
    
    async def generate_text(self, request: LLMRequest) -> str:
        """generate_text(prompt, max_tokens, temp)"""
        try:
            # Rate limiting check
            await self._check_rate_limit()
            
            # Execute with retry
            result = await self._execute_with_retry(
                lambda: self._generate_text_internal(request)
            )
            
            return str(result) if result else ""
            
        except Exception as e:
            logger.error("generate_text_error", error=str(e))
            raise

    async def _generate_text_internal(self, request: LLMRequest) -> str:
        """Internal text generation with provider selection"""
        if self.current_provider == ModelProvider.OPENAI:
            return await self._generate_openai(request)
        elif self.current_provider == ModelProvider.ANTHROPIC:
            return await self._generate_anthropic(request)
        elif self.current_provider == ModelProvider.LOCAL:
            return await self._generate_local(request)
        else:
            raise ValueError(f"Unsupported provider: {self.current_provider}")
    
    async def _generate_openai(self, request: LLMRequest) -> str:
        """OpenAI ile metin üret"""
        try:
            if not self.api_keys.get("openai"):
                raise ValueError("OpenAI API key not configured")
            
            headers = {
                "Authorization": f"Bearer {self.api_keys['openai']}",
                "Content-Type": "application/json"
            }
            
            # Messages formatını hazırla
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            
            if request.messages:
                messages.extend(request.messages)
            else:
                messages.append({"role": "user", "content": request.prompt})
            
            data = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
            }
            
            if request.stop:
                data["stop"] = request.stop
            
            if not self.session:
                raise ValueError("HTTP session not initialized")
                
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {error_text}")

        except Exception as e:
            logger.error("openai_generation_error", error=str(e))
            raise
    
    async def _generate_anthropic(self, request: LLMRequest) -> str:
        """Anthropic ile metin üret"""
        try:
            if not self.api_keys.get("anthropic"):
                raise ValueError("Anthropic API key not configured")
            
            headers = {
                "x-api-key": self.api_keys["anthropic"],
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            # System message'ı messages'a ekle
            messages = []
            if request.system_prompt:
                messages.append({"role": "user", "content": f"System: {request.system_prompt}\n\nUser: {request.prompt}"})
            else:
                messages.append({"role": "user", "content": request.prompt})
            
            data = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            }
            
            if not self.session:
                raise ValueError("HTTP session not initialized")
                
            async with self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["content"][0]["text"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error: {error_text}")
                    
        except Exception as e:
            logger.error("anthropic_generation_error", error=str(e))
            raise
    
    async def _generate_local(self, request: LLMRequest) -> str:
        """Local model ile metin üret"""
        try:
            # Local model için basit implementasyon
            # Gerçek uygulamada Ollama, vLLM gibi local model servisleri kullanılır
            
            logger.warning("local_model_not_implemented", model=request.model)
            return f"[Local model {request.model} response placeholder]"
            
        except Exception as e:
            logger.error("local_generation_error", error=str(e))
            raise
    
    async def analyze_text(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """analyze_text(text, type) + retry / rate-limit"""
        try:
            # Rate limiting check
            await self._check_rate_limit()
            
            # Prepare analysis prompt based on type
            if analysis_type == "difficulty":
                prompt = f"Analyze the difficulty level of this question (1-5 scale): {text}"
                system_prompt = "You are an educational content analyzer. Rate difficulty from 1 (easiest) to 5 (hardest)."
            elif analysis_type == "topic":
                prompt = f"Identify the main topic and subtopics for this question: {text}"
                system_prompt = "You are a subject matter expert. Identify the main topic and related subtopics."
            elif analysis_type == "quality":
                prompt = f"Evaluate the quality and clarity of this question: {text}"
                system_prompt = "You are an educational content evaluator. Assess question quality and clarity."
            else:
                prompt = f"Analyze this text: {text}"
                system_prompt = "You are a general text analyzer."
            
            # Create request with retry mechanism
            request = LLMRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            # Execute with retry
            result = await self._execute_with_retry(
                lambda: self.generate_text(request)
            )
            
            return {
                "analysis_type": analysis_type,
                "text": text,
                "result": result,
                "model": self.current_model,
                "provider": self.current_provider.value
            }
            
        except Exception as e:
            logger.error("analyze_text_error", analysis_type=analysis_type, error=str(e))
            raise
    
    async def generate_study_plan(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """AI ile çalışma planı oluştur"""
        try:
            system_prompt = """You are an expert educational planner. Create a personalized study plan based on the user's profile. 
            Consider their current level, target subjects, available time, and learning preferences."""
            
            user_prompt = f"""
            Create a study plan for a student with the following profile:
            - Current level: {user_profile.get('level', 'beginner')}
            - Target subjects: {user_profile.get('target_subjects', [])}
            - Available time per day: {user_profile.get('study_hours_per_day', 2)} hours
            - Learning preferences: {user_profile.get('learning_preferences', 'visual')}
            - Weak areas: {user_profile.get('weak_areas', [])}
            
            Provide a structured study plan with daily activities, recommended resources, and progress milestones.
            """
            
            request = LLMRequest(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            response = await self.generate_text(request)

            return {
                "study_plan": response,
                "user_profile": user_profile,
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("generate_study_plan_error", error=str(e))
            raise
    
    async def generate_explanation(self, question_text: str, user_answer: str, correct_answer: str) -> str:
        """Soru açıklaması oluştur"""
        try:
            system_prompt = """You are an expert teacher. Provide a clear, educational explanation for why the student's answer is correct or incorrect. 
            Focus on the learning concept and help the student understand the reasoning."""
            
            user_prompt = f"""
            Question: {question_text}
            Student's answer: {user_answer}
            Correct answer: {correct_answer}
            
            Provide an educational explanation that helps the student understand the concept.
            """
            
            request = LLMRequest(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=300,
                temperature=0.5
            )
            
            return await self.generate_text(request)

        except Exception as e:
            logger.error("generate_explanation_error", error=str(e))
            raise
    
    async def switch_model(self, model_key: str) -> bool:
        """Model switch mekanizması (switch_model(key))"""
        try:
            # Parse model key to get provider and model name
            if "/" in model_key:
                provider_name, model_name = model_key.split("/", 1)
            else:
                # Default to OpenAI if no provider specified
                provider_name = "openai"
                model_name = model_key
            
            # Validate provider
            try:
                provider = ModelProvider(provider_name.lower())
            except ValueError:
                logger.error("invalid_provider", provider=provider_name)
                return False
            
            # Validate model exists for provider
            if provider.value not in self.models or model_name not in self.models[provider.value]:
                logger.error("model_not_found", provider=provider.value, model=model_name)
                return False
            
            # Switch model
            self.current_provider = provider
            self.current_model = model_name
            
            logger.info("model_switched", 
                       provider=provider.value, 
                       model=model_name)
            return True
            
        except Exception as e:
            logger.error("switch_model_error", model_key=model_key, error=str(e))
            return False
    
    async def get_available_models(self) -> Dict[str, Any]:
        """get_available_models()"""
        try:
            available_models = {}
            
            for provider_name, models in self.models.items():
                provider_models = []
                for model_name, config in models.items():
                    provider_models.append({
                        "name": model_name,
                        "max_tokens": config["max_tokens"],
                        "cost_per_1k": config["cost_per_1k"],
                        "available": True  # TODO: Add availability check
                    })
                
                available_models[provider_name] = {
                    "models": provider_models,
                    "current_model": self.current_model if self.current_provider.value == provider_name else None
                }
            
            return {
                "available_models": available_models,
                "current_provider": self.current_provider.value,
                "current_model": self.current_model
            }
            
        except Exception as e:
            logger.error("get_available_models_error", error=str(e))
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """LLM servisi istatistiklerini getir"""
        try:
            return {
                "current_model": self.current_model,
                "current_provider": self.current_provider.value,
                "initialized": self.initialized,
                "api_keys_configured": {
                    provider: bool(api_key) 
                    for provider, api_key in self.api_keys.items()
                }
            }
            
        except Exception as e:
            logger.error("get_llm_stats_error", error=str(e))
            return {"error": str(e)}
    
    async def is_healthy(self) -> bool:
        """LLM servisi sağlık kontrolü"""
        try:
            if not self.initialized:
                return False
            
            # Basit bir test isteği yap
            test_request = LLMRequest(
                prompt="Hello",
                max_tokens=10,
                temperature=0.0
            )
            
            # Test isteği yap (timeout ile)
            try:
                await asyncio.wait_for(
                    self.generate_text(test_request),
                    timeout=10.0
                )
                return True
            except asyncio.TimeoutError:
                logger.warning("llm_service_timeout")
                return False
                
        except Exception as e:
            logger.error("llm_health_check_error", error=str(e))
            return False
    
    async def cleanup(self):
        """LLM servisini temizle"""
        try:
            if self.session:
                await self.session.close()
            
            logger.info("llm_service_cleanup_completed")

        except Exception as e:
            logger.error("llm_cleanup_error", error=str(e))

    # Retry and Rate Limiting
    async def _execute_with_retry(self, operation, max_retries: int = 3, base_delay: float = 1.0):
        """Execute operation with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning("retry_attempt", 
                                 attempt=attempt + 1, 
                                 max_retries=max_retries,
                                 delay=delay,
                                 error=str(e))
                    await asyncio.sleep(delay)
                else:
                    logger.error("max_retries_exceeded", 
                               max_retries=max_retries,
                               error=str(e))
                    raise last_exception

    async def _check_rate_limit(self):
        """Rate limiting check"""
        # Simple rate limiting - can be enhanced with Redis
        current_time = datetime.now()
        
        if not hasattr(self, '_last_request_time'):
            self._last_request_time = current_time
            return
        
        time_diff = (current_time - self._last_request_time).total_seconds()
        min_interval = 0.1  # Minimum 100ms between requests
        
        if time_diff < min_interval:
            await asyncio.sleep(min_interval - time_diff)
        
        self._last_request_time = datetime.now()


# Global instance
llm_service = LLMService()
