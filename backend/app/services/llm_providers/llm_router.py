import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from .base import BaseLLMProvider, LLMResponse, LLMProvider
from .openai_provider import GPT4Provider, GPT35Provider
from .anthropic_provider import ClaudeOpusProvider, ClaudeSonnetProvider, ClaudeHaikuProvider
from app.core.config import settings
from app.core.cache import cache_service


class CostController:
    """LLM maliyet kontrolü"""
    
    def __init__(self, daily_budget: float = 100.0):
        self.daily_budget = daily_budget
        self.cache_prefix = "llm_cost"
    
    async def get_daily_spend(self) -> float:
        """Günlük harcamayı getir"""
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{self.cache_prefix}:daily:{today}"
        
        daily_spend = await cache_service.get(cache_key)
        return daily_spend or 0.0
    
    async def add_cost(self, cost: float) -> bool:
        """Maliyet ekle"""
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{self.cache_prefix}:daily:{today}"
        
        current_spend = await self.get_daily_spend()
        new_spend = current_spend + cost
        
        # 24 saat TTL
        await cache_service.set(cache_key, new_spend, expire=86400)
        
        return new_spend <= self.daily_budget
    
    async def can_afford(self, estimated_cost: float) -> bool:
        """Maliyet karşılanabilir mi?"""
        current_spend = await self.get_daily_spend()
        return (current_spend + estimated_cost) <= self.daily_budget
    
    async def get_remaining_budget(self) -> float:
        """Kalan bütçe"""
        current_spend = await self.get_daily_spend()
        return max(0, self.daily_budget - current_spend)
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Kullanım istatistikleri"""
        current_spend = await self.get_daily_spend()
        remaining = await self.get_remaining_budget()
        
        return {
            "daily_budget": self.daily_budget,
            "current_spend": current_spend,
            "remaining_budget": remaining,
            "usage_percentage": (current_spend / self.daily_budget) * 100,
            "budget_exceeded": current_spend > self.daily_budget
        }


class FallbackStrategy:
    """LLM fallback stratejileri"""
    
    @staticmethod
    async def rule_based_evaluation(question: str, student_answer: str, correct_answer: str) -> Dict[str, Any]:
        """Rule-based cevap değerlendirme"""
        is_correct = student_answer.strip().lower() == correct_answer.strip().lower()
        
        return {
            "is_correct": is_correct,
            "score": 100 if is_correct else 0,
            "feedback": "Doğru!" if is_correct else f"Yanlış. Doğru cevap: {correct_answer}",
            "method": "rule_based"
        }
    
    @staticmethod
    async def template_based_generation(error_patterns: List[str], difficulty: int) -> Dict[str, Any]:
        """Template-based soru üretimi"""
        templates = {
            "past_tense": "I ____ to school yesterday. A) go B) went C) going D) goes",
            "present_perfect": "I ____ never been to Paris. A) have B) has C) had D) having",
            "vocabulary": "The ____ is very beautiful. A) house B) mouse C) horse D) course"
        }
        
        # İlk hata pattern'ına göre template seç
        pattern = error_patterns[0] if error_patterns else "past_tense"
        template = templates.get(pattern, templates["past_tense"])
        
        return {
            "id": f"template_{hash(template)}",
            "content": template,
            "question_type": "multiple_choice",
            "difficulty_level": difficulty,
            "method": "template_based"
        }


class LLMRouter:
    """LLM Provider Router ve Management"""
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.cost_controller = CostController(daily_budget=settings.daily_llm_budget)
        self.fallback_strategy = FallbackStrategy()
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Provider'ları başlat"""
        try:
            # OpenAI providers
            if settings.openai_api_key:
                self.providers["gpt4"] = GPT4Provider(settings.openai_api_key)
                self.providers["gpt35"] = GPT35Provider(settings.openai_api_key)
            
            # Anthropic providers
            if settings.anthropic_api_key:
                self.providers["claude_opus"] = ClaudeOpusProvider(settings.anthropic_api_key)
                self.providers["claude_sonnet"] = ClaudeSonnetProvider(settings.anthropic_api_key)
                self.providers["claude_haiku"] = ClaudeHaikuProvider(settings.anthropic_api_key)
        
        except Exception as e:
            print(f"Provider initialization error: {e}")
    
    async def get_provider_for_task(self, task_type: str, complexity: str = "medium") -> Optional[BaseLLMProvider]:
        """Görev tipine göre en uygun provider'ı seç"""
        
        # Task type'a göre provider önceliği
        task_preferences = {
            "question_generation": ["gpt4", "claude_sonnet", "gpt35"],
            "answer_evaluation": ["claude_haiku", "gpt35", "claude_sonnet"],
            "content_analysis": ["claude_sonnet", "gpt4", "claude_opus"],
            "quick_classification": ["claude_haiku", "gpt35"],
            "complex_reasoning": ["claude_opus", "gpt4", "claude_sonnet"]
        }
        
        # Complexity'e göre ayarlama
        if complexity == "high":
            preferred_providers = ["gpt4", "claude_opus", "claude_sonnet"]
        elif complexity == "low":
            preferred_providers = ["claude_haiku", "gpt35"]
        else:
            preferred_providers = task_preferences.get(task_type, ["gpt4", "claude_sonnet"])
        
        # Mevcut provider'ları kontrol et
        for provider_name in preferred_providers:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                
                # Maliyet kontrolü
                estimated_cost = self._estimate_cost(provider, task_type)
                if await self.cost_controller.can_afford(estimated_cost):
                    return provider
        
        return None
    
    async def generate_with_fallback(
        self,
        task_type: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        complexity: str = "medium",
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback ile LLM çağrısı"""
        
        # Primary provider'ı dene
        provider = await self.get_provider_for_task(task_type, complexity)
        
        if provider:
            try:
                response = await provider.generate_text(prompt, system_prompt, **kwargs)
                
                if response.success:
                    # Maliyeti kaydet
                    await self.cost_controller.add_cost(response.cost)
                    return {
                        "success": True,
                        "content": response.content,
                        "provider": response.provider,
                        "cost": response.cost,
                        "method": "llm"
                    }
            except Exception as e:
                print(f"LLM provider error: {e}")
        
        # Fallback stratejileri
        if task_type == "answer_evaluation":
            # Rule-based evaluation
            question = kwargs.get("question", "")
            student_answer = kwargs.get("student_answer", "")
            correct_answer = kwargs.get("correct_answer", "")
            
            if question and student_answer and correct_answer:
                result = await self.fallback_strategy.rule_based_evaluation(
                    question, student_answer, correct_answer
                )
                result["success"] = True
                return result
        
        elif task_type == "question_generation":
            # Template-based generation
            error_patterns = kwargs.get("error_patterns", [])
            difficulty = kwargs.get("difficulty_level", 3)
            
            result = await self.fallback_strategy.template_based_generation(
                error_patterns, difficulty
            )
            result["success"] = True
            return result
        
        # Son çare: hata döndür
        return {
            "success": False,
            "error": "No available providers and no fallback strategy",
            "method": "failed"
        }
    
    async def generate_structured_with_fallback(
        self,
        task_type: str,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        complexity: str = "medium",
        **kwargs
    ) -> Dict[str, Any]:
        """Structured output ile fallback"""
        
        provider = await self.get_provider_for_task(task_type, complexity)
        
        if provider:
            try:
                response = await provider.generate_structured_output(
                    prompt, schema, system_prompt, **kwargs
                )
                
                if response.success:
                    await self.cost_controller.add_cost(response.cost)
                    return {
                        "success": True,
                        "content": response.content,
                        "provider": response.provider,
                        "cost": response.cost,
                        "method": "llm_structured"
                    }
            except Exception as e:
                print(f"Structured LLM error: {e}")
        
        # Fallback: normal generation'a düş
        return await self.generate_with_fallback(
            task_type, prompt, system_prompt, complexity, **kwargs
        )
    
    def _estimate_cost(self, provider: BaseLLMProvider, task_type: str) -> float:
        """Tahmini maliyet hesapla"""
        # Task type'a göre tahmini token sayısı
        estimated_tokens = {
            "question_generation": 500,
            "answer_evaluation": 300,
            "content_analysis": 800,
            "quick_classification": 200,
            "complex_reasoning": 1000
        }
        
        tokens = estimated_tokens.get(task_type, 500)
        return provider.calculate_cost(tokens // 2, tokens // 2)  # Input/output split
    
    async def get_provider_status(self) -> Dict[str, Any]:
        """Provider durumları"""
        status = {
            "available_providers": list(self.providers.keys()),
            "cost_controller": await self.cost_controller.get_usage_stats(),
            "provider_details": {}
        }
        
        for name, provider in self.providers.items():
            status["provider_details"][name] = provider.get_provider_info()
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """Sağlık kontrolü"""
        health_status = {
            "overall_healthy": True,
            "providers": {},
            "cost_status": await self.cost_controller.get_usage_stats()
        }
        
        for name, provider in self.providers.items():
            try:
                # Basit test çağrısı
                test_response = await provider.generate_text(
                    "Test", max_tokens=10, temperature=0
                )
                health_status["providers"][name] = {
                    "healthy": test_response.success,
                    "error": test_response.error if not test_response.success else None
                }
                
                if not test_response.success:
                    health_status["overall_healthy"] = False
                    
            except Exception as e:
                health_status["providers"][name] = {
                    "healthy": False,
                    "error": str(e)
                }
                health_status["overall_healthy"] = False
        
        return health_status


# Global LLM router instance
llm_router = LLMRouter()