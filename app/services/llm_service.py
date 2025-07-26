import os
import json
import requests
import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class LLMService:
    """Hibrit LLM servisi - Batch ve Runtime işlemleri için"""

    def __init__(self, provider: LLMProvider = LLMProvider.OPENAI):
        self.provider = provider
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.hf_model = os.getenv("HUGGINGFACE_MODEL", "gpt2")

    def _get_api_key(self) -> Optional[str]:
        """API anahtarını al"""
        if self.provider == LLMProvider.OPENAI:
            return self.openai_key
        else:
            return self.hf_token

    def _get_base_url(self) -> str:
        """Base URL'i al"""
        if self.provider == LLMProvider.OPENAI:
            return "https://api.openai.com/v1"
        else:
            return "https://api-inference.huggingface.co"

    async def generate_text(self, prompt: str) -> Dict[str, Any]:
        """Metin üret"""
        try:
            if self.provider == LLMProvider.OPENAI:
                return await self._call_openai(prompt)
            else:
                return await self._call_huggingface(prompt)
        except Exception as e:
            logger.error(f"LLM çağrısı başarısız: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": "AI servisi şu anda kullanılamıyor."
            }

    async def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """OpenAI API çağrısı"""
        if not self.openai_key:
            raise ValueError("OpenAI API anahtarı bulunamadı")
        
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{self._get_base_url()}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return {
            "success": True,
            "text": result["choices"][0]["message"]["content"].strip(),
            "provider": "openai"
        }

    async def _call_huggingface(self, prompt: str) -> Dict[str, Any]:
        """HuggingFace API çağrısı"""
        if not self.hf_token:
            raise ValueError("HuggingFace API anahtarı bulunamadı")
        
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        url = f"{self._get_base_url()}/models/{self.hf_model}"
        
        data = {"inputs": prompt}
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list) and result:
            text = result[0].get("generated_text", "").strip()
        elif isinstance(result, dict) and "generated_text" in result:
            text = result["generated_text"].strip()
        else:
            text = str(result)
        
        return {
            "success": True,
            "text": text,
            "provider": "huggingface"
        }

    # Batch işlemleri (İçe aktarma sırasında)
    async def generate_question_hint(self, question_content: str, subject: str) -> str:
        """Soru için ipucu üret (Batch)"""
        prompt = f"""
        Aşağıdaki soru için kısa ve etkili bir ipucu üret:
        
        Soru: {question_content}
        Konu: {subject}
        
        İpucu (2-3 cümle, öğrenciyi yönlendirici):
        """
        
        result = await self.generate_text(prompt)
        return result["text"] if result["success"] else "İpucu üretilemedi."

    async def generate_question_explanation(self, question_content: str, correct_answer: str, subject: str) -> str:
        """Soru için açıklama üret (Batch)"""
        prompt = f"""
        Aşağıdaki soru ve cevap için detaylı açıklama üret:
        
        Soru: {question_content}
        Doğru Cevap: {correct_answer}
        Konu: {subject}
        
        Açıklama (adım adım çözüm):
        """
        
        result = await self.generate_text(prompt)
        return result["text"] if result["success"] else "Açıklama üretilemedi."

    async def analyze_question_difficulty(self, question_content: str, subject: str) -> Dict[str, Any]:
        """Soru zorluğunu analiz et (Batch)"""
        prompt = f"""
        Aşağıdaki sorunun zorluk seviyesini analiz et:
        
        Soru: {question_content}
        Konu: {subject}
        
        JSON formatında yanıtla:
        {{
            "difficulty_level": 1-5 arası sayı,
            "required_knowledge": ["gerekli konular"],
            "solution_steps": adım sayısı,
            "grade_level": "sınıf seviyesi",
            "explanation": "zorluk açıklaması"
        }}
        """
        
        result = await self.generate_text(prompt)
        if result["success"]:
            try:
                # JSON parse etmeye çalış
                analysis = json.loads(result["text"])
                return analysis
            except:
                # Fallback: Basit analiz
                return {
                    "difficulty_level": 3,
                    "required_knowledge": [subject],
                    "solution_steps": 3,
                    "grade_level": "9-10",
                    "explanation": "Orta zorlukta soru"
                }
        else:
            return {
                "difficulty_level": 3,
                "required_knowledge": [subject],
                "solution_steps": 3,
                "grade_level": "9-10",
                "explanation": "Zorluk analizi yapılamadı"
            }

    # Runtime işlemleri (Öğrenci etkileşimi sırasında)
    async def adjust_difficulty_runtime(self, question_content: str, student_level: int,
                                      current_difficulty: int, student_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Runtime'da soru zorluğunu öğrenci performansına göre ayarla"""
        prompt = f"""
        Öğrenci performansına göre soru zorluğunu ayarla:
        
        Soru: {question_content}
        Öğrenci Seviyesi: {student_level}
        Mevcut Zorluk: {current_difficulty}
        Performans Verileri: {student_performance}
        
        JSON formatında yanıtla:
        {{
            "adjusted_difficulty": 1-5 arası sayı,
            "reason": "ayarlama nedeni",
            "confidence": 0.0-1.0 arası güven,
            "recommended_next_questions": ["önerilen konular"]
        }}
        """
        
        result = await self.generate_text(prompt)
        if result["success"]:
            try:
                adjustment = json.loads(result["text"])
                return adjustment
            except:
                # Fallback: Basit mantık
                accuracy = student_performance.get('recent_accuracy', 0.5)
                if accuracy > 0.8 and current_difficulty < 5:
                    return {"adjusted_difficulty": current_difficulty + 1, "reason": "Yüksek performans"}
                elif accuracy < 0.4 and current_difficulty > 1:
                    return {"adjusted_difficulty": current_difficulty - 1, "reason": "Düşük performans"}
                else:
                    return {"adjusted_difficulty": current_difficulty, "reason": "Performans uygun"}
        else:
            return {"adjusted_difficulty": current_difficulty, "reason": "AI servisi kullanılamıyor"}

    async def generate_adaptive_hint(self, question_content: str, student_level: int,
                                   previous_hints_used: int, student_struggling: bool) -> str:
        """Öğrenci durumuna göre adaptif ipucu üret (Runtime)"""
        prompt = f"""
        Öğrenci durumuna göre adaptif ipucu üret:
        
        Soru: {question_content}
        Öğrenci Seviyesi: {student_level}
        Kullanılan İpucu Sayısı: {previous_hints_used}
        Zorlanıyor mu: {student_struggling}
        
        İpucu (öğrenci durumuna uygun):
        """
        
        result = await self.generate_text(prompt)
        return result["text"] if result["success"] else "Adaptif ipucu üretilemedi."

    async def generate_contextual_explanation(self, question_content: str, correct_answer: str,
                                            student_answer: str, student_level: int) -> str:
        """Öğrencinin cevabına göre bağlamsal açıklama üret (Runtime)"""
        prompt = f"""
        Öğrencinin cevabına göre bağlamsal açıklama üret:
        
        Soru: {question_content}
        Doğru Cevap: {correct_answer}
        Öğrenci Cevabı: {student_answer}
        Öğrenci Seviyesi: {student_level}
        
        Açıklama (öğrencinin hatasına odaklan):
        """
        
        result = await self.generate_text(prompt)
        return result["text"] if result["success"] else "Bağlamsal açıklama üretilemedi."

    # Eski metodlar (geriye uyumluluk için)
    def get_hint(self, question: str) -> str:
        """Eski ipucu metodu (senkron)"""
        import asyncio
        return asyncio.run(self.generate_question_hint(question, "mathematics"))

    def get_explanation(self, question: str, answer: str) -> str:
        """Eski açıklama metodu (senkron)"""
        import asyncio
        return asyncio.run(self.generate_question_explanation(question, answer, "mathematics"))

    def analyze_difficulty(self, question: str) -> str:
        """Eski zorluk analizi metodu (senkron)"""
        import asyncio
        analysis = asyncio.run(self.analyze_question_difficulty(question, "mathematics"))
        return f"Seviye {analysis.get('difficulty_level', 1)} - {analysis.get('explanation', 'Analiz edilemedi')}"


# Global instance
llm_service = LLMService() 