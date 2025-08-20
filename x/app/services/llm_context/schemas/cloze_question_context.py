from typing import List, Optional, Dict, Any
from pydantic import Field
from .base_context import BaseContext


class ClozeQuestionGenerationContext(BaseContext):
    """Cloze soru üretimi için özel context"""
    
    # Task Definition Context
    num_questions: int = Field(..., description="Üretilecek soru sayısı")
    difficulty_level: int = Field(..., ge=1, le=5, description="Zorluk seviyesi")
    
    # User Context
    user_id: str = Field(..., description="Kullanıcı ID")
    user_error_patterns: List[str] = Field(default_factory=list, description="Kullanıcının hata pattern'ları")
    user_level: Optional[int] = Field(None, description="Kullanıcının seviyesi")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="Kullanıcı tercihleri")
    
    # Knowledge Context
    grammar_rules: List[str] = Field(default_factory=list, description="İlgili grammar kuralları")
    vocabulary_context: Optional[str] = Field(None, description="Kelime bilgisi bağlamı")
    topic_context: Optional[str] = Field(None, description="Konu bağlamı")
    
    # Output Format Context
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Çıktı JSON şeması")
    
    def to_prompt(self) -> str:
        """Cloze soru üretimi için prompt oluştur"""
        
        from ..templates import load_template
        
        try:
            template_content = load_template("cloze_question_generation")
            return self.render_template(template_content)
        except FileNotFoundError:
            # Fallback template
            template_content = """
{{ task_definition }}

Kullanıcı Bilgileri:
- Kullanıcı ID: {{ user_id }}
- Seviye: {{ user_level or "Belirlenmemiş" }}
- Hata Yaptığı Alanlar: {{ ", ".join(user_error_patterns) if user_error_patterns else "Genel grammar" }}

{% if grammar_rules %}
İlgili Grammar Kuralları:
{% for rule in grammar_rules %}
- {{ rule }}
{% endfor %}
{% endif %}

{% if vocabulary_context %}
Kelime Bilgisi Bağlamı:
{{ vocabulary_context }}
{% endif %}

{% if topic_context %}
Konu Bağlamı:
{{ topic_context }}
{% endif %}

Soru Özellikleri:
- Soru Sayısı: {{ num_questions }}
- Zorluk Seviyesi: {{ difficulty_level }}/5

{% if output_format_context %}
{{ output_format_context }}
{% endif %}
"""
            return self.render_template(template_content)
    
    def to_system_prompt(self) -> str:
        """Cloze soru üretimi için sistem prompt'u oluştur"""
        
        return """Sen deneyimli bir İngilizce öğretmenisin. 
Öğrencilerin hata yaptığı alanlara odaklanarak kişiselleştirilmiş cloze soruları oluşturuyorsun.

Soru oluştururken şu kriterlere dikkat et:
- Öğrencinin hata yaptığı alanlara odaklan
- Zorluk seviyesine uygun ol
- Eğitici ve net ol
- Türkçe açıklamalar içer
- Doğru cevap şıklarda olmasın

Her soru için şu bilgileri sağla:
- Orijinal cümle
- Cloze cümlesi (boşluklu)
- Doğru cevap
- Yanlış şıklar (distractors)
- Açıklama
- Odaklanılan hata türü"""
