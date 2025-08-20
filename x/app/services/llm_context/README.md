# LLM Context Management Module

Bu modül, LLM'e gönderilecek bağlamları yapısal olarak yönetir ve prompt mühendisliğini kolaylaştırır.

## Mimari

```
llm_context/
├── __init__.py                 # Modül giriş noktası
├── schemas/                    # Context şemaları
│   ├── __init__.py
│   ├── base_context.py         # Temel context sınıfı
│   └── cloze_question_context.py # Cloze soru üretimi context'i
├── builders/                   # Context builder'lar
│   ├── __init__.py
│   ├── base_context_builder.py # Temel builder sınıfı
│   └── cloze_question_context_builder.py # Cloze soru builder'ı
└── templates/                  # Jinja2 şablonları
    ├── __init__.py
    └── cloze_question_generation.jinja2
```

## Kullanım

### 1. Context Şeması Oluşturma

```python
from app.services.llm_context.schemas.cloze_question_context import ClozeQuestionGenerationContext

context = ClozeQuestionGenerationContext(
    task_definition="5 adet cloze sorusu oluştur",
    num_questions=5,
    difficulty_level=3,
    user_id="user_123",
    user_error_patterns=["past_tense", "articles"],
    grammar_rules=["Past tense rule", "Article rule"]
)
```

### 2. Context Builder Kullanma

```python
from app.services.llm_context.builders.cloze_question_context_builder import ClozeQuestionContextBuilder

builder = ClozeQuestionContextBuilder(retriever, error_pattern_repo)
context = await builder.build(
    session=db_session,
    user_id="user_123",
    num_questions=3,
    last_n_errors=5
)
```

### 3. Prompt Oluşturma

```python
# Context'ten prompt oluştur
prompt = context.to_prompt()
system_prompt = context.to_system_prompt()

# LLM'e gönder
result = await llm_gateway.generate_json(
    prompt=prompt,
    system_prompt=system_prompt,
    schema=context.output_schema
)
```

## Özellikler

### ✅ Tamamlanmış

1. **Bağlam Şemalarını (Context Schemas) Tanımlama**
   - `BaseContext` temel sınıfı
   - `ClozeQuestionGenerationContext` özel context'i
   - Pydantic validation

2. **Bağlam Oluşturucu ("Context Builder") Katmanı**
   - `BaseContextBuilder` temel sınıfı
   - `ClozeQuestionContextBuilder` implementasyonu
   - Veri toplama ve context oluşturma

3. **Prompt Şablon Motoru (Template Engine) Entegrasyonu**
   - Jinja2 template desteği
   - Template dosyalarından yükleme
   - Context verileriyle doldurma

4. **EnglishClozeService Yeniden Yapılandırma**
   - ContextBuilder kullanımı
   - Template-based prompt oluşturma
   - Orkestrasyon pattern'i

5. **LLM Gateway Genelleştirme**
   - `generate_json` metodu context desteği
   - String prompt ve context parametreleri

6. **Testler**
   - Unit testler (context şemaları ve builder'lar)
   - Integration testler (EnglishClozeService)
   - Mock ve fixture'lar

### 🔄 Geliştirilebilir

1. **Daha Fazla Context Türü**
   - Math question generation context
   - Answer evaluation context
   - Feedback generation context

2. **Template Yönetimi**
   - Template versioning
   - A/B testing için template varyasyonları
   - Template performance monitoring

3. **Context Caching**
   - Context cache'leme
   - Cache invalidation stratejileri

## Test Çalıştırma

```bash
# Unit testler
pytest tests/unit/test_context_schemas.py -v
pytest tests/unit/test_context_builders.py -v

# Integration testler
pytest tests/integration/test_english_cloze_service_integration.py -v
```

## Bağımlılıklar

- `jinja2>=3.1.0` - Template engine
- `pydantic>=2.5.0` - Data validation
- `sqlalchemy[asyncio]>=2.0.0` - Database operations
