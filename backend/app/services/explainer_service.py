from __future__ import annotations

from typing import List, Dict, Any
import os, json, hashlib
import redis
from app.core.config import settings


class ExplainerService:
    def __init__(self):
        self.provider = (settings.LLM_PROVIDER or "none").lower()
        self.model_id = settings.LLM_MODEL_ID or "gpt-4o-mini"
        self.temp = float(getattr(settings, "EXPLAIN_TEMPERATURE", 0.2))
        self.max_tokens = int(getattr(settings, "EXPLAIN_MAX_TOKENS", 256))
        self.cache_ttl = int(getattr(settings, "EXPLAIN_CACHE_TTL_S", 3600))
        self.r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        self._client = None
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "cohere":
            import cohere
            self._client = cohere.Client(os.getenv("COHERE_API_KEY"))
        elif self.provider == "anthropic":
            # Minimal Anthropic support (messages API)
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider == "hf":
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=os.getenv("HF_API_TOKEN"))

    def _cache_key(self, student_id: int, question_id: int, context_docs: List[Dict[str, Any]]) -> str:
        material = json.dumps({
            "student_id": int(student_id),
            "question_id": int(question_id),
            "docs": [{"id": d.get("id"), "text": d.get("meta", {}).get("text")[:256]} for d in context_docs],
            "provider": self.provider,
            "model": self.model_id,
        }, ensure_ascii=False)
        hid = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return f"explain:v1:{hid}"

    def _prompt(self, question_text: str, docs: List[str]) -> str:
        context = "\n\n".join(docs)
        return (
            "Aşağıdaki soruyu öğrenci yanlış yanıtladı. "
            "Öğrenci seviyesine uygun, kısa ve anlaşılır bir açıklama ve ipucu üret. "
            "Kademeli anlatım kullan. Maksimum 5 cümle.\n\n"
            f"Soru: {question_text}\n\n"
            f"Bağlam:\n{context}\n\n"
            "Yanıt:"
        )

    def explain(self, student_id: int, question_id: int, question_text: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        key = self._cache_key(student_id, question_id, context_docs)
        cached = self.r.get(key)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass
        docs_text = []
        for d in context_docs[:5]:
            txt = (d.get("meta", {}) or {}).get("text") or ""
            if txt:
                docs_text.append(txt)
        if self.provider == "openai" and self._client is not None:
            prompt = self._prompt(question_text, docs_text)
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temp,
                max_tokens=self.max_tokens,
            )
            content = resp.choices[0].message.content
        elif self.provider == "cohere" and self._client is not None:
            prompt = self._prompt(question_text, docs_text)
            resp = self._client.generate(prompt=prompt, model=self.model_id, temperature=self.temp, max_tokens=self.max_tokens)
            content = resp.generations[0].text
        elif self.provider == "anthropic" and self._client is not None:
            prompt = self._prompt(question_text, docs_text)
            msg = self._client.messages.create(model=self.model_id, max_tokens=self.max_tokens, temperature=self.temp, messages=[{"role": "user", "content": prompt}])
            content = msg.content[0].text if getattr(msg, "content", None) else ""
        elif self.provider == "hf" and self._client is not None:
            prompt = self._prompt(question_text, docs_text)
            content = self._client.text_generation(prompt, model=self.model_id, temperature=self.temp, max_new_tokens=self.max_tokens)
        else:
            # No provider: simple fallback explanation
            content = "Benzer kavramlara odaklan: temel tanımı tekrar et, örnek çözüm üzerinde adım adım ilerle ve hata yaptığın adımı tespit et."
        out = {"explanation": content}
        try:
            self.r.setex(key, self.cache_ttl, json.dumps(out, ensure_ascii=False))
        except Exception:
            pass
        return out


