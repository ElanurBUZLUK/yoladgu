from __future__ import annotations

from typing import Dict, Any, Optional, List
import json, os
import structlog
from app.core.config import settings
from app.services.retriever_mcp import MCPRetrieverClient
from app.services.en_qg.repo import MistakeRepo
from app.services.en_qg.selector import choose_target
from app.services.en_qg.schemas import GeneratedQuestion, SubmitIn, NextParams, GeneratedOption
from app.services.en_qg.prompt_templates import VOCAB_GAP_TEMPLATE, GRAMMAR_FIX_TEMPLATE


logger = structlog.get_logger()


class EnQGService:
    def __init__(self, repo: Optional[MistakeRepo] = None, mcp: Optional[MCPRetrieverClient] = None):
        self.repo = repo or MistakeRepo()
        self.mcp = mcp or MCPRetrieverClient()
        self.provider = (settings.LLM_PROVIDER or "none").lower()
        self.model_id = settings.LLM_MODEL_ID or "gpt-4o-mini"
        self.temp = float(getattr(settings, "EN_QG_TEMPERATURE", 0.4))
        self.max_tokens = int(getattr(settings, "EN_QG_MAX_TOKENS", 256))
        self._client = self._init_llm_client()

    def submit(self, body: SubmitIn) -> Dict[str, Any]:
        target = (body.meta or {}).get("target", {})
        correct = body.student_answer.strip() == body.gold_answer.strip()
        try:
            if target.get("type") == "vocab":
                lemma = target.get("lemma", "")
                if lemma:
                    self.repo.inc_vocab(body.student_id, lemma, correct)
            elif target.get("type") == "grammar":
                rc = target.get("rule_code", "")
                if rc:
                    self.repo.inc_rule(body.student_id, rc, correct)
        except Exception as e:
            logger.error("enqg.submit.update_failed", error=str(e), student_id=body.student_id)
        return {"ok": True, "correct": correct}

    def next(self, p: NextParams) -> GeneratedQuestion:
        try:
            cands = self.repo.pick_candidates(p.student_id, p.mode)
            target = choose_target(cands, p.mode) or {"type": "vocab", "key": "example"}
            ctx_text = self._retrieve_context(target, p.k_ctx)
            if target["type"] == "vocab":
                q = self._build_vocab_item(lemma=target["key"], context=ctx_text, cefr=p.cefr or "A2")
            else:
                q = self._build_grammar_item(rule_code=target["key"], context=ctx_text)
            if not self._validate_question(q):
                logger.info("enqg.next.invalid_question", target=target)
                return self._fallback_question(target)
            return q
        except Exception as e:
            logger.error("enqg.next.failed", error=str(e), student_id=p.student_id, mode=p.mode)
            return self._fallback_question()

    def _retrieve_context(self, target: Dict[str, str], k_ctx: int) -> str:
        q = target["key"]
        items = self.mcp.retrieve_context(q, language="en") or []
        texts: List[str] = [it.get("text", "") for it in items][:k_ctx]
        return "\n".join([t for t in texts if t])

    def _build_vocab_item(self, lemma: str, context: str, cefr: str) -> GeneratedQuestion:
        prompt = VOCAB_GAP_TEMPLATE.format(cefr=cefr, context=context, lemma=lemma)
        data = self._call_llm_json(prompt) or {"stem": f"{lemma.capitalize()} ___.", "options": ["A","B","C","D"], "correct_index": 0, "rationale":"stub"}
        return GeneratedQuestion(
            id=f"vocab:{lemma}",
            type="vocab_gap",
            stem=data["stem"],
            options=[GeneratedOption(text=o) for o in data["options"]],
            correct_index=int(data["correct_index"]),
            rationale=data.get("rationale"),
            meta={"target":{"type":"vocab","lemma":lemma}},
        )

    def _build_grammar_item(self, rule_code: str, context: str) -> GeneratedQuestion:
        prompt = GRAMMAR_FIX_TEMPLATE.format(rule_code=rule_code, context=context)
        data = self._call_llm_json(prompt) or {"stem": f"Fix the error related to {rule_code}.", "options": ["A","B","C","D"], "correct_index": 0, "rationale":"stub"}
        return GeneratedQuestion(
            id=f"grammar:{rule_code}",
            type="grammar_fix",
            stem=data["stem"],
            options=[GeneratedOption(text=o) for o in data["options"]],
            correct_index=int(data["correct_index"]),
            rationale=data.get("rationale"),
            meta={"target":{"type":"grammar","rule_code":rule_code}},
        )

    def _validate_question(self, q: GeneratedQuestion) -> bool:
        if len(q.options) != 4:
            return False
        if q.correct_index not in (0,1,2,3):
            return False
        banned = ["credit card", "password", "ssn"]
        if any(b in q.stem.lower() for b in banned):
            return False
        return True

    def _fallback_question(self, target: Optional[Dict[str, str]] = None) -> GeneratedQuestion:
        stem = "Choose the correct option."
        return GeneratedQuestion(
            id="fallback",
            type="mcq",
            stem=stem,
            options=[GeneratedOption(text=o) for o in ["A","B","C","D"]],
            correct_index=0,
            rationale="",
            meta={"target": target or {}},
        )

    def _init_llm_client(self):
        try:
            from app.utils.vault import get_secret
        except Exception:
            def get_secret(_): return None
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=get_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))
        if self.provider == "cohere":
            import cohere
            return cohere.Client(get_secret("COHERE_API_KEY") or os.getenv("COHERE_API_KEY"))
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        if self.provider == "hf":
            from huggingface_hub import InferenceClient
            return InferenceClient(token=get_secret("HF_API_TOKEN") or os.getenv("HF_API_TOKEN"))
        if self.provider == "google":
            try:
                import google.generativeai as genai  # type: ignore
                api_key = get_secret("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    return genai.GenerativeModel(self.model_id)
            except Exception:
                return None
        return None

    def _call_llm_json(self, prompt: str) -> dict | None:
        try:
            if self.provider == "openai" and self._client is not None:
                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temp,
                    max_tokens=self.max_tokens,
                )
                content = resp.choices[0].message.content
            elif self.provider == "cohere" and self._client is not None:
                resp = self._client.generate(prompt=prompt, model=self.model_id, temperature=self.temp, max_tokens=self.max_tokens)
                content = resp.generations[0].text
            elif self.provider == "anthropic" and self._client is not None:
                msg = self._client.messages.create(model=self.model_id, max_tokens=self.max_tokens, temperature=self.temp, messages=[{"role": "user", "content": prompt}])
                content = msg.content[0].text if getattr(msg, "content", None) else ""
            elif self.provider == "hf" and self._client is not None:
                content = self._client.text_generation(prompt, model=self.model_id, temperature=self.temp, max_new_tokens=self.max_tokens)
            elif self.provider == "google" and self._client is not None:
                try:
                    resp = self._client.generate_content(prompt, generation_config={"temperature": self.temp, "max_output_tokens": self.max_tokens})  # type: ignore[attr-defined]
                    content = getattr(resp, "text", None) or getattr(resp, "candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                except Exception:
                    content = ""
            else:
                return None
            try:
                return json.loads((content or "").strip())
            except Exception:
                return None
        except Exception as e:
            logger.error("enqg.llm.failed", error=str(e))
            return None


