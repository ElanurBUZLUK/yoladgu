import os, time, uuid
from typing import Dict, Any
from app.core.costs import estimate_cost_usd
from app.services.providers import openai_client, anthropic_client, vertex_client, local_vllm_client

DEFAULT_POLICY = {
    "order": [
        {"prov":"openai", "model":"gpt-4o-mini", "max_tps": 20},
        {"prov":"anthropic", "model":"claude-3-haiku"},
        {"prov":"vertex", "model":"gemini-1.5-pro"},
        {"prov":"local", "model":"vllm"}
    ],
    "timeout": 22,
    "max_usd_per_request": float(os.getenv("MAX_USD_PER_REQUEST", "0.05"))
}

PROVIDERS = {
    "openai": openai_client,
    "anthropic": anthropic_client,
    "vertex": vertex_client,
    "local": local_vllm_client,
}

class LLMRouter:
    def __init__(self, policy: Dict[str,Any] | None = None):
        self.policy = policy or DEFAULT_POLICY

    def run(self, prompt: str) -> Dict[str,Any]:
        req_id = str(uuid.uuid4())
        last_err = None
        for p in self.policy["order"]:
            prov, model = p["prov"], p["model"]
            client = PROVIDERS.get(prov)
            if not client:
                continue
            try:
                res = client.complete(model=model, prompt=prompt, timeout=self.policy.get("timeout",22))
                pt = int(res.get("usage",{}).get("prompt_tokens",0))
                ct = int(res.get("usage",{}).get("completion_tokens",0))
                cost = estimate_cost_usd(prov, model, pt, ct)
                if cost > 0 and cost > self.policy["max_usd_per_request"]:
                    last_err = RuntimeError(f"Budget exceeded: {cost}")
                    continue
                return {
                    "text": res["text"],
                    "usage": {
                        "provider": prov,
                        "model": model,
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "cost_usd": float(cost),
                        "request_id": req_id,
                    }
                }
            except Exception as e:
                last_err = e
                time.sleep(0.4)
                continue
        return {
            "text": "{\"kind\":\"mcq\",\"stem\":\"(fallback) Choose the correct option.\",\"options\":[\"A\",\"B\",\"C\",\"D\"],\"answer\":{\"index\":0}}",
            "usage": {"provider":"fallback","model":"rule","prompt_tokens":0,"completion_tokens":0,"cost_usd":0.0, "request_id": req_id}
        }