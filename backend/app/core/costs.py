import os
from decimal import Decimal

DEFAULT = {
    ("openai", "gpt-4o-mini"): {"in": Decimal("0.150"), "out": Decimal("0.600")},
    ("openai", "text-embedding-3-small"): {"in": Decimal("0.020"), "out": Decimal("0.020")},
    ("anthropic", "claude-3-haiku"): {"in": Decimal("0.250"), "out": Decimal("1.250")},
    ("vertex", "gemini-1.5-pro"): {"in": Decimal("0.250"), "out": Decimal("0.500")},
    ("local", "vllm"): {"in": Decimal("0.000"), "out": Decimal("0.000")},
}

def estimate_cost_usd(provider: str, model: str, prompt_toks: int, completion_toks: int) -> Decimal:
    t = DEFAULT.get((provider, model), {"in": Decimal("0"), "out": Decimal("0")})
    return (Decimal(prompt_toks) * t["in"] + Decimal(completion_toks) * t["out"]) / Decimal(1000)


