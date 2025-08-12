from __future__ import annotations

import json
from typing import Dict, Any


def handle(data: bytes, context: Any) -> bytes:
    try:
        req = json.loads(data.decode("utf-8"))
        # Placeholder model logic: simple sum weighted features
        uf: Dict[str, float] = req.get("user_features", {}) or {}
        qf: Dict[str, float] = req.get("question_features", {}) or {}
        score = 0.0
        for v in list(uf.values())[:16]:
            score += float(v)
        for v in list(qf.values())[:16]:
            score += 0.5 * float(v)
        return json.dumps({"score": float(score)}).encode("utf-8")
    except Exception:
        return json.dumps({"score": 0.0}).encode("utf-8")


