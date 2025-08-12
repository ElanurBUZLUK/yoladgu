from __future__ import annotations

from typing import Dict, Any
import requests
from app.core.config import settings


class TorchServeClient:
    def __init__(self, base_url: str | None = None, model_name: str | None = None) -> None:
        self.base_url = base_url or (settings.TS_URL or "")
        self.model = model_name or (settings.TS_MODEL_NAME or "")

    def predict(self, payload: Dict[str, Any]) -> float:
        if not self.base_url or not self.model:
            return 0.0
        url = f"{self.base_url}/predictions/{self.model}"
        try:
            resp = requests.post(url, json=payload, timeout=2.0)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "score" in data:
                return float(data["score"])      # handler returns {score: float}
            if isinstance(data, (int, float)):
                return float(data)
            return 0.0
        except Exception:
            return 0.0


