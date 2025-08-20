import os
import httpx
from typing import Optional

DICT_ENDPOINT = os.getenv("DICT_API", "https://api.dictionaryapi.dev/api/v2/entries/en/")

class MCPWeb:
    @staticmethod
    def dictionary(term: str) -> Optional[dict]:
        url = f"{DICT_ENDPOINT}{term}"
        try:
            with httpx.Client(timeout=5.0) as c:
                r = c.get(url)
                if r.status_code == 200:
                    return r.json()
        except Exception:
            return None
        return None

    @staticmethod
    def cefr_estimate(text: str) -> str:
        words = len(text.split())
        if words < 8: return "A1"
        if words < 20: return "A2"
        if words < 40: return "B1"
        if words < 80: return "B2"
        return "C1"