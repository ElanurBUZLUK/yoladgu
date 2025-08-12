from __future__ import annotations

from typing import Protocol, List, Dict, Any


class IRetrieverService(Protocol):
    def search(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        ...


