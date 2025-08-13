from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
from app.mcp.fastmcp import FastMCP
from app.core.config import settings
from app.services.vector_index_manager import VectorIndexManager
from app.services.embedding_api import get_embedding_provider
from app.services.content.questions_service import QuestionsService


mcp = FastMCP("RAG MCP Service")


@mcp.tool()
def retrieve_context(text: str, language: str = "tr", k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k contexts for a query using the active vector index."""
    idx = VectorIndexManager(settings.REDIS_URL)
    embed = get_embedding_provider()
    qsvc = QuestionsService(settings.REDIS_URL)
    vec = np.array([embed.embed_one(text)], dtype=np.float32)
    ids, _ = idx.search(vec[0], k=max(5, k))
    out: List[Dict[str, Any]] = []
    for qid in ids[:k]:
        meta = qsvc.get(int(qid)) or {}
        out.append({"id": int(qid), "text": meta.get("text", ""), "meta": meta})
    return out


if __name__ == "__main__":
    mcp.run()


