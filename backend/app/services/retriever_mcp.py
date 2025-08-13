from __future__ import annotations

from typing import List, Dict, Any, Optional
import json
import socket
import time
import http.client

from app.core.config import settings
from app.services.interfaces.retriever import IRetrieverMCP
from app.services.vector_index_manager import VectorIndexManager
from app.services.embedding_api import get_embedding_provider
from app.services.content.questions_service import QuestionsService


class MCPRetrieverClient(IRetrieverMCP):
    def __init__(self, base_url: Optional[str] = None, timeout_s: float = 2.0):
        self.base_url = base_url or getattr(settings, "MCP_BASE_URL", None) or "http://localhost:7800"
        self.timeout_s = float(timeout_s)
        # Fallback local components
        self._idx = VectorIndexManager(settings.REDIS_URL)
        self._embed = get_embedding_provider()
        self._qsvc = QuestionsService(settings.REDIS_URL)

    # --- IRetrieverService ---
    def search(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            ctx = self.retrieve_context(text, language="tr")
            if not ctx:
                return self._fallback_vector(text, k)
            # If MCP returns context docs with ids, map to metadata if possible
            out: List[Dict[str, Any]] = []
            added = 0
            for item in ctx:
                doc_id = item.get("id")
                meta = None
                if doc_id is not None:
                    meta = self._qsvc.get(int(doc_id)) or {}
                else:
                    meta = {"text": item.get("text", "")}
                out.append({"id": int(doc_id) if doc_id is not None else -1, "meta": meta, "text": meta.get("text", "")})
                added += 1
                if added >= k:
                    break
            return out[:k]
        except Exception:
            return self._fallback_vector(text, k)

    # --- IRetrieverMCP ---
    def retrieve_context(self, text: str, language: str) -> List[Dict[str, Any]]:
        try:
            # Very small HTTP client without external deps
            url = self.base_url.rstrip("/")
            host_port = url.replace("http://", "").replace("https://", "")
            if "/" in host_port:
                host_port, path_base = host_port.split("/", 1)
                path_base = "/" + path_base
            else:
                path_base = ""
            if ":" in host_port:
                host, port = host_port.split(":", 1)
                port = int(port)
            else:
                host, port = host_port, 80
            conn = http.client.HTTPConnection(host, port=port, timeout=self.timeout_s)
            payload = json.dumps({"text": text, "language": language})
            headers = {"Content-Type": "application/json"}
            # Prefer new tool-style endpoint if available
            conn.request("POST", f"{path_base}/mcp/tool/retrieve_context", body=json.dumps({"args": {"text": text, "language": language}}), headers=headers)
            resp = conn.getresponse()
            if resp.status != 200:
                # Fallback to legacy alias
                conn = http.client.HTTPConnection(host, port=port, timeout=self.timeout_s)
                conn.request("POST", f"{path_base}/mcp/retrieve", body=payload, headers=headers)
                resp = conn.getresponse()
                if resp.status != 200:
                    return []
            raw = resp.read()
            data = json.loads(raw.decode("utf-8"))
            # tool response: {"result": [...]} ; alias: {"results": [...]}
            items = data.get("result") or data.get("results") or data.get("contexts") or data
            if isinstance(items, list):
                return [i for i in items if isinstance(i, dict)]
            return []
        except (socket.timeout, ConnectionRefusedError, TimeoutError):
            return []
        except Exception:
            return []

    # --- Fallback path ---
    def _fallback_vector(self, text: str, k: int) -> List[Dict[str, Any]]:
        import numpy as np
        vec = np.array([self._embed.embed_one(text)], dtype=np.float32)
        ids, _ = self._idx.search(vec[0], k=k)
        out: List[Dict[str, Any]] = []
        for qid in ids:
            meta = self._qsvc.get(int(qid)) or {}
            out.append({"id": int(qid), "meta": meta, "text": meta.get("text", "")})
        return out


