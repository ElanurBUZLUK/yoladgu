from __future__ import annotations

from typing import Callable, Dict, Any, List, Optional
import inspect
import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


class _ToolSpec(BaseModel):
    name: str
    doc: Optional[str]
    params: List[Dict[str, Any]]


class FastMCP:
    def __init__(self, name: str):
        self.name = name
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._resources: Dict[str, Callable[..., Any]] = {}
        self._prompts: Dict[str, Callable[..., Any]] = {}
        self._app: Optional[FastAPI] = None

    # Decorators
    def tool(self):
        def _decorator(func: Callable[..., Any]):
            self._tools[func.__name__] = func
            return func
        return _decorator

    def resource(self, _pattern: str):
        def _decorator(func: Callable[..., Any]):
            self._resources[_pattern] = func
            return func
        return _decorator

    def prompt(self):
        def _decorator(func: Callable[..., Any]):
            self._prompts[func.__name__] = func
            return func
        return _decorator

    # Runtime
    def build_app(self) -> FastAPI:
        if self._app is not None:
            return self._app
        app = FastAPI(title=self.name)

        class _InvokeIn(BaseModel):
            args: Dict[str, Any] = {}

        @app.get("/mcp/tools")
        def list_tools() -> Dict[str, Any]:
            specs: List[Dict[str, Any]] = []
            for name, fn in self._tools.items():
                sig = inspect.signature(fn)
                params = []
                for p in sig.parameters.values():
                    params.append({
                        "name": p.name,
                        "kind": str(p.kind),
                        "default": None if p.default is inspect._empty else p.default,
                        "annotation": str(p.annotation) if p.annotation is not inspect._empty else None,
                    })
                specs.append(_ToolSpec(name=name, doc=(fn.__doc__ or None), params=params).model_dump())
            return {"tools": specs}

        @app.post("/mcp/tool/{name}")
        def invoke_tool(name: str, body: _InvokeIn):
            if name not in self._tools:
                return {"error": f"tool not found: {name}"}
            fn = self._tools[name]
            result = fn(**(body.args or {}))
            return {"result": result}

        # Provide a compatibility endpoint for simple clients
        @app.post("/mcp/retrieve")
        def retrieve_alias(body: _InvokeIn):
            if "retrieve_context" not in self._tools:
                return {"results": []}
            res = self._tools["retrieve_context"](**(body.args or {}))
            # normalize to {"results": [...]} if possible
            if isinstance(res, dict) and "results" in res:
                return res
            if isinstance(res, list):
                return {"results": res}
            return {"results": []}

        self._app = app
        return app

    def run(self, host: str | None = None, port: int | None = None):
        app = self.build_app()
        host = host or os.getenv("MCP_HOST", "0.0.0.0")
        port = int(port or os.getenv("MCP_PORT", "7800"))
        uvicorn.run(app, host=host, port=port)


