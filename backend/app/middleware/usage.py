**Dosya: `backend/app/middleware/usage.py`**
```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import text
from app.core.database import get_db

class UsageLogger(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        usage = getattr(request.state, "llm_usage", None)
        if usage:
            db = next(get_db())
            db.execute(text(
                """
                INSERT INTO token_usage(user_id, request_id, provider, model, prompt_tokens, completion_tokens, cost_usd)
                VALUES (:user_id, :req, :prov, :model, :pt, :ct, :cost)
                """
            ), {
                "user_id": getattr(request.state, "user_id", None),
                "req": usage.get("request_id"),
                "prov": usage["provider"],
                "model": usage["model"],
                "pt": usage.get("prompt_tokens", 0),
                "ct": usage.get("completion_tokens", 0),
                "cost": str(usage.get("cost_usd", 0)),
            })
            db.commit()
        return response