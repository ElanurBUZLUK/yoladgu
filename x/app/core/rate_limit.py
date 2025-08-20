from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse

def rate_key_func(request: Request):
    user = getattr(request.state, "user_id", None)
    ip = get_remote_address(request)
    return f"{user or 'anon'}:{ip}"

limiter = Limiter(key_func=rate_key_func, default_limits=["120/minute", "2000/hour"])

async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={
        "detail": "Too many requests",
        "retry_after": exc.retry_after
    })