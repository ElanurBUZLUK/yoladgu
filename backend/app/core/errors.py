from typing import Any, Dict
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


def json_error_response(status_code: int, code: str, message: str, details: Dict[str, Any] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
            }
        },
    )


async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return json_error_response(exc.status_code, code="http_error", message=str(exc.detail))


async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return json_error_response(422, code="validation_error", message="Invalid request", details={"errors": exc.errors()})


async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    # Avoid leaking internal information
    return json_error_response(500, code="internal_error", message="Internal server error")


