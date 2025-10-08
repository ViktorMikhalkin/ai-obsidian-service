from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from ai_obsidian_service.logging_utils import get_request_id, set_request_id

# ------------------------ Middlewares ------------------------


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Ensures every request has a requestId:
      - reads from X-Request-Id header if present,
      - otherwise generates UUID4,
      - stores in contextvar (logging_utils) and adds to response.
    """

    def __init__(self, app: ASGIApp, header_name: str = "X-Request-Id") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        rid = request.headers.get(self.header_name) or str(uuid.uuid4())
        # ContextVar is task-scoped; manual cleanup not required
        set_request_id(rid)
        response = await call_next(request)
        response.headers.setdefault(self.header_name, rid)
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """
    Compact access log in JSON format (logger: "aiobs"), including requestId.
    """

    def __init__(self, app: ASGIApp, logger_name: str = "aiobs") -> None:
        super().__init__(app)
        self._log = logging.getLogger(logger_name)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            extra = {
                "status_code": 500,
                "path": request.url.path,
                "method": request.method,
                "duration_ms": round(duration_ms, 3),
                "client": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "requestId": get_request_id(),
            }
            self._log.exception("access", extra=extra)
            raise

        duration_ms = (time.perf_counter() - t0) * 1000.0
        extra = {
            "status_code": response.status_code,
            "path": request.url.path,
            "method": request.method,
            "duration_ms": round(duration_ms, 3),
            "client": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "requestId": get_request_id(),
        }
        self._log.info("access", extra=extra)
        return response


# ------------------------ Unified error schema ------------------------


def _code_name(status_code: int) -> str:
    mapping = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        422: "validation_error",
        429: "too_many_requests",
        500: "internal_error",
        503: "service_unavailable",
    }
    return mapping.get(status_code, f"http_{status_code}")


async def http_exception_handler(_request: Request, exc: Exception) -> Response:
    """
    Unified HTTP error handling: always return {code:str, message:str, requestId:str|None}.
    Function signature compatible with FastAPI/mypy (accepts Exception).
    """
    if isinstance(exc, StarletteHTTPException):
        payload = {
            "code": _code_name(exc.status_code),
            "message": exc.detail if isinstance(exc.detail, str) else "HTTP error",
            "requestId": get_request_id(),
        }
        return JSONResponse(status_code=exc.status_code, content=payload)

    # Just in case (though this would be caught by unhandled handler)
    payload = {
        "code": "internal_error",
        "message": "Internal server error",
        "requestId": get_request_id(),
    }
    return JSONResponse(status_code=500, content=payload)


async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    """
    Catch unexpected exceptions: don't expose internals to the outside.
    """
    logging.getLogger("aiobs").exception(
        "unhandled_error",
        extra={"path": request.url.path, "requestId": get_request_id()},
    )
    payload = {
        "code": "internal_error",
        "message": "Internal server error",
        "requestId": get_request_id(),
    }
    return JSONResponse(status_code=500, content=payload)


# ------------------------ Wiring helpers ------------------------


def ensure_request_id_middleware(app: FastAPI) -> None:
    """Connect middleware for requestId setup."""
    app.add_middleware(RequestIdMiddleware)


def install_access_logger(app: FastAPI) -> None:
    """Connect middleware for access logging."""
    app.add_middleware(AccessLogMiddleware, logger_name="aiobs")


def install_error_handlers(app: FastAPI) -> None:
    """Connect error handlers with unified schema."""

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(
        _request: Request, _exc: RequestValidationError
    ) -> Response:
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "Request validation failed",
                "requestId": get_request_id(),
            },
        )

    # Important: register by StarletteHTTPException type,
    # while the handler function signature is (Request, Exception), see above.
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


def install(app: FastAPI) -> None:
    """
    Entry point: order matters — requestId first, then access log, then error handlers.
    """
    ensure_request_id_middleware(app)
    install_access_logger(app)
    install_error_handlers(app)
