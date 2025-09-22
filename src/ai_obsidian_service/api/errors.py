from __future__ import annotations

import time
import uuid
import logging
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ai_obsidian_service.logging_utils import set_request_id, clear_request_id, get_request_id

__all__ = [
    "install_error_handlers",
    "ensure_request_id_middleware",
    "install_access_logger",
    "status_code_to_code",
]

def status_code_to_code(status_code: int) -> str:
    mapping = {
        400: "bad_request",
        401: "unauthenticated",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        422: "validation_error",
        429: "rate_limited",
        500: "internal_error",
    }
    return mapping.get(status_code, f"http_{status_code}")

class _RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = rid
        set_request_id(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            clear_request_id()

def _content_length(response) -> Optional[int]:
    try:
        h = response.headers.get("content-length")
        if h is not None:
            return int(h)
    except Exception:
        return None
    size = None
    body = getattr(response, "body", None)
    if isinstance(body, (bytes, bytearray)):
        size = len(body)
    elif isinstance(body, str):
        size = len(body.encode("utf-8", "ignore"))
    return size

class _AccessLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, logger_name: str = "ai_obsidian_service.api.access"):
        super().__init__(app)
        self._logger = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        rid = get_request_id() or request.headers.get("X-Request-ID") or "-"
        method = request.method
        path = request.url.path
        try:
            response = await call_next(request)
            status = response.status_code
            size = _content_length(response)
            return response
        finally:
            dur_ms = (time.perf_counter() - start) * 1000.0
            extra = {
                "requestId": rid,
                "method": method,
                "path": path,
                "duration_ms": round(dur_ms, 3),
                "status": locals().get("status", 0),
                "size_bytes": locals().get("size", None),
            }
            self._logger.info("access", extra=extra)

def ensure_request_id_middleware(app: FastAPI) -> None:
    for m in app.user_middleware:
        if getattr(m, "cls", None) is _RequestIdMiddleware:
            break
    else:
        app.add_middleware(_RequestIdMiddleware)

def install_access_logger(app: FastAPI) -> None:
    for m in app.user_middleware:
        if getattr(m, "cls", None) is _AccessLogMiddleware:
            break
    else:
        app.add_middleware(_AccessLogMiddleware)

def _rid(request: Request) -> str:
    rid = getattr(getattr(request, "state", object()), "request_id", None)
    if rid:
        return str(rid)
    return request.headers.get("X-Request-ID") or uuid.uuid4().hex

def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http_exc_handler(request: Request, exc: HTTPException):
        rid = _rid(request)
        payload = {
            "code": status_code_to_code(exc.status_code),
            "message": str(exc.detail) if exc.detail else "HTTP error",
            "requestId": rid,
        }
        return JSONResponse(status_code=exc.status_code, content=payload, headers={"X-Request-ID": rid})

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        rid = _rid(request)
        payload = {"code": "validation_error", "message": "Validation failed", "requestId": rid}
        return JSONResponse(status_code=422, content=payload, headers={"X-Request-ID": rid})

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        rid = _rid(request)
        payload = {"code": "internal_error", "message": "Internal Server Error", "requestId": rid}
        return JSONResponse(status_code=500, content=payload, headers={"X-Request-ID": rid})
