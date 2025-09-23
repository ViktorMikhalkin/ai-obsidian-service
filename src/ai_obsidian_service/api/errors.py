from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from ai_obsidian_service.logging_utils import get_request_id, set_request_id


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Ensure every request has a requestId:
      - read from X-Request-Id if provided,
      - otherwise generate a UUID4,
      - expose it via logging_utils contextvar and add response header.
    """

    def __init__(self, app: ASGIApp, header_name: str = "X-Request-Id") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rid = request.headers.get(self.header_name) or str(uuid.uuid4())
        # Note: we purposefully do NOT clear the contextvar here after the call.
        # ContextVars are request-task scoped; leaving it set avoids order issues
        # with other middlewares (e.g., access logging) and wonâ€™t leak across requests.
        set_request_id(rid)
        response = await call_next(request)
        response.headers.setdefault(self.header_name, rid)
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """
    Compact, JSON-friendly access log that includes requestId.
    """

    def __init__(self, app: ASGIApp, logger_name: str = "aiobs") -> None:
        super().__init__(app)
        self._log = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
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
            "requestId": get_request_id(),  # LogRecord has this attribute (tests expect it)
        }
        self._log.info("access", extra=extra)
        return response


async def http_exception_handler(_request: Request, exc: Exception) -> Response:
    """Handle HTTP exceptions with proper typing for FastAPI."""
    if isinstance(exc, HTTPException):
        payload = {
            "code": exc.status_code,
            "message": exc.detail,
            "requestId": get_request_id(),
        }
        return JSONResponse(status_code=exc.status_code, content=payload)
    else:
        # Fallback for non-HTTP exceptions
        payload = {
            "code": 500,
            "message": "Internal Server Error",
            "requestId": get_request_id(),
        }
        return JSONResponse(status_code=500, content=payload)


async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    """Handle unhandled exceptions with proper typing for FastAPI."""
    logging.getLogger("aiobs").exception(
        "unhandled_error",
        extra={"path": request.url.path, "requestId": get_request_id()},
    )
    payload = {
        "code": 500,
        "message": "Internal Server Error",
        "requestId": get_request_id(),
    }
    return JSONResponse(status_code=500, content=payload)


def ensure_request_id_middleware(app: FastAPI) -> None:
    """Add RequestIdMiddleware to the app."""
    app.add_middleware(RequestIdMiddleware)


def install_access_logger(app: FastAPI) -> None:
    """Add AccessLogMiddleware to the app."""
    app.add_middleware(AccessLogMiddleware, logger_name="aiobs")


def install_error_handlers(app: FastAPI) -> None:
    """Install exception handlers with proper typing."""
    # Both handlers now accept Exception and return Response, satisfying FastAPI's requirements
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


def install(app: FastAPI) -> None:
    """
    Wire up middlewares and exception handlers. Keep this thin.
    Order: request-id first (so everyone sees it), then access logging.
    """
    ensure_request_id_middleware(app)
    install_access_logger(app)
    install_error_handlers(app)
