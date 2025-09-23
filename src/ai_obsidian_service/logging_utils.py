from __future__ import annotations

import contextvars
import datetime
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

__all__ = [
    "JsonFormatter",
    "install_json_logging",
    "install_request_id_filter",
    "set_request_id",
    "clear_request_id",
    "get_request_id",
    "configure_logging",
]

# ContextVar for MDC-like request id propagation
_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

def set_request_id(rid: str | None) -> None:
    _request_id_var.set(rid)

def clear_request_id() -> None:
    _request_id_var.set(None)

def get_request_id() -> str | None:
    return _request_id_var.get()

_BASE_FIELDS = {
    "name","msg","args","levelname","levelno","pathname","filename","module",
    "exc_info","exc_text","stack_info","lineno","funcName","created","msecs",
    "relativeCreated","thread","threadName","processName","process","message",
}

def _is_jsonable(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None

def _to_jsonable(obj: Any) -> Any:
    if _is_jsonable(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return str(obj)

class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for Python logging."""
    def __init__(self, *, time_key: str = "ts"):
        super().__init__()
        self.time_key = time_key

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        payload: dict[str, Any] = {
            self.time_key: datetime.datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.message,
        }
        # attach requestId if present on record (from filter) or leave as-is if supplied via extra
        rid = getattr(record, "requestId", None)
        if rid is not None:
            payload["requestId"] = rid

        # include source hints for debug
        if record.levelno <= logging.DEBUG:
            payload.update(module=record.module, func=record.funcName, line=record.lineno)

        extras = {k: v for k, v in record.__dict__.items() if k not in _BASE_FIELDS}
        if extras:
            for k, v in extras.items():
                if k == "requestId":  # already handled
                    continue
                payload[k] = _to_jsonable(v)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

class RequestIdFilter(logging.Filter):
    """Injects requestId from ContextVar into every record if not already provided."""
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "requestId"):
            rid = get_request_id()
            if rid is not None:
                record.requestId = rid
        return True

def install_json_logging(
    *, level: int = logging.INFO, logger_names: Sequence[str] | None = None, include_uvicorn: bool = True
) -> None:
    """Install a JSON StreamHandler on selected loggers (and uvicorn.* when enabled)."""
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())

    targets = list(logger_names or [])
    if include_uvicorn:
        targets.extend(["uvicorn.error", "uvicorn.access"])

    if targets:
        for name in targets:
            logger = logging.getLogger(name)
            # prevent duplicate formatter installation
            if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers if h.formatter):
                logger.addHandler(handler)
            logger.setLevel(level)
            logger.propagate = False
    else:
        root = logging.getLogger()
        if not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers if h.formatter):
            root.addHandler(handler)
        root.setLevel(level)

def install_request_id_filter(logger_names: Sequence[str] | None = None) -> None:
    """Attach RequestIdFilter to selected loggers and root so every record gets requestId if available."""
    filt = RequestIdFilter()
    if logger_names:
        for name in logger_names:
            logger = logging.getLogger(name)
            if not any(isinstance(f, RequestIdFilter) for f in logger.filters):
                logger.addFilter(filt)
    else:
        root = logging.getLogger()
        if not any(isinstance(f, RequestIdFilter) for f in root.filters):
            root.addFilter(filt)

def configure_logging(*, json_enabled: bool, level: int, log_uvicorn: bool) -> None:
    """Single entrypoint to configure logging across the app."""
    if json_enabled:
        install_json_logging(
            level=level,
            logger_names=["ai_obsidian_service.api.access"],
            include_uvicorn=log_uvicorn,
        )
    # Always install requestId filter to root and key loggers
    install_request_id_filter(logger_names=["", "uvicorn.error", "uvicorn.access", "ai_obsidian_service"])
