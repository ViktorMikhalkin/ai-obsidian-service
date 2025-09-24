
from __future__ import annotations

import contextvars
import datetime
import json
import logging
import os
from collections.abc import Iterable
from typing import Any

_TZ = getattr(datetime, "UTC", datetime.UTC)

# ------------ request-id context ------------

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

def set_request_id(value: str | None) -> None:
    _request_id.set(value)

def get_request_id() -> str | None:
    return _request_id.get()

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        rid = get_request_id()
        if rid is not None:
            record.requestId = rid  # type: ignore[attr-defined]
        return True

# ------------ JSON logging ------------

class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter with ISO 8601 timestamps and requestId propagation."""
    def __init__(self, *, extra_keys: Iterable[str] | None = None) -> None:
        super().__init__()
        self._extra_keys = tuple(extra_keys or ())

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": datetime.datetime.now(tz=_TZ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = getattr(record, "requestId", None)
        if rid:
            payload["requestId"] = rid
        for k in self._extra_keys:
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        return json.dumps(payload, ensure_ascii=False)

def configure_logging(level: int | None = None) -> None:
    level_name = os.getenv("AIOS_LOG_LEVEL", "").upper()
    if level is None and level_name:
        level = getattr(logging, level_name, logging.INFO)
    if level is None:
        level = logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(extra_keys=("latency_ms",)))
    handler.addFilter(RequestIdFilter())
    root.handlers.clear()
    root.addHandler(handler)

def get_json_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.propagate = True
    return lg
