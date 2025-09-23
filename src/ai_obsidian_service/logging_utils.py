from __future__ import annotations

import contextlib
import contextvars
import datetime
import json
import logging
from collections.abc import Iterable, Sequence
from typing import Any

# Python 3.11+: datetime.UTC exists; older: fallback to timezone.utc
_TZ = getattr(datetime, "UTC", datetime.UTC)

# ------------ request-id context ------------

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aiobs_request_id",
    default=None,
)

def set_request_id(rid: str | None) -> None:
    """Set the request id into a task-local contextvar."""
    _request_id_var.set(rid)

def get_request_id() -> str | None:
    """Get the current request id (or None if not set)."""
    return _request_id_var.get()

@contextlib.contextmanager
def with_request_id(rid: str | None):
    """
    Temporarily bind a request id for the current task.
    Useful in jobs/tests that want scoped correlation.
    """
    token = _request_id_var.set(rid)
    try:
        yield
    finally:
        _request_id_var.reset(token)

# ------------ logging filter & formatter ------------

class RequestIdFilter(logging.Filter):
    """
    Injects `requestId` attribute into LogRecord from the contextvar.
    Tests look for LogRecord.requestId specifically.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            record.requestId = get_request_id()
        except Exception:
            record.requestId = None
        return True


class JsonFormatter(logging.Formatter):
    """
    Minimal JSON formatter.

    - ISO 8601 time with timezone awareness (no deprecated utcfromtimestamp).
    - Includes arbitrary "extra" fields if present on the LogRecord.
    - Always carries `requestId` if available (via RequestIdFilter).
    """

    def __init__(self, *, extra_keys: Iterable[str] | None = None) -> None:
        super().__init__()
        self._extra_keys = tuple(extra_keys or ())

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": datetime.datetime.fromtimestamp(record.created, _TZ).isoformat(),
        }

        rid = getattr(record, "requestId", None)
        if rid is not None:
            payload["requestId"] = rid

        # Common HTTP-ish attributes + user-configured keys
        for key in ("status_code", "path", "method", "duration_ms", "client", "user_agent", *self._extra_keys):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)

# ------------ installers ------------

def install_json_logging(
        *,
        level: int = logging.INFO,
        logger_names: Sequence[str] | None = None,
        include_uvicorn: bool = True,
        extra_keys: Iterable[str] | None = None,
) -> None:
    """
    Configure root (and optionally uvicorn) to use JSON logging and carry requestId.

    - Attaches a single StreamHandler with JsonFormatter to the root logger.
    - Adds RequestIdFilter so LogRecord.requestId is always present.
    - Clears pre-existing handlers on target loggers and lets them propagate to root.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(extra_keys=extra_keys))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    targets: list[str] = list(logger_names or [])
    if include_uvicorn:
        targets.extend(["uvicorn", "uvicorn.error", "uvicorn.access"])

    for name in targets:
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.handlers.clear()
        lg.propagate = True  # bubble to root where our handler/filter live

def install_request_id_filter(logger_names: Sequence[str] | None = None) -> None:
    """
    Add RequestIdFilter to existing handlers on selected loggers (and root).
    Useful if tooling attached its own handlers before our install_json_logging.
    """
    filt = RequestIdFilter()
    targets = [logging.getLogger()] + [logging.getLogger(n) for n in (logger_names or ())]
    for lg in targets:
        for h in lg.handlers:
            h.addFilter(filt)

# ------------ optional utilities (kept small) ------------

def configure_logger(
        name: str,
        *,
        level: int = logging.INFO,
        use_json: bool = True,
        extra_keys: Iterable[str] | None = None,
        add_request_id_filter: bool = True,
) -> logging.Logger:
    """
    Create/refresh a named logger with a local handler (bypassing root).
    Prefer `install_json_logging` for global config; use this for ad-hoc loggers.
    """
    lg = logging.getLogger(name)
    lg.setLevel(level)
    lg.handlers.clear()

    handler = logging.StreamHandler()
    if use_json:
        handler.setFormatter(JsonFormatter(extra_keys=extra_keys))
    if add_request_id_filter:
        handler.addFilter(RequestIdFilter())
    lg.addHandler(handler)
    lg.propagate = False
    return lg

def get_json_logger(name: str) -> logging.Logger:
    """
    Return a logger that propagates to root (assumes install_json_logging was called).
    Handy for libs: no local handlers; root controls format/filters.
    """
    lg = logging.getLogger(name)
    lg.propagate = True
    return lg
