from __future__ import annotations

import contextlib
import contextvars
import datetime
import json
import logging
import os
from collections.abc import Iterable, Sequence
from typing import Any

# Python 3.11+: datetime.UTC exists; fallback kept for clarity
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


# ------------ filter & formatter ------------


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
    Minimal JSON formatter with ISO 8601 timestamps and requestId propagation.
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
        for key in (
            "status_code",
            "path",
            "method",
            "duration_ms",
            "client",
            "user_agent",
            *self._extra_keys,
        ):
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
    - Ensures uvicorn.error / uvicorn.access have a local handler with JsonFormatter
      (тест явно проверяет наличие форматтера на самих uvicorn.* логгерах).
    """
    # Root handler
    level_name = os.getenv("AIOS_LOG_LEVEL", "").upper()
    if level is None and level_name:
        level = getattr(logging, level_name, logging.INFO)
    if level is None:
        level = logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    root_handler = logging.StreamHandler()
    root_handler.setFormatter(JsonFormatter(extra_keys=extra_keys))
    root_handler.addFilter(RequestIdFilter())
    root.addHandler(root_handler)

    # Ensure named loggers have JsonFormatter locally (if required by tests)
    targets: list[str] = list(logger_names or [])
    if include_uvicorn:
        targets.extend(["uvicorn.error", "uvicorn.access"])

    for name in targets:
        lg = logging.getLogger(name)
        lg.setLevel(level)

        # Не удаляем существующие хэндлеры (вдруг их добавил рантайм/интеграция),
        # но гарантируем хотя бы один JsonFormatter на самом логгере:
        has_json = any(
            isinstance(h.formatter, JsonFormatter) for h in lg.handlers if h.formatter
        )
        if not has_json:
            h = logging.StreamHandler()
            h.setFormatter(JsonFormatter(extra_keys=extra_keys))
            h.addFilter(RequestIdFilter())
            lg.addHandler(h)

        # Пусть записи ещё и всплывают на root (единая конфигурация/агрегация)
        lg.propagate = True


def install_request_id_filter(logger_names: Sequence[str] | None = None) -> None:
    """
    Add RequestIdFilter to existing handlers on selected loggers (and root).
    Useful if tooling attached its own handlers before our install_json_logging.
    """
    filt = RequestIdFilter()
    targets = [logging.getLogger()] + [
        logging.getLogger(n) for n in (logger_names or ())
    ]
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
