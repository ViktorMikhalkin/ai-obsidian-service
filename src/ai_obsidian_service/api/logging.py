"""Logging utilities."""

from __future__ import annotations

import logging
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Request tracking
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


def log_structured(level: str, message: str, **kwargs):
    """Structured logging helper that adds request_id automatically."""
    request_id = request_id_ctx.get("")
    fields = {"request_id": request_id, **kwargs}
    field_str = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    log_msg = f"{message} | {field_str}" if field_str else message
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(log_msg)
