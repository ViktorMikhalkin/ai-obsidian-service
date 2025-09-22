from __future__ import annotations
import logging
from fastapi.testclient import TestClient

from ai_obsidian_service.logging_utils import JsonFormatter, install_json_logging, install_request_id_filter

def test_uvicorn_loggers_have_json_formatter():
    install_json_logging(logger_names=["uvicorn.error", "uvicorn.access"])
    for name in ("uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        assert any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers if h.formatter)

def test_request_id_mdc_filter_propagates(caplog):
    # ensure filter on root
    install_request_id_filter(logger_names=[""])

    from ai_obsidian_service.api.app import app
    caplog.set_level(logging.INFO)

    rid = "mdc-123"
    with TestClient(app) as client:
        res = client.get("/health", headers={"X-Request-ID": rid})
        assert res.status_code == 200

    # our /health logs to ai_obsidian_service.app without extra; filter must inject requestId
    recs = [r for r in caplog.records if r.name == "ai_obsidian_service.app"]
    assert recs, "no app log captured"
    assert any(getattr(r, "requestId", None) == rid for r in recs), "requestId was not injected by MDC filter"
