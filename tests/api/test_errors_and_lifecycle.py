from __future__ import annotations

import logging

from fastapi import HTTPException
from fastapi.testclient import TestClient


def test_validation_error_schema():
    from ai_obsidian_service.api.app import app

    with TestClient(app) as client:
        res = client.post("/search", json={})  # invalid body
        assert res.status_code == 422
        body = res.json()
        assert set(("code", "message", "requestId")).issubset(body.keys())
        assert body["code"] == "validation_error"
        assert "X-Request-ID" in res.headers


def test_http_exception_mapped(monkeypatch):
    from ai_obsidian_service.api import app as api_app_module

    app = api_app_module.app

    class StubService:
        def search_text(self, *a, **kw):
            raise HTTPException(status_code=503, detail="Service unavailable")

        def resolve_meta(self, *a, **kw):
            return {}

    app.dependency_overrides[api_app_module.get_search_service] = lambda: StubService()

    with TestClient(app) as client:
        res = client.post("/search", json={"query": "q", "top_k": 1})
        assert res.status_code == 503
        body = res.json()
        assert set(("code", "message", "requestId")).issubset(body.keys())
        assert body["code"] in ("http_503", "service_unavailable")
        assert body["message"] == "Service unavailable"

    app.dependency_overrides.pop(api_app_module.get_search_service, None)


def test_unhandled_exception_is_internal_error(monkeypatch):
    from ai_obsidian_service.api import app as api_app_module

    app = api_app_module.app

    class StubService:
        def search_text(self, *a, **kw):
            raise RuntimeError("boom")

        def resolve_meta(self, *a, **kw):
            return {}

    app.dependency_overrides[api_app_module.get_search_service] = lambda: StubService()

    with TestClient(app) as client:
        res = client.post("/search", json={"query": "q", "top_k": 1})
        assert res.status_code == 500
        body = res.json()
        assert set(("code", "message", "requestId")).issubset(body.keys())
        assert body["code"] == "internal_error"

    app.dependency_overrides.pop(api_app_module.get_search_service, None)


def test_request_id_preserved_from_header():
    from ai_obsidian_service.api.app import app

    with TestClient(app) as client:
        rid = "req-123"
        res = client.get("/health", headers={"X-Request-ID": rid})
        assert res.status_code == 200
        assert res.headers.get("X-Request-ID") == rid


def test_access_logger_emits_correlation(caplog):
    from ai_obsidian_service.api.app import app

    caplog.set_level(logging.INFO, logger="ai_obsidian_service.api.access")
    rid = "req-xyz"
    with TestClient(app) as client:
        _ = client.get("/health", headers={"X-Request-ID": rid})
    # ищем запись, куда middleware положил requestId
    assert any(getattr(rec, "requestId", None) == rid for rec in caplog.records)
