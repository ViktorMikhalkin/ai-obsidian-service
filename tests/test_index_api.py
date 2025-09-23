import pytest
from fastapi.testclient import TestClient

from ai_obsidian_service.api.app import app


def _route_exists(path: str, method: str = "GET") -> bool:
    method = method.upper()
    for r in app.router.routes:
        if getattr(r, "path", None) == path and method in getattr(r, "methods", set()):
            return True
    return False


def test_index_stats():
    if not _route_exists("/index/stats", "GET"):
        pytest.skip("Endpoint /index/stats not implemented in this app version")
    with TestClient(app) as client:
        r = client.get("/index/stats")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        assert "total_chunks" in data
        assert "total_documents" in data
        assert "index_size_mb" in data  # New field
        assert "last_updated" in data  # New field


def test_index_rebuild_start():
    if not _route_exists("/index/rebuild", "POST"):
        pytest.skip("Endpoint /index/rebuild not implemented in this app version")
    with TestClient(app) as client:
        r = client.post("/index/rebuild", json={"timeout_sec": 0})
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        assert data.get("message") == "Index rebuild completed"
        assert "stats" in data
