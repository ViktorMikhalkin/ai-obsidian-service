import pytest
from fastapi.testclient import TestClient
from indexer.app import app

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
        assert "exists" in data
        assert "index_path" in data

def test_index_rebuild_start():
    if not _route_exists("/index/rebuild", "POST"):
        pytest.skip("Endpoint /index/rebuild not implemented in this app version")
    with TestClient(app) as client:
        r = client.post("/index/rebuild", json={"timeout_sec": 0})
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        assert data.get("status") == "started"
        assert "pid" in data