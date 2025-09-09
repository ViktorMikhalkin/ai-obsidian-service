
from fastapi.testclient import TestClient
from indexer.app import app

client = TestClient(app)

def test_index_stats():
    r = client.get("/index/stats")
    assert r.status_code == 200
    data = r.json()
    assert "exists" in data
    assert "index_path" in data

def test_index_rebuild_start():
    r = client.post("/index/rebuild", json={"timeout_sec": 0})
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "started"
    assert "pid" in data
