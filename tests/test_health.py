from fastapi.testclient import TestClient

from ai_obsidian_service.api.app import app


def test_health():
    # Use context manager to properly close the client
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, dict)
        # In this test setup we provide config and minimal index + mock embedder ⇒ ok should be True
        assert body.get("ok") is True
        assert body.get("errors") == []
