import importlib


def test_main_module_exports_fastapi_app():
    """Test that main module directly exports a FastAPI app."""
    mod = importlib.import_module("ai_obsidian_service.main")
    assert hasattr(mod, "app")

    # Verify it's a FastAPI app
    from fastapi import FastAPI

    assert isinstance(mod.app, FastAPI)


def test_fastapi_app_has_expected_routes():
    """Test that the FastAPI app has the required endpoints."""
    mod = importlib.import_module("ai_obsidian_service.main")

    # Extract route paths
    route_paths = {route.path for route in mod.app.routes}

    # Check for expected endpoints
    assert "/health" in route_paths
    assert "/search" in route_paths
    assert "/answer" in route_paths


def test_health_endpoint_responds():
    """Test that the health endpoint works."""
    from fastapi.testclient import TestClient

    mod = importlib.import_module("ai_obsidian_service.main")

    client = TestClient(mod.app)
    response = client.get("/health")
    assert response.status_code == 200
