"""
Automated tests for AURA public health and identity endpoint (GET /).
"""

from fastapi.testclient import TestClient
import pytest

from backend.app.main import app
from backend.app.utils.metadata import APP_NAME, APP_VERSION


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a FastAPI TestClient."""
    return TestClient(app)


def test_health_root_status_code(client: TestClient) -> None:
    """GET / must return HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200


def test_health_root_payload(client: TestClient) -> None:
    """GET / must return assistant identity, status, version, and welcome message."""
    response = client.get("/")
    data = response.json()

    assert data.get("assistant") == APP_NAME
    assert data.get("assistant") == "AURA"
    assert data.get("status") == "Running"
    assert data.get("version") == APP_VERSION
    assert data.get("version") == "0.1.0"
    assert "message" in data
    assert isinstance(data["message"], str)
    assert len(data["message"].strip()) > 0
