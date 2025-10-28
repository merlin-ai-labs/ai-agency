"""Tests for main FastAPI application.

TODO:
- Add tests for POST /runs
- Add tests for GET /runs/{id}
- Add integration tests for flow execution
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_healthz(client):
    """Test health check endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ai-agency"}


def test_create_run_stub(client):
    """Test creating a run (stub implementation)."""
    payload = {
        "flow_name": "maturity_assessment",
        "tenant_id": "test_tenant",
        "input_data": {"test": True},
    }

    response = client.post("/runs", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert "run_id" in data


def test_get_run_stub(client):
    """Test getting run status (stub implementation)."""
    response = client.get("/runs/test_run_id")
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert "status" in data


# TODO: Add more tests
# - Test invalid flow_name
# - Test missing tenant_id
# - Test authentication (when implemented)
# - Test rate limiting (when implemented)
