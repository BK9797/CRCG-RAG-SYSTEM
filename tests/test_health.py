"""Tests for the /api/v1/health endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Provide a TestClient with a mocked RAG service in app.state.

    Overrides the app's lifespan with a no-op so the test never tries to
    load ChromaDB or validate GROQ_API_KEY against real resources. State
    is set directly so routes can read app.state.rag_service / .settings.
    """
    mock_settings = MagicMock()
    mock_settings.groq_model = "openai/gpt-oss-120b"

    mock_rag_service = MagicMock()
    mock_rag_service.is_loaded = True

    @asynccontextmanager
    async def noop_lifespan(_app):
        _app.state.settings = mock_settings
        _app.state.rag_service = mock_rag_service
        yield

    app.router.lifespan_context = noop_lifespan

    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_200(client: TestClient) -> None:
    """The health endpoint should return HTTP 200."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_status_healthy(client: TestClient) -> None:
    """The health endpoint should report status 'healthy' when loaded."""
    response = client.get("/api/v1/health")
    body = response.json()
    assert body["status"] == "healthy"


def test_health_response_schema(client: TestClient) -> None:
    """The health response should match the expected schema."""
    response = client.get("/api/v1/health")
    body = response.json()
    assert "status" in body
    assert "vectorstore_loaded" in body
    assert "model" in body
    assert body["vectorstore_loaded"] is True
    assert body["model"] == "openai/gpt-oss-120b"
