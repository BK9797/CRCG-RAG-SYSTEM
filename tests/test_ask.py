"""Tests for the /api/v1/ask endpoint.

The RAG service is fully mocked — these tests never call Groq or Chroma.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.response import AskResponse, SourceItem


@pytest.fixture
def mock_rag_service() -> MagicMock:
    """A mocked RAGService returning a canned AskResponse."""
    service = MagicMock()
    service.is_loaded = True
    service.ask.return_value = AskResponse(
        question="What is the emergency travel assistance number?",
        answer="The 24/7 emergency travel assistance number is +20-100-234-5678.",
        sources=[SourceItem(page=5, section="Emergency Support")],
    )
    return service


@pytest.fixture
def client(mock_rag_service: MagicMock) -> TestClient:
    """Provide a TestClient with the mocked RAG service, bypassing real startup."""
    mock_settings = MagicMock()
    mock_settings.groq_model = "openai/gpt-oss-120b"

    @asynccontextmanager
    async def noop_lifespan(_app):
        _app.state.settings = mock_settings
        _app.state.rag_service = mock_rag_service
        yield

    app.router.lifespan_context = noop_lifespan

    with TestClient(app) as test_client:
        yield test_client


def test_ask_returns_200(client: TestClient) -> None:
    """The ask endpoint should return HTTP 200 for a valid question."""
    response = client.post(
        "/api/v1/ask",
        json={"question": "What is the emergency travel assistance number?"},
    )
    assert response.status_code == 200


def test_ask_answer_exists(client: TestClient) -> None:
    """The response should contain a non-empty answer field."""
    response = client.post(
        "/api/v1/ask",
        json={"question": "What is the emergency travel assistance number?"},
    )
    body = response.json()
    assert "answer" in body
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0


def test_ask_response_schema(client: TestClient) -> None:
    """The response should match the AskResponse schema exactly."""
    response = client.post(
        "/api/v1/ask",
        json={"question": "What is the emergency travel assistance number?"},
    )
    body = response.json()

    assert set(body.keys()) == {"question", "answer", "sources"}
    assert isinstance(body["sources"], list)
    if body["sources"]:
        source = body["sources"][0]
        assert "page" in source
        assert "section" in source


def test_ask_calls_service_not_groq_or_chroma(
    client: TestClient, mock_rag_service: MagicMock
) -> None:
    """The route should delegate to the mocked service exactly once."""
    client.post(
        "/api/v1/ask",
        json={"question": "What is the emergency travel assistance number?"},
    )
    mock_rag_service.ask.assert_called_once_with(
        "What is the emergency travel assistance number?"
    )


def test_ask_rejects_empty_question(client: TestClient) -> None:
    """An empty question string should fail Pydantic validation (422)."""
    response = client.post("/api/v1/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_service_unavailable_returns_503(client: TestClient, mock_rag_service: MagicMock) -> None:
    """If the RAG service isn't loaded yet, the endpoint should return 503."""
    mock_rag_service.is_loaded = False
    response = client.post(
        "/api/v1/ask",
        json={"question": "What is the emergency travel assistance number?"},
    )
    assert response.status_code == 503
