"""
Automated tests for AURA chat endpoint (POST /chat).

Tests validation, dependency injection, mock LLM generation, and error mapping.
"""

from collections.abc import Generator
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from backend.app.ai.llm import (
    LLMConnectionError,
    LLMInferenceError,
    LLMService,
    LLMServiceError,
)
import backend.app.ai.llm as llm_module
from backend.app.api.chat import get_llm_service
from backend.app.main import app


# Define LLMTimeoutError fallback if not yet present in production module
if hasattr(llm_module, "LLMTimeoutError"):
    LLMTimeoutError = getattr(llm_module, "LLMTimeoutError")
else:
    class LLMTimeoutError(LLMServiceError):  # type: ignore[no-redef]
        """Fallback definition for testing timeout mapping."""


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provide a TestClient with dependency override cleanup."""
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_valid_chat_success(client: TestClient) -> None:
    """POST /chat with valid message returns generated response."""
    mock_service = MagicMock(spec=LLMService)
    mock_service.generate.return_value = "Hello from AURA"

    def override_get_llm_service() -> Generator[MagicMock, None, None]:
        yield mock_service

    app.dependency_overrides[get_llm_service] = override_get_llm_service

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 200
    assert response.json() == {"response": "Hello from AURA"}
    mock_service.generate.assert_called_once_with("Hello AURA")


def test_whitespace_only_message_rejected(client: TestClient) -> None:
    """POST /chat with whitespace-only message returns 422 Unprocessable Entity."""
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 422


def test_chat_llm_connection_error(client: TestClient) -> None:
    """POST /chat maps LLMConnectionError to HTTP 503."""
    mock_service = MagicMock(spec=LLMService)
    mock_service.generate.side_effect = LLMConnectionError("Could not reach Ollama")

    def override_get_llm_service() -> Generator[MagicMock, None, None]:
        yield mock_service

    app.dependency_overrides[get_llm_service] = override_get_llm_service

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 503
    assert "Could not reach Ollama" in response.json()["detail"]


def test_chat_llm_timeout_error(client: TestClient) -> None:
    """POST /chat maps LLMTimeoutError to HTTP 504."""
    mock_service = MagicMock(spec=LLMService)
    mock_service.generate.side_effect = LLMTimeoutError("Ollama inference timed out")

    def override_get_llm_service() -> Generator[MagicMock, None, None]:
        yield mock_service

    app.dependency_overrides[get_llm_service] = override_get_llm_service

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 504
    assert "Ollama inference timed out" in response.json()["detail"]


def test_chat_llm_inference_error(client: TestClient) -> None:
    """POST /chat maps LLMInferenceError to HTTP 502."""
    mock_service = MagicMock(spec=LLMService)
    mock_service.generate.side_effect = LLMInferenceError("Ollama returned HTTP 500")

    def override_get_llm_service() -> Generator[MagicMock, None, None]:
        yield mock_service

    app.dependency_overrides[get_llm_service] = override_get_llm_service

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 502
    assert "Ollama returned HTTP 500" in response.json()["detail"]


def test_chat_generic_llm_service_error(client: TestClient) -> None:
    """POST /chat maps generic LLMServiceError to HTTP 500."""
    mock_service = MagicMock(spec=LLMService)
    mock_service.generate.side_effect = LLMServiceError("Unexpected LLM failure")

    def override_get_llm_service() -> Generator[MagicMock, None, None]:
        yield mock_service

    app.dependency_overrides[get_llm_service] = override_get_llm_service

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 500
    assert "Unexpected LLM failure" in response.json()["detail"]
