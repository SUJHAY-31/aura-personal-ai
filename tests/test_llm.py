"""
Unit tests for LLMService isolated from live Ollama server.
"""

from unittest.mock import MagicMock

import httpx
import pytest

from backend.app.ai.llm import (
    LLMConnectionError,
    LLMInferenceError,
    LLMService,
    LLMServiceError,
)
import backend.app.ai.llm as llm_module

# Fallback definition if LLMTimeoutError is not yet implemented in production
if hasattr(llm_module, "LLMTimeoutError"):
    LLMTimeoutError = getattr(llm_module, "LLMTimeoutError")
else:
    class LLMTimeoutError(LLMServiceError):  # type: ignore[no-redef]
        """Fallback definition for timeout testing."""


def test_generate_success() -> None:
    """LLMService.generate returns response text on successful Ollama call."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = httpx.Response(
        status_code=200,
        json={"response": "AURA SUCCESS"},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    mock_client.post.return_value = mock_response

    service = LLMService(
        ollama_url="http://127.0.0.1:11434",
        model="qwen2.5:7b",
        client=mock_client,
    )
    result = service.generate("Hello AURA")

    assert result == "AURA SUCCESS"


def test_generate_endpoint_and_payload() -> None:
    """Verify endpoint path and JSON payload structure sent to Ollama."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = httpx.Response(
        status_code=200,
        json={"response": "AURA SUCCESS"},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    mock_client.post.return_value = mock_response

    service = LLMService(
        ollama_url="http://127.0.0.1:11434",
        model="qwen2.5:7b",
        client=mock_client,
    )
    service.generate("Hello AURA")

    mock_client.post.assert_called_once()
    called_url, kwargs = mock_client.post.call_args
    assert called_url[0] == "http://127.0.0.1:11434/api/generate"
    assert kwargs.get("json") == {
        "model": "qwen2.5:7b",
        "prompt": "Hello AURA",
        "stream": False,
    }


def test_ollama_http_error() -> None:
    """Mock HTTP 500 response raises LLMInferenceError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = httpx.Response(
        status_code=500,
        json={"error": "model 'qwen2.5:7b' not found"},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    mock_client.post.return_value = mock_response

    service = LLMService(client=mock_client)
    with pytest.raises(LLMInferenceError) as exc_info:
        service.generate("Hello AURA")

    assert "500" in str(exc_info.value)
    assert "model 'qwen2.5:7b' not found" in str(exc_info.value)


def test_connection_failure() -> None:
    """Mock httpx.ConnectError raises LLMConnectionError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = httpx.ConnectError(
        "Connection refused",
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    service = LLMService(client=mock_client)
    with pytest.raises(LLMConnectionError) as exc_info:
        service.generate("Hello AURA")

    assert "Could not reach Ollama" in str(exc_info.value)


def test_read_timeout() -> None:
    """Mock httpx.ReadTimeout raises LLMTimeoutError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = httpx.ReadTimeout(
        "Read timed out",
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    service = LLMService(client=mock_client)
    with pytest.raises(LLMTimeoutError):
        service.generate("Hello AURA")


def test_missing_response_field() -> None:
    """Successful HTTP response missing 'response' string field raises LLMInferenceError."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = httpx.Response(
        status_code=200,
        json={"done": True},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    mock_client.post.return_value = mock_response

    service = LLMService(client=mock_client)
    with pytest.raises(LLMInferenceError) as exc_info:
        service.generate("Hello AURA")

    assert "missing string field 'response'" in str(exc_info.value)


def test_url_normalization() -> None:
    """Ollama URL with trailing slash is normalized to prevent double slashes."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = httpx.Response(
        status_code=200,
        json={"response": "OK"},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    mock_client.post.return_value = mock_response

    service = LLMService(
        ollama_url="http://127.0.0.1:11434/",
        client=mock_client,
    )
    service.generate("Test prompt")

    called_url = mock_client.post.call_args[0][0]
    assert called_url == "http://127.0.0.1:11434/api/generate"
    assert "//api" not in called_url


def test_timeout_configuration() -> None:
    """Verify default timeout configuration: read 120s, connect 10s."""
    service = LLMService()
    try:
        client_timeout = service._client.timeout
        assert isinstance(client_timeout, httpx.Timeout)
        assert client_timeout.read == 120.0
        assert client_timeout.connect == 10.0
    finally:
        service.close()
