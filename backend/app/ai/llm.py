"""
Local LLM inference via Ollama HTTP API.

All Ollama-specific protocol details are confined to this module so higher
layers (planner, API routes, automation) depend on ``LLMService.generate`` only.
"""

from __future__ import annotations

from typing import Any

import httpx

from backend.app.config import Settings, settings


class LLMServiceError(Exception):
    """Base error for LLM service failures."""


class LLMConnectionError(LLMServiceError):
    """Raised when the Ollama server cannot be reached."""


class LLMInferenceError(LLMServiceError):
    """Raised when Ollama responds with an HTTP or application-level error."""


class LLMService:
    """
    Thin client for non-streaming text generation through a local Ollama server.

    Parameters are optional so tests can inject a mock ``httpx.Client`` and
    custom URLs/models without touching environment variables.
    """

    _GENERATE_PATH = "/api/generate"

    def __init__(
        self,
        *,
        ollama_url: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
        app_settings: Settings | None = None,
    ) -> None:
        cfg = app_settings or settings
        self._ollama_url = (ollama_url or cfg.OLLAMA_URL).rstrip("/")
        self._model = model or cfg.DEFAULT_MODEL
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def model(self) -> str:
        """Ollama model tag used for ``generate`` calls."""
        return self._model

    @property
    def ollama_url(self) -> str:
        """Base URL of the Ollama HTTP API."""
        return self._ollama_url

    def generate(self, prompt: str) -> str:
        """
        Send ``prompt`` to Ollama and return the model's reply text.

        Args:
            prompt: User or system-assembled input for the model.

        Returns:
            Generated text from the model response body.

        Raises:
            ValueError: If ``prompt`` is empty or whitespace-only.
            LLMConnectionError: If the Ollama server is unreachable.
            LLMInferenceError: If Ollama returns a non-success status or payload.
        """
        stripped = prompt.strip()
        if not stripped:
            raise ValueError("prompt must not be empty")

        payload = {
            "model": self._model,
            "prompt": stripped,
            "stream": False,
        }

        try:
            response = self._client.post(
                f"{self._ollama_url}{self._GENERATE_PATH}",
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise LLMConnectionError(
                f"Timed out connecting to Ollama at {self._ollama_url}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMConnectionError(
                f"Could not reach Ollama at {self._ollama_url}: {exc}"
            ) from exc

        if response.is_error:
            detail = self._extract_error_message(response)
            raise LLMInferenceError(
                f"Ollama returned HTTP {response.status_code}: {detail}"
            )

        try:
            body: dict[str, Any] = response.json()
        except ValueError as exc:
            raise LLMInferenceError(
                "Ollama returned a non-JSON response body"
            ) from exc

        text = body.get("response")
        if not isinstance(text, str):
            raise LLMInferenceError(
                "Ollama response JSON missing string field 'response'"
            )

        return text

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        """Best-effort parse of Ollama error bodies."""
        try:
            data = response.json()
        except ValueError:
            return response.text.strip() or response.reason_phrase

        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, str) and error.strip():
                return error.strip()

        text = response.text.strip()
        return text or response.reason_phrase

    def close(self) -> None:
        """Close the underlying HTTP client when this service owns it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LLMService:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
