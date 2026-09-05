"""
Application configuration for AURA.

Centralizes environment-driven settings using Pydantic Settings (v2).
Infrastructure and domain code should import ``settings`` (or ``get_settings``)
instead of reading ``os.environ`` directly.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Typed configuration loaded from environment variables and optional ``.env``.

    Environment variables use the ``AURA_`` prefix (e.g. ``AURA_PORT=9000``).
    """

    model_config = SettingsConfigDict(
        env_prefix="AURA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Human-readable product name surfaced in logs, CLI, and client branding.
    APP_NAME: str = Field(default="AURA")

    # API and assistant release version (semver); keep in sync with deployments.
    VERSION: str = Field(default="0.1.0")

    # Bind address for the ASGI server (uvicorn). Use 0.0.0.0 to listen on all interfaces.
    HOST: str = Field(default="127.0.0.1")

    # TCP port for the HTTP API.
    PORT: int = Field(default=8000, ge=1, le=65535)

    # Base URL of the local Ollama HTTP API (no trailing path required).
    OLLAMA_URL: str = Field(default="http://127.0.0.1:11434")

    # Default Ollama model tag for inference when a request does not specify one.
    DEFAULT_MODEL: str = Field(default="qwen2.5")

    # Path to the SQLite database file for conversation memory and facts.
    DB_PATH: str = Field(default="aura.db")

    # Maximum number of recent conversation turns to retain in short-term context.
    MEMORY_MAX_TURNS: int = Field(default=10, ge=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached ``Settings`` instance (singleton).

    ``lru_cache`` ensures one load per process and allows tests to reset via
    ``get_settings.cache_clear()`` after changing environment variables.
    """
    return Settings()


# Module-level singleton for convenient imports: ``from app.config import settings``.
settings: Settings = get_settings()
