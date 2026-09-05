"""
Unit tests for AURA application configuration.
"""

import pytest

from backend.app.config import Settings, get_settings


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default configuration values without custom environment variables."""
    # Ensure no environment overrides affect the defaults test
    monkeypatch.delenv("AURA_APP_NAME", raising=False)
    monkeypatch.delenv("AURA_VERSION", raising=False)
    monkeypatch.delenv("AURA_HOST", raising=False)
    monkeypatch.delenv("AURA_PORT", raising=False)
    monkeypatch.delenv("AURA_OLLAMA_URL", raising=False)
    monkeypatch.delenv("AURA_DEFAULT_MODEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.APP_NAME == "AURA"
    assert settings.VERSION == "0.1.0"
    assert settings.HOST == "127.0.0.1"
    assert settings.PORT == 8000
    assert settings.OLLAMA_URL == "http://127.0.0.1:11434"


def test_settings_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that environment variables with AURA_ prefix override settings."""
    monkeypatch.setenv("AURA_APP_NAME", "AURA_TEST")
    monkeypatch.setenv("AURA_VERSION", "1.0.0")
    monkeypatch.setenv("AURA_HOST", "0.0.0.0")
    monkeypatch.setenv("AURA_PORT", "9000")
    monkeypatch.setenv("AURA_OLLAMA_URL", "http://ollama-host:11434")

    settings = Settings(_env_file=None)

    assert settings.APP_NAME == "AURA_TEST"
    assert settings.VERSION == "1.0.0"
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 9000
    assert settings.OLLAMA_URL == "http://ollama-host:11434"


def test_default_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify DEFAULT_MODEL can be overridden to qwen2.5:7b via AURA_DEFAULT_MODEL."""
    monkeypatch.setenv("AURA_DEFAULT_MODEL", "qwen2.5:7b")

    settings = Settings(_env_file=None)

    assert settings.DEFAULT_MODEL == "qwen2.5:7b"


def test_get_settings_caching() -> None:
    """Verify get_settings returns cached singleton and cache_clear refreshes it."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()
