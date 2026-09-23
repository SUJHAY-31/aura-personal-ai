"""Domain models for the AURA orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.config import settings
from backend.app.memory.prompt import DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class OrchestratorRequest:
    """Incoming request to process a single conversational turn."""

    message: str
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestratorResult:
    """Outcome of an orchestrated conversational turn."""

    response: str
    session_id: str
    turns_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestratorConfig:
    """Runtime configuration for orchestrator execution."""

    max_history_turns: int = settings.MEMORY_MAX_TURNS
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT
