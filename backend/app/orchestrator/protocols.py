"""Protocols defining dependencies for the AURA orchestrator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from backend.app.models.memory import ConversationTurn


@runtime_checkable
class ConversationManagerProtocol(Protocol):
    """Protocol for conversation session resolution and memory persistence."""

    def get_or_create_session(self, session_id: str | None = None) -> Any:
        """Resolve an existing session or initialize a new one."""
        ...

    def get_recent_turns(
        self, session_id: str, limit: int = 10
    ) -> Sequence[ConversationTurn]:
        """Fetch the most recent turns for the specified session in chronological order."""
        ...

    def add_exchange(
        self, session_id: str, user_content: str, assistant_content: str
    ) -> Any:
        """Persist a user-assistant exchange to the session."""
        ...


@runtime_checkable
class PromptBuilderProtocol(Protocol):
    """Protocol for assembling LLM prompt strings."""

    def build_prompt(
        self,
        current_message: str,
        *,
        history: Sequence[ConversationTurn] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Assemble a deterministic prompt string from history and current input."""
        ...


@runtime_checkable
class LLMServiceProtocol(Protocol):
    """Protocol for invoking language model text generation."""

    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""
        ...
