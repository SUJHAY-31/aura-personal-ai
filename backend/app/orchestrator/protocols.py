"""Protocols defining dependencies for the AURA orchestrator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from backend.app.models.memory import ConversationTurn

if TYPE_CHECKING:
    from backend.app.orchestrator.models import ParsedAction
    from backend.app.tools.models import ToolDefinition


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


@runtime_checkable
class ActionProtocolParserProtocol(Protocol):
    """Interface for parsing LLM generation into structured actions."""

    def parse(self, raw_text: str) -> ParsedAction:
        """
        Parse raw LLM generation into a typed action.

        Returns:
            DirectResponseAction, ToolCallAction, or ParseFailureAction.
        """
        ...

    def format_tool_prompt(
        self,
        tools: list[ToolDefinition],
    ) -> str:
        """Format registered tools and action protocol instructions for prompt inclusion."""
        ...
