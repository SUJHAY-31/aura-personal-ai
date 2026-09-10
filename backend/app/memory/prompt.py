"""
Pure prompt assembly for AURA LLM inference.

Constructs predictable, deterministic prompt strings from optional system instructions,
turn history, and current user input without side effects, HTTP calls, or database dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence

from backend.app.models.memory import ConversationTurn

DEFAULT_SYSTEM_PROMPT = "You are AURA, an advanced, helpful, and concise personal AI assistant."


class PromptBuilder:
    """
    Pure builder for formatting system instructions, multi-turn history,
    and latest user input into an LLM-ready prompt.
    """

    def __init__(self, default_system_prompt: str | None = DEFAULT_SYSTEM_PROMPT) -> None:
        self.default_system_prompt = default_system_prompt

    def build_prompt(
        self,
        current_message: str,
        *,
        history: Sequence[ConversationTurn] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Assemble a deterministic prompt string.

        Args:
            current_message: The latest user input message.
            history: Optional sequence of prior ConversationTurn records.
            system_prompt: Optional custom system instructions. If None, uses default.
                           Set to empty string ("") to omit system prompt completely.

        Returns:
            The formatted prompt string for LLM generation.

        Raises:
            ValueError: If current_message is empty or whitespace-only.
        """
        stripped_message = current_message.strip()
        if not stripped_message:
            raise ValueError("current_message must not be empty or whitespace-only")

        sections: list[str] = []

        # System prompt resolution
        active_system = system_prompt if system_prompt is not None else self.default_system_prompt
        if active_system is not None and active_system.strip():
            sections.append(f"System: {active_system.strip()}")

        # Turn history resolution
        if history:
            for turn in history:
                role_label = turn.role.capitalize()
                sections.append(f"{role_label}: {turn.content.strip()}")

        # Current user message and Assistant reply cue
        sections.append(f"User: {stripped_message}")
        sections.append("Assistant:")

        return "\n\n".join(sections)


def build_prompt(
    current_message: str,
    *,
    history: Sequence[ConversationTurn] | None = None,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Convenience functional wrapper for PromptBuilder().build_prompt()."""
    builder = PromptBuilder(default_system_prompt=system_prompt)
    return builder.build_prompt(
        current_message=current_message,
        history=history,
        system_prompt=system_prompt,
    )
