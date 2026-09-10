"""
Unit tests for pure PromptBuilder and deterministic prompt formatting.
"""

from datetime import datetime
import pytest

from backend.app.memory.prompt import (
    DEFAULT_SYSTEM_PROMPT,
    PromptBuilder,
    build_prompt,
)
from backend.app.models.memory import ConversationTurn


def test_build_prompt_empty_history_defaults() -> None:
    """Verify default prompt structure with empty history."""
    builder = PromptBuilder()
    prompt = builder.build_prompt("Hello AURA")

    expected = (
        f"System: {DEFAULT_SYSTEM_PROMPT}\n\n"
        "User: Hello AURA\n\n"
        "Assistant:"
    )
    assert prompt == expected


def test_build_prompt_custom_system_prompt() -> None:
    """Verify custom system prompt replaces default."""
    builder = PromptBuilder()
    prompt = builder.build_prompt(
        "Who are you?",
        system_prompt="You are an expert coding assistant.",
    )

    expected = (
        "System: You are an expert coding assistant.\n\n"
        "User: Who are you?\n\n"
        "Assistant:"
    )
    assert prompt == expected


def test_build_prompt_disabled_system_prompt() -> None:
    """Verify system prompt can be omitted by passing empty string."""
    builder = PromptBuilder()
    prompt = builder.build_prompt(
        "Hello",
        system_prompt="",
    )

    expected = "User: Hello\n\nAssistant:"
    assert prompt == expected


def test_build_prompt_with_multi_turn_history() -> None:
    """Verify multi-turn history is formatted chronologically before current message."""
    now = datetime.now()
    history = [
        ConversationTurn(
            id="t1",
            session_id="s1",
            role="user",
            content="My name is Alex.",
            token_count=5,
            created_at=now,
        ),
        ConversationTurn(
            id="t2",
            session_id="s1",
            role="assistant",
            content="Nice to meet you, Alex! How can I help you today?",
            token_count=12,
            created_at=now,
        ),
    ]

    builder = PromptBuilder()
    prompt = builder.build_prompt("What is my name?", history=history)

    expected = (
        f"System: {DEFAULT_SYSTEM_PROMPT}\n\n"
        "User: My name is Alex.\n\n"
        "Assistant: Nice to meet you, Alex! How can I help you today?\n\n"
        "User: What is my name?\n\n"
        "Assistant:"
    )
    assert prompt == expected


def test_build_prompt_deterministic_output() -> None:
    """Verify multiple invocations with identical parameters produce identical output."""
    now = datetime.now()
    history = [
        ConversationTurn(
            id="t1",
            session_id="s1",
            role="user",
            content="Ping",
            created_at=now,
        ),
        ConversationTurn(
            id="t2",
            session_id="s1",
            role="assistant",
            content="Pong",
            created_at=now,
        ),
    ]

    out1 = build_prompt("Status?", history=history, system_prompt="System instructions")
    out2 = build_prompt("Status?", history=history, system_prompt="System instructions")
    assert out1 == out2


def test_build_prompt_rejects_empty_message() -> None:
    """Verify build_prompt raises ValueError on empty or whitespace-only messages."""
    builder = PromptBuilder()
    with pytest.raises(ValueError, match="current_message must not be empty"):
        builder.build_prompt("   ")
