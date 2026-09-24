"""Domain models for the AURA orchestrator."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from backend.app.config import settings
from backend.app.memory.prompt import DEFAULT_SYSTEM_PROMPT


class ActionType(str, Enum):
    """Classification of an action extracted from language model output."""

    DIRECT_RESPONSE = "direct_response"
    TOOL_CALL = "tool_call"
    PARSE_FAILURE = "parse_failure"


@dataclass(frozen=True)
class DirectResponseAction:
    """Conversational text response with no tool invocation requested."""

    content: str


@dataclass(frozen=True)
class ToolCallAction:
    """Structured request to invoke a registered tool capability."""

    tool_name: str
    arguments: dict[str, Any]
    intent: str = ""
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class ParseFailureAction:
    """Represents a failure to parse or validate an action envelope."""

    raw_text: str
    error_message: str


ParsedAction = DirectResponseAction | ToolCallAction | ParseFailureAction


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
