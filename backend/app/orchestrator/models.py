"""Domain models for the AURA orchestrator."""

from __future__ import annotations

import threading
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
class Observation:
    """
    Sanitized representation of a tool execution outcome fed to the reasoning layer.

    Wraps untrusted tool data with deterministic envelope delimiters and metadata.
    ObservationSanitizer returns this container.
    Observation.to_context_string() is the required method for producing the
    final untrusted wrapper before the LLM sees the content.
    The final LLM context must never receive raw ToolResult.output.
    """

    tool_name: str
    status: str
    content: str
    raw_length: int
    duration_ms: float = 0.0
    is_truncated: bool = False
    is_sanitization_failure: bool = False
    error_message: str | None = None

    def to_context_string(self) -> str:
        """Format as an untrusted XML-style observation block for prompt context."""
        header = f'<observation tool="{self.tool_name}" status="{self.status}">'
        footer = "</observation>"
        return f"{header}\n{self.content}\n{footer}"


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
    requires_confirmation: bool = False
    confirmation_prompt: str | None = None
    pending_action: ToolCallAction | None = None


@dataclass
class StepRecord:
    """
    Observable audit record of a single step in the reasoning loop.

    Explicitly excludes hidden model reasoning traces or chain-of-thought.
    """

    iteration: int
    action: ParsedAction
    observation: Observation | None = None
    intent: str = ""


@dataclass
class LoopState:
    """
    Isolated state container for a single multi-step tool reasoning turn.
    """

    iteration_count: int = 0
    max_iterations: int = 5
    deadline_monotonic: float = 0.0
    cancellation_token: threading.Event | None = None
    observations: list[Observation] = field(default_factory=list)
    step_records: list[StepRecord] = field(default_factory=list)
    executed_invocation_ids: list[str] = field(default_factory=list)
    call_signatures: dict[str, int] = field(default_factory=dict)
    cached_results: dict[str, Observation] = field(default_factory=dict)
    pending_action: ToolCallAction | None = None
    is_paused: bool = False


@dataclass(frozen=True)
class ToolLoopConfig:
    """Configuration for the tool reasoning loop."""

    max_iterations: int = 5
    overall_timeout_seconds: float = 60.0
    duplicate_call_threshold: int = 2


@dataclass(frozen=True)
class OrchestratorConfig:
    """Runtime configuration for orchestrator execution."""

    max_history_turns: int = settings.MEMORY_MAX_TURNS
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT
