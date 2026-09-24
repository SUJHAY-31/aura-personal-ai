"""Chat endpoints backed by the local LLM service, tool loop controller, and conversation memory."""

import re
from collections.abc import Generator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..ai.llm import (
    LLMConnectionError,
    LLMInferenceError,
    LLMService,
    LLMServiceError,
    LLMTimeoutError,
)
from ..config import settings
from ..memory.manager import ConversationManager
from ..memory.prompt import PromptBuilder
from ..orchestrator import (
    ActionProtocolParser,
    AuraOrchestrator,
    LoopLimitExceededError,
    ObservationSanitizer,
    OrchestrationFailureError,
    OrchestratorConfig,
    OrchestratorError,
    OrchestratorRequest,
    PersistenceError,
    SessionInitializationError,
    ToolLoopController,
    TurnCancelledError,
    TurnTimeoutError,
)
from ..permissions import PermissionEngine, create_default_permission_gate
from ..tools import ToolExecutor, ToolRegistry, register_builtin_tools

router = APIRouter(tags=["Chat"])

SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)(password|token|secret|api[_-]?key|auth|authorization|credential|private[_-]?key)"
)


def redact_sensitive_arguments(args: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively sanitize/redact sensitive arguments before exposing over HTTP.

    Protects credentials such as passwords, tokens, API keys, and private keys
    from leaking across the public API transport boundary while preserving
    non-sensitive operational arguments and structure for human confirmation.
    """
    sanitized: dict[str, Any] = {}
    for key, val in args.items():
        if SENSITIVE_KEY_PATTERN.search(key):
            sanitized[key] = "[REDACTED]"
        elif isinstance(val, dict):
            sanitized[key] = redact_sensitive_arguments(val)
        elif isinstance(val, list):
            sanitized[key] = [
                redact_sensitive_arguments(item) if isinstance(item, dict) else item
                for item in val
            ]
        elif isinstance(val, str):
            if ("BEGIN " in val and "PRIVATE KEY" in val) or val.strip().lower().startswith("bearer "):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = val
        else:
            sanitized[key] = val
    return sanitized


class ChatRequest(BaseModel):
    """Incoming user message for chat completion."""

    message: str = Field(
        ...,
        description="User message sent to the assistant.",
        examples=["What is AURA?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional ID of an existing conversation session. If omitted or not found, a session is initialized.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only input before calling the LLM."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be empty or whitespace-only")
        return stripped


class PendingActionResponse(BaseModel):
    """Safe, redacted descriptor of an action awaiting user confirmation."""

    tool_name: str = Field(description="Name of the registered tool requesting execution.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Sanitized and credential-redacted arguments for confirmation review.",
    )
    intent: str = Field(default="", description="Stated intent for the tool execution.")
    call_id: str = Field(description="Unique server-generated identifier for this tool call.")


class ChatResponse(BaseModel):
    """Assistant reply generated for the user message."""

    response: str = Field(
        ...,
        description="Text returned by the language model.",
        examples=["AURA is a modular AI personal assistant platform."],
    )
    session_id: str = Field(
        ...,
        description="Identifier of the conversation session.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    requires_confirmation: bool = Field(
        default=False,
        description="Indicates if an action requires explicit user confirmation before executing.",
    )
    confirmation_prompt: str | None = Field(
        default=None,
        description="Clarification or warning prompt presented to the user for confirmation.",
    )
    pending_action: PendingActionResponse | None = Field(
        default=None,
        description="Redacted, safe descriptor of the tool action awaiting confirmation.",
    )


def get_llm_service() -> Generator[LLMService, None, None]:
    """
    Provide an ``LLMService`` per request and release its HTTP client afterward.

    Override this dependency in tests to inject a mock or stub service.
    """
    llm = LLMService()
    try:
        yield llm
    finally:
        llm.close()


def get_memory_manager() -> ConversationManager:
    """Provide a ConversationManager instance for session and turn persistence."""
    return ConversationManager()


def get_prompt_builder() -> PromptBuilder:
    """Provide a PromptBuilder instance for prompt assembly."""
    return PromptBuilder()


_default_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Provide a stable singleton ToolRegistry populated with safe built-in tools."""
    global _default_registry
    if _default_registry is None:
        reg = ToolRegistry()
        register_builtin_tools(reg)
        _default_registry = reg
    return _default_registry


def get_permission_engine() -> PermissionEngine:
    """Provide the default permission engine with standard policy."""
    return create_default_permission_gate()


def get_tool_executor(
    registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
    permission_engine: Annotated[PermissionEngine, Depends(get_permission_engine)],
) -> ToolExecutor:
    """Provide a ToolExecutor wired with registry and permission gate."""
    return ToolExecutor(
        registry=registry,
        permission_gate=permission_engine,
    )


def get_action_parser() -> ActionProtocolParser:
    """Provide an ActionProtocolParser instance."""
    return ActionProtocolParser()


def get_observation_sanitizer() -> ObservationSanitizer:
    """Provide an ObservationSanitizer instance."""
    return ObservationSanitizer()


def get_tool_loop_controller(
    llm: Annotated[LLMService, Depends(get_llm_service)],
    parser: Annotated[ActionProtocolParser, Depends(get_action_parser)],
    registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
    permission_engine: Annotated[PermissionEngine, Depends(get_permission_engine)],
    tool_executor: Annotated[ToolExecutor, Depends(get_tool_executor)],
    sanitizer: Annotated[ObservationSanitizer, Depends(get_observation_sanitizer)],
) -> ToolLoopController:
    """Provide a ToolLoopController wired with LLM, parser, registry, permission engine, executor, and sanitizer."""
    return ToolLoopController(
        llm_service=llm,
        parser=parser,
        tool_registry=registry,
        permission_engine=permission_engine,
        tool_executor=tool_executor,
        sanitizer=sanitizer,
    )


def get_orchestrator(
    llm: Annotated[LLMService, Depends(get_llm_service)],
    memory: Annotated[ConversationManager, Depends(get_memory_manager)],
    prompt_builder: Annotated[PromptBuilder, Depends(get_prompt_builder)],
    tool_loop_controller: Annotated[ToolLoopController, Depends(get_tool_loop_controller)],
) -> AuraOrchestrator:
    """
    Provide an ``AuraOrchestrator`` instance for orchestrating conversational turns.

    Uses dependency injection for memory, LLM, prompt, and tool-loop components,
    ensuring request-scoped resource cleanup (e.g. LLMService HTTP client).
    """
    return AuraOrchestrator(
        memory_manager=memory,
        llm_service=llm,
        tool_loop_controller=tool_loop_controller,
        prompt_builder=prompt_builder,
        config=OrchestratorConfig(
            max_history_turns=settings.MEMORY_MAX_TURNS,
        ),
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message",
    response_description="Model-generated reply text with conversation session ID.",
    status_code=status.HTTP_200_OK,
)
def create_chat(
    request: ChatRequest,
    orchestrator: Annotated[AuraOrchestrator, Depends(get_orchestrator)],
) -> ChatResponse:
    """Generate an assistant reply for the user message via the AuraOrchestrator."""
    orchestrator_req = OrchestratorRequest(
        message=request.message,
        session_id=request.session_id,
    )

    try:
        result = orchestrator.process_turn(orchestrator_req)
    except (LLMTimeoutError, TurnTimeoutError) as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except TurnCancelledError as exc:
        raise HTTPException(
            status_code=499,
            detail=str(exc),
        ) from exc
    except LoopLimitExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except LLMConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except LLMInferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except SessionInitializationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize conversation.",
        ) from exc
    except PersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save conversation.",
        ) from exc
    except (OrchestrationFailureError, OrchestratorError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process conversation.",
        ) from exc

    pending_action_resp: PendingActionResponse | None = None
    if result.requires_confirmation and result.pending_action is not None:
        pending_action_resp = PendingActionResponse(
            tool_name=result.pending_action.tool_name,
            arguments=redact_sensitive_arguments(result.pending_action.arguments),
            intent=result.pending_action.intent,
            call_id=result.pending_action.call_id,
        )

    return ChatResponse(
        response=result.response,
        session_id=result.session_id,
        requires_confirmation=result.requires_confirmation,
        confirmation_prompt=result.confirmation_prompt,
        pending_action=pending_action_resp,
    )
