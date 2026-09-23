"""Chat endpoints backed by the local LLM service and conversation memory."""

from collections.abc import Generator
from typing import Annotated

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
    AuraOrchestrator,
    OrchestrationFailureError,
    OrchestratorConfig,
    OrchestratorError,
    OrchestratorRequest,
    PersistenceError,
    SessionInitializationError,
)

router = APIRouter(tags=["Chat"])


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


def get_orchestrator(
    llm: Annotated[LLMService, Depends(get_llm_service)],
    memory: Annotated[ConversationManager, Depends(get_memory_manager)],
    prompt_builder: Annotated[PromptBuilder, Depends(get_prompt_builder)],
) -> AuraOrchestrator:
    """
    Provide an ``AuraOrchestrator`` instance for orchestrating conversational turns.

    Uses dependency injection for memory, LLM, and prompt components,
    ensuring request-scoped resource cleanup (e.g. LLMService HTTP client).
    """
    return AuraOrchestrator(
        memory_manager=memory,
        llm_service=llm,
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
    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
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

    return ChatResponse(
        response=result.response,
        session_id=result.session_id,
    )
