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


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message",
    response_description="Model-generated reply text with conversation session ID.",
    status_code=status.HTTP_200_OK,
)
def create_chat(
    request: ChatRequest,
    llm: Annotated[LLMService, Depends(get_llm_service)],
    memory: Annotated[ConversationManager, Depends(get_memory_manager)],
    prompt_builder: Annotated[PromptBuilder, Depends(get_prompt_builder)],
) -> ChatResponse:
    """Generate an assistant reply for the user message within a persistent conversation session."""
    try:
        session = memory.get_or_create_session(session_id=request.session_id)
        history = memory.get_recent_turns(
            session_id=session.id,
            limit=settings.MEMORY_MAX_TURNS,
        )
        prompt = prompt_builder.build_prompt(
            current_message=request.message,
            history=history,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize conversation.",
        ) from exc

    try:
        text = llm.generate(prompt)
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

    try:
        memory.add_exchange(
            session_id=session.id,
            user_content=request.message,
            assistant_content=text,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save conversation.",
        ) from exc

    return ChatResponse(response=text, session_id=session.id)
