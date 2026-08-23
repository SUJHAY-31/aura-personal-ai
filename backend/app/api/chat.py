"""Chat endpoints backed by the local LLM service."""

from collections.abc import Generator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ..ai.llm import LLMConnectionError, LLMInferenceError, LLMService, LLMServiceError

router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):
    """Incoming user message for chat completion."""

    message: str = Field(
        ...,
        description="User message sent to the assistant.",
        examples=["What is AURA?"],
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


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message",
    response_description="Model-generated reply text.",
    status_code=status.HTTP_200_OK,
)
def create_chat(
    request: ChatRequest,
    llm: Annotated[LLMService, Depends(get_llm_service)],
) -> ChatResponse:
    """Generate an assistant reply for the given user message."""
    try:
        text = llm.generate(request.message)
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

    return ChatResponse(response=text)
