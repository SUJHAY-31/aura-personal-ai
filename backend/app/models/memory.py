"""
Typed domain models for conversation memory.

Independent of FastAPI, HTTP transport, and LLM runtime.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConversationSession(BaseModel):
    """Represents a conversation thread/session in AURA."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Unique identifier for the session (UUID).")
    title: str = Field(default="New Conversation", description="Human-readable session title.")
    created_at: datetime | str = Field(description="Timestamp when the session was created.")
    updated_at: datetime | str = Field(description="Timestamp when the session was last updated.")


class ConversationTurn(BaseModel):
    """Represents a single message turn (user, assistant, or system) within a session."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Unique identifier for the turn (UUID).")
    session_id: str = Field(description="Foreign key pointing to the owning conversation session.")
    role: Literal["user", "assistant", "system"] = Field(
        description="Role of the turn author; enforced by database CHECK constraint."
    )
    content: str = Field(description="Raw text content of the message.")
    token_count: int = Field(
        default=0,
        ge=0,
        description="Estimated or reported token count for the turn content.",
    )
    created_at: datetime | str = Field(description="Timestamp when the turn was recorded.")
