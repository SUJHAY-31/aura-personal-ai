"""Exceptions raised by the AURA orchestrator."""

from __future__ import annotations

from typing import Any


class OrchestratorError(Exception):
    """Base exception for all orchestrator-level failures."""


class SessionInitializationError(OrchestratorError):
    """Raised when resolving or initializing a conversation session fails."""


class PersistenceError(OrchestratorError):
    """Raised when persisting conversation turns/exchanges fails."""


class OrchestrationFailureError(OrchestratorError):
    """Raised when an unexpected failure occurs during the orchestration workflow."""


class ProtocolParseError(OrchestratorError):
    """Raised on unrecoverable internal action protocol parsing errors."""


class LoopLimitExceededError(OrchestratorError):
    """Raised when the reasoning loop exceeds its maximum allowed iterations."""

    def __init__(self, message: str, *, state: Any = None) -> None:
        super().__init__(message)
        self.state = state


class TurnTimeoutError(OrchestratorError):
    """Raised when turn execution exceeds the configured overall deadline."""

    def __init__(self, message: str, *, state: Any = None) -> None:
        super().__init__(message)
        self.state = state


class TurnCancelledError(OrchestratorError):
    """Raised when turn execution is cancelled via cancellation token."""

    def __init__(self, message: str, *, state: Any = None) -> None:
        super().__init__(message)
        self.state = state
