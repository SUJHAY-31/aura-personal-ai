"""Exceptions raised by the AURA orchestrator."""

from __future__ import annotations


class OrchestratorError(Exception):
    """Base exception for all orchestrator-level failures."""


class SessionInitializationError(OrchestratorError):
    """Raised when resolving or initializing a conversation session fails."""


class PersistenceError(OrchestratorError):
    """Raised when persisting conversation turns/exchanges fails."""


class OrchestrationFailureError(OrchestratorError):
    """Raised when an unexpected failure occurs during the orchestration workflow."""
