"""
AURA Orchestrator package.

Provides the central coordination engine for conversational turns, linking
memory retrieval, prompt building, LLM inference, and turn persistence.
"""

from backend.app.orchestrator.core import AuraOrchestrator
from backend.app.orchestrator.errors import (
    OrchestrationFailureError,
    OrchestratorError,
    PersistenceError,
    SessionInitializationError,
)
from backend.app.orchestrator.models import (
    OrchestratorConfig,
    OrchestratorRequest,
    OrchestratorResult,
)
from backend.app.orchestrator.protocols import (
    ConversationManagerProtocol,
    LLMServiceProtocol,
    PromptBuilderProtocol,
)

__all__ = [
    "AuraOrchestrator",
    "ConversationManagerProtocol",
    "LLMServiceProtocol",
    "OrchestrationFailureError",
    "OrchestratorConfig",
    "OrchestratorError",
    "OrchestratorRequest",
    "OrchestratorResult",
    "PersistenceError",
    "PromptBuilderProtocol",
    "SessionInitializationError",
]
