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
    ProtocolParseError,
    SessionInitializationError,
)
from backend.app.orchestrator.models import (
    ActionType,
    DirectResponseAction,
    Observation,
    OrchestratorConfig,
    OrchestratorRequest,
    OrchestratorResult,
    ParsedAction,
    ParseFailureAction,
    ToolCallAction,
)
from backend.app.orchestrator.parser import ActionProtocolParser
from backend.app.orchestrator.protocols import (
    ActionProtocolParserProtocol,
    ConversationManagerProtocol,
    LLMServiceProtocol,
    ObservationSanitizerProtocol,
    PromptBuilderProtocol,
)
from backend.app.orchestrator.sanitizer import ObservationSanitizer

__all__ = [
    "ActionProtocolParser",
    "ActionProtocolParserProtocol",
    "ActionType",
    "AuraOrchestrator",
    "ConversationManagerProtocol",
    "DirectResponseAction",
    "LLMServiceProtocol",
    "Observation",
    "ObservationSanitizer",
    "ObservationSanitizerProtocol",
    "OrchestrationFailureError",
    "OrchestratorConfig",
    "OrchestratorError",
    "OrchestratorRequest",
    "OrchestratorResult",
    "ParseFailureAction",
    "ParsedAction",
    "PersistenceError",
    "PromptBuilderProtocol",
    "ProtocolParseError",
    "SessionInitializationError",
    "ToolCallAction",
]
