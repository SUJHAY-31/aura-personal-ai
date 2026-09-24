"""
AURA Orchestrator package.

Provides the central coordination engine for conversational turns, linking
memory retrieval, prompt building, LLM inference, and turn persistence.
"""

from backend.app.orchestrator.core import AuraOrchestrator
from backend.app.orchestrator.errors import (
    LoopLimitExceededError,
    OrchestrationFailureError,
    OrchestratorError,
    PersistenceError,
    ProtocolParseError,
    SessionInitializationError,
    TurnCancelledError,
    TurnTimeoutError,
)
from backend.app.orchestrator.loop import (
    ToolLoopController,
    compute_call_signature,
)
from backend.app.orchestrator.models import (
    ActionType,
    DirectResponseAction,
    LoopState,
    Observation,
    OrchestratorConfig,
    OrchestratorRequest,
    OrchestratorResult,
    ParsedAction,
    ParseFailureAction,
    StepRecord,
    ToolCallAction,
    ToolLoopConfig,
)
from backend.app.orchestrator.parser import ActionProtocolParser
from backend.app.orchestrator.protocols import (
    ActionProtocolParserProtocol,
    ConversationManagerProtocol,
    LLMServiceProtocol,
    ObservationSanitizerProtocol,
    PromptBuilderProtocol,
    ToolLoopControllerProtocol,
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
    "LoopLimitExceededError",
    "LoopState",
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
    "StepRecord",
    "ToolCallAction",
    "ToolLoopConfig",
    "ToolLoopController",
    "ToolLoopControllerProtocol",
    "TurnCancelledError",
    "TurnTimeoutError",
    "compute_call_signature",
]
