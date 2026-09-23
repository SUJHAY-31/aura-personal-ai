"""
Core orchestrator implementation for AURA.

Coordinates session resolution, history retrieval, prompt assembly,
LLM inference, and turn persistence across clean module boundaries.
"""

from __future__ import annotations

from backend.app.memory.prompt import PromptBuilder
from backend.app.orchestrator.errors import (
    OrchestrationFailureError,
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


class AuraOrchestrator:
    """Central coordinator for processing conversational turns."""

    def __init__(
        self,
        *,
        memory_manager: ConversationManagerProtocol,
        llm_service: LLMServiceProtocol,
        prompt_builder: PromptBuilderProtocol | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._memory = memory_manager
        self._llm = llm_service
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._config = config or OrchestratorConfig()

    def process_turn(
        self,
        request: OrchestratorRequest,
    ) -> OrchestratorResult:
        """
        Process a single conversational turn through the orchestration pipeline.

        Flow:
        1. Validate user message is non-blank.
        2. Resolve or create session and load recent history.
        3. Assemble deterministic prompt using PromptBuilder.
        4. Invoke LLMService.generate(prompt).
        5. Persist user and assistant exchange in memory.
        6. Return OrchestratorResult.

        Raises:
            ValueError: If the input message is empty or whitespace-only.
            SessionInitializationError: If resolving session or history fails.
            LLMServiceError: If the LLM call encounters a connection, timeout, or inference error.
            PersistenceError: If saving the exchange to memory fails.
            OrchestrationFailureError: On any other unexpected orchestration failure.
        """
        stripped_message = request.message.strip() if request.message else ""
        if not stripped_message:
            raise ValueError("request.message must not be empty or whitespace-only")

        try:
            session = self._memory.get_or_create_session(session_id=request.session_id)
            history = self._memory.get_recent_turns(
                session_id=session.id,
                limit=self._config.max_history_turns,
            )
        except Exception as exc:
            raise SessionInitializationError(
                f"Failed to initialize or resolve conversation session: {exc}"
            ) from exc

        try:
            prompt = self._prompt_builder.build_prompt(
                current_message=stripped_message,
                history=history,
                system_prompt=self._config.system_prompt,
            )
        except Exception as exc:
            raise OrchestrationFailureError(f"Failed to build prompt: {exc}") from exc

        # LLMServiceError (and subclasses) propagates directly without being caught or rewritten
        response_text = self._llm.generate(prompt)

        try:
            self._memory.add_exchange(
                session_id=session.id,
                user_content=stripped_message,
                assistant_content=response_text,
            )
        except Exception as exc:
            raise PersistenceError(
                f"Failed to persist conversation exchange: {exc}"
            ) from exc

        turns_count = len(history) + 2

        return OrchestratorResult(
            response=response_text,
            session_id=session.id,
            turns_count=turns_count,
            metadata=dict(request.metadata),
        )
