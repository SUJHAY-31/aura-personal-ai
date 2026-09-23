"""
Unit tests for the AURA Orchestrator Core.

Verifies end-to-end orchestration logic, dependency injection,
exception propagation, and boundary isolation without FastAPI or live databases.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.app.ai.llm import (
    LLMConnectionError,
    LLMInferenceError,
    LLMServiceError,
    LLMTimeoutError,
)
from backend.app.models.memory import ConversationSession, ConversationTurn
from backend.app.orchestrator import (
    AuraOrchestrator,
    ConversationManagerProtocol,
    LLMServiceProtocol,
    OrchestrationFailureError,
    OrchestratorConfig,
    OrchestratorRequest,
    OrchestratorResult,
    PersistenceError,
    PromptBuilderProtocol,
    SessionInitializationError,
)

NOW = datetime.now(timezone.utc).isoformat()


@pytest.fixture
def fake_session() -> ConversationSession:
    """Provide a stub ConversationSession for tests."""
    return ConversationSession(
        id="session-test-uuid",
        title="Test Session",
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.fixture
def mock_memory(fake_session: ConversationSession) -> MagicMock:
    """Provide a mock ConversationManager conforming to ConversationManagerProtocol."""
    mock = MagicMock(spec=ConversationManagerProtocol)
    mock.get_or_create_session.return_value = fake_session
    mock.get_recent_turns.return_value = []
    mock.add_exchange.return_value = (
        ConversationTurn(id="turn-1", session_id=fake_session.id, role="user", content="Hello", created_at=NOW),
        ConversationTurn(id="turn-2", session_id=fake_session.id, role="assistant", content="Hi there", created_at=NOW),
    )
    return mock


@pytest.fixture
def mock_llm() -> MagicMock:
    """Provide a mock LLMService conforming to LLMServiceProtocol."""
    mock = MagicMock(spec=LLMServiceProtocol)
    mock.generate.return_value = "Hello from mock AURA"
    return mock


@pytest.fixture
def mock_prompt_builder() -> MagicMock:
    """Provide a mock PromptBuilder conforming to PromptBuilderProtocol."""
    mock = MagicMock(spec=PromptBuilderProtocol)
    mock.build_prompt.return_value = "System: Test\n\nUser: Hello\n\nAssistant:"
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_orchestrator_successful_new_session(
    mock_memory: MagicMock,
    mock_llm: MagicMock,
    fake_session: ConversationSession,
) -> None:
    """1. Successful new session initializes memory, invokes LLM, and persists turn."""
    orchestrator = AuraOrchestrator(
        memory_manager=mock_memory,
        llm_service=mock_llm,
    )

    request = OrchestratorRequest(message="Hello AURA")
    result = orchestrator.process_turn(request)

    assert isinstance(result, OrchestratorResult)
    assert result.response == "Hello from mock AURA"
    assert result.session_id == fake_session.id
    assert result.turns_count == 2
    mock_memory.get_or_create_session.assert_called_once_with(session_id=None)
    mock_memory.get_recent_turns.assert_called_once_with(
        session_id=fake_session.id,
        limit=10,
    )
    mock_llm.generate.assert_called_once()
    mock_memory.add_exchange.assert_called_once_with(
        session_id=fake_session.id,
        user_content="Hello AURA",
        assistant_content="Hello from mock AURA",
    )


def test_orchestrator_existing_session_continuation(
    mock_memory: MagicMock,
    mock_llm: MagicMock,
) -> None:
    """2. Existing session continuation passes provided session_id to memory manager."""
    custom_session = ConversationSession(
        id="custom-session-123",
        title="Existing",
        created_at=NOW,
        updated_at=NOW,
    )
    mock_memory.get_or_create_session.return_value = custom_session

    orchestrator = AuraOrchestrator(
        memory_manager=mock_memory,
        llm_service=mock_llm,
    )

    request = OrchestratorRequest(message="Continue discussion", session_id="custom-session-123")
    result = orchestrator.process_turn(request)

    assert result.session_id == "custom-session-123"
    mock_memory.get_or_create_session.assert_called_once_with(session_id="custom-session-123")


def test_orchestrator_history_passed_to_prompt_builder(
    mock_memory: MagicMock,
    mock_llm: MagicMock,
    mock_prompt_builder: MagicMock,
    fake_session: ConversationSession,
) -> None:
    """3. History retrieved from memory is passed to PromptBuilder."""
    prior_turns = [
        ConversationTurn(id="t1", session_id=fake_session.id, role="user", content="My name is Alex", created_at=NOW),
        ConversationTurn(id="t2", session_id=fake_session.id, role="assistant", content="Nice to meet you Alex", created_at=NOW),
    ]
    mock_memory.get_recent_turns.return_value = prior_turns

    orchestrator = AuraOrchestrator(
        memory_manager=mock_memory,
        llm_service=mock_llm,
        prompt_builder=mock_prompt_builder,
    )

    request = OrchestratorRequest(message="What is my name?")
    orchestrator.process_turn(request)

    mock_prompt_builder.build_prompt.assert_called_once_with(
        current_message="What is my name?",
        history=prior_turns,
        system_prompt=OrchestratorConfig().system_prompt,
    )


def test_orchestrator_prompt_passed_to_llm_service(
    mock_memory: MagicMock,
    mock_llm: MagicMock,
    mock_prompt_builder: MagicMock,
) -> None:
    """4. Exact prompt returned by PromptBuilder is passed to LLMService.generate."""
    expected_prompt = "Custom formatted prompt text for inference"
    mock_prompt_builder.build_prompt.return_value = expected_prompt

    orchestrator = AuraOrchestrator(
        memory_manager=mock_memory,
        llm_service=mock_llm,
        prompt_builder=mock_prompt_builder,
    )

    request = OrchestratorRequest(message="Explain relativity")
    orchestrator.process_turn(request)

    mock_llm.generate.assert_called_once_with(expected_prompt)


def test_orchestrator_exchange_persisted_after_successful_llm_response(
    mock_memory: MagicMock,
    mock_llm: MagicMock,
    fake_session: ConversationSession,
) -> None:
    """5. add_exchange is called only after successful LLM generation with correct values."""
    mock_llm.generate.return_value = "Generated assistant reply"

    orchestrator = AuraOrchestrator(
        memory_manager=mock_memory,
        llm_service=mock_llm,
    )

    request = OrchestratorRequest(message="   Tell me a joke   ")
    result = orchestrator.process_turn(request)

    assert result.response == "Generated assistant reply"
    mock_memory.add_exchange.assert_called_once_with(
        session_id=fake_session.id,
        user_content="Tell me a joke",
        assistant_content="Generated assistant reply",
    )


@pytest.mark.parametrize(
    "error_instance",
    [
        LLMConnectionError("Connection refused by Ollama"),
        LLMTimeoutError("Inference timed out after 30s"),
        LLMInferenceError("Ollama returned 500 error"),
        LLMServiceError("Generic LLM failure"),
    ],
)
def test_orchestrator_llm_service_error_propagates_unchanged(
    mock_memory: MagicMock,
    mock_llm: MagicMock,
    error_instance: LLMServiceError,
) -> None:
    """6. LLMServiceError and its subclasses propagate completely unchanged without rewrite."""
    mock_llm.generate.side_effect = error_instance

    orchestrator = AuraOrchestrator(
        memory_manager=mock_memory,
        llm_service=mock_llm,
    )

    request = OrchestratorRequest(message="Hello")

    with pytest.raises(type(error_instance)) as exc_info:
        orchestrator.process_turn(request)

    assert exc_info.value is error_instance
    # Ensure nothing was persisted when LLM generation failed
    mock_memory.add_exchange.assert_not_called()


def test_orchestrator_session_failure_becomes_session_initialization_error(
    mock_memory: MagicMock,
    mock_llm: MagicMock,
) -> None:
    """7. Session resolution failure wraps exception into SessionInitializationError."""
    mock_memory.get_or_create_session.side_effect = RuntimeError("Database locked")

    orchestrator = AuraOrchestrator(
        memory_manager=mock_memory,
        llm_service=mock_llm,
    )

    request = OrchestratorRequest(message="Hello")

    with pytest.raises(SessionInitializationError) as exc_info:
        orchestrator.process_turn(request)

    assert "Database locked" in str(exc_info.value)
    mock_llm.generate.assert_not_called()
    mock_memory.add_exchange.assert_not_called()


def test_orchestrator_history_retrieval_failure_becomes_session_initialization_error(
    mock_memory: MagicMock,
    mock_llm: MagicMock,
) -> None:
    """7b. History retrieval failure also wraps into SessionInitializationError."""
    mock_memory.get_recent_turns.side_effect = RuntimeError("Failed to read turns")

    orchestrator = AuraOrchestrator(
        memory_manager=mock_memory,
        llm_service=mock_llm,
    )

    request = OrchestratorRequest(message="Hello")

    with pytest.raises(SessionInitializationError) as exc_info:
        orchestrator.process_turn(request)

    assert "Failed to read turns" in str(exc_info.value)
    mock_llm.generate.assert_not_called()


def test_orchestrator_persistence_failure_becomes_persistence_error(
    mock_memory: MagicMock,
    mock_llm: MagicMock,
) -> None:
    """8. Exchange persistence failure wraps exception into PersistenceError."""
    mock_memory.add_exchange.side_effect = RuntimeError("Disk full writing turn")

    orchestrator = AuraOrchestrator(
        memory_manager=mock_memory,
        llm_service=mock_llm,
    )

    request = OrchestratorRequest(message="Hello")

    with pytest.raises(PersistenceError) as exc_info:
        orchestrator.process_turn(request)

    assert "Disk full writing turn" in str(exc_info.value)
    mock_llm.generate.assert_called_once()


@pytest.mark.parametrize("blank_message", ["", "   ", "\t\n  "])
def test_orchestrator_blank_message_rejected(
    mock_memory: MagicMock,
    mock_llm: MagicMock,
    blank_message: str,
) -> None:
    """9. Empty or whitespace-only messages are rejected with ValueError before memory/LLM calls."""
    orchestrator = AuraOrchestrator(
        memory_manager=mock_memory,
        llm_service=mock_llm,
    )

    request = OrchestratorRequest(message=blank_message)

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        orchestrator.process_turn(request)

    mock_memory.get_or_create_session.assert_not_called()
    mock_llm.generate.assert_not_called()
    mock_memory.add_exchange.assert_not_called()


def test_orchestrator_turns_count_calculation(
    mock_memory: MagicMock,
    mock_llm: MagicMock,
    fake_session: ConversationSession,
) -> None:
    """10. turns_count accurately computes previous_history_count + 2."""
    prior_turns = [
        ConversationTurn(id="t1", session_id=fake_session.id, role="user", content="Turn 1", created_at=NOW),
        ConversationTurn(id="t2", session_id=fake_session.id, role="assistant", content="Turn 2", created_at=NOW),
        ConversationTurn(id="t3", session_id=fake_session.id, role="user", content="Turn 3", created_at=NOW),
        ConversationTurn(id="t4", session_id=fake_session.id, role="assistant", content="Turn 4", created_at=NOW),
    ]
    mock_memory.get_recent_turns.return_value = prior_turns

    orchestrator = AuraOrchestrator(
        memory_manager=mock_memory,
        llm_service=mock_llm,
    )

    request = OrchestratorRequest(message="Turn 5")
    result = orchestrator.process_turn(request)

    assert result.turns_count == 6  # 4 previous turns + 2 (current user + assistant)


def test_orchestrator_dependencies_are_injectable_with_custom_config(
    fake_session: ConversationSession,
) -> None:
    """11. Custom fake/mock dependencies and config work seamlessly via constructor injection."""
    calls: list[str] = []

    class FakeMemory:
        def get_or_create_session(self, session_id: str | None = None) -> Any:
            calls.append("get_or_create_session")
            return fake_session

        def get_recent_turns(self, session_id: str, limit: int = 10) -> list[Any]:
            calls.append(f"get_recent_turns:{limit}")
            return []

        def add_exchange(self, session_id: str, user_content: str, assistant_content: str) -> None:
            calls.append("add_exchange")

    class FakePromptBuilder:
        def build_prompt(
            self,
            current_message: str,
            *,
            history: Any = None,
            system_prompt: str | None = None,
        ) -> str:
            calls.append(f"build_prompt:{system_prompt}")
            return f"Prompt:{current_message}"

    class FakeLLM:
        def generate(self, prompt: str) -> str:
            calls.append("generate")
            return "Fake response"

    config = OrchestratorConfig(max_history_turns=5, system_prompt="Custom System Prompt")
    orchestrator = AuraOrchestrator(
        memory_manager=FakeMemory(),
        llm_service=FakeLLM(),
        prompt_builder=FakePromptBuilder(),
        config=config,
    )

    request = OrchestratorRequest(message="Test message", metadata={"trace_id": "tr-123"})
    result = orchestrator.process_turn(request)

    assert result.response == "Fake response"
    assert result.metadata == {"trace_id": "tr-123"}
    assert calls == [
        "get_or_create_session",
        "get_recent_turns:5",
        "build_prompt:Custom System Prompt",
        "generate",
        "add_exchange",
    ]


def test_orchestrator_prompt_builder_failure_becomes_orchestration_failure_error(
    mock_memory: MagicMock,
    mock_llm: MagicMock,
    mock_prompt_builder: MagicMock,
) -> None:
    """12. Prompt builder unexpected error wraps into OrchestrationFailureError."""
    mock_prompt_builder.build_prompt.side_effect = TypeError("Unexpected prompt construction error")

    orchestrator = AuraOrchestrator(
        memory_manager=mock_memory,
        llm_service=mock_llm,
        prompt_builder=mock_prompt_builder,
    )

    request = OrchestratorRequest(message="Hello")

    with pytest.raises(OrchestrationFailureError) as exc_info:
        orchestrator.process_turn(request)

    assert "Failed to build prompt" in str(exc_info.value)
    mock_llm.generate.assert_not_called()
