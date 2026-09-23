"""
Automated tests for AURA chat endpoint (POST /chat) with AuraOrchestrator integration.

Tests route adaptation, session handling, response mapping, input validation,
LLM error mapping, orchestrator error sanitization, resource lifecycle, and orchestrator execution.
"""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from backend.app.ai.llm import (
    LLMConnectionError,
    LLMInferenceError,
    LLMService,
    LLMServiceError,
    LLMTimeoutError,
)
from backend.app.api.chat import (
    get_llm_service,
    get_memory_manager,
    get_orchestrator,
)
from backend.app.database.connection import get_db_connection
from backend.app.database.schema import init_db
from backend.app.main import app
from backend.app.memory.manager import ConversationManager
from backend.app.orchestrator import (
    AuraOrchestrator,
    OrchestrationFailureError,
    OrchestratorRequest,
    OrchestratorResult,
    PersistenceError,
    SessionInitializationError,
)


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    """Provide a mock AuraOrchestrator."""
    return MagicMock(spec=AuraOrchestrator)


@pytest.fixture
def client(mock_orchestrator: MagicMock) -> Generator[TestClient, None, None]:
    """Provide a TestClient with AuraOrchestrator dependency override."""
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_chat_successful_new_conversation(
    client: TestClient,
    mock_orchestrator: MagicMock,
) -> None:
    """POST /chat without session_id converts request, calls process_turn, and returns ChatResponse."""
    mock_orchestrator.process_turn.return_value = OrchestratorResult(
        response="Hello from AURA!",
        session_id="session-uuid-1",
        turns_count=2,
    )

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Hello from AURA!"
    assert data["session_id"] == "session-uuid-1"

    mock_orchestrator.process_turn.assert_called_once()
    req = mock_orchestrator.process_turn.call_args[0][0]
    assert isinstance(req, OrchestratorRequest)
    assert req.message == "Hello AURA"
    assert req.session_id is None


def test_chat_continuing_existing_session(
    client: TestClient,
    mock_orchestrator: MagicMock,
) -> None:
    """POST /chat with existing session_id passes session_id to OrchestratorRequest."""
    mock_orchestrator.process_turn.return_value = OrchestratorResult(
        response="Your name is Alex.",
        session_id="session-uuid-existing",
        turns_count=4,
    )

    response = client.post(
        "/chat",
        json={"message": "What is my name?", "session_id": "session-uuid-existing"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Your name is Alex."
    assert data["session_id"] == "session-uuid-existing"

    mock_orchestrator.process_turn.assert_called_once()
    req = mock_orchestrator.process_turn.call_args[0][0]
    assert isinstance(req, OrchestratorRequest)
    assert req.message == "What is my name?"
    assert req.session_id == "session-uuid-existing"


def test_chat_response_and_session_id_mapping(
    client: TestClient,
    mock_orchestrator: MagicMock,
) -> None:
    """POST /chat response strictly contains only response and session_id fields."""
    mock_orchestrator.process_turn.return_value = OrchestratorResult(
        response="Exact test response",
        session_id="strict-session-id-999",
        turns_count=10,
        metadata={"internal_note": "do_not_leak"},
    )

    response = client.post("/chat", json={"message": "Test mapping"})

    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"response", "session_id"}
    assert data["response"] == "Exact test response"
    assert data["session_id"] == "strict-session-id-999"
    assert "turns_count" not in data
    assert "metadata" not in data


@pytest.mark.parametrize("blank_message", ["", "   ", "\t\n  ", " \r\n "])
def test_chat_blank_input_validation(
    client: TestClient,
    mock_orchestrator: MagicMock,
    blank_message: str,
) -> None:
    """POST /chat with empty or whitespace-only message returns 422 without calling orchestrator."""
    response = client.post("/chat", json={"message": blank_message})
    assert response.status_code == 422
    mock_orchestrator.process_turn.assert_not_called()


def test_chat_llm_timeout_maps_to_504(
    client: TestClient,
    mock_orchestrator: MagicMock,
) -> None:
    """POST /chat maps propagated LLMTimeoutError to HTTP 504."""
    mock_orchestrator.process_turn.side_effect = LLMTimeoutError("Ollama inference timed out")

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 504
    assert "Ollama inference timed out" in response.json()["detail"]


def test_chat_llm_connection_maps_to_503(
    client: TestClient,
    mock_orchestrator: MagicMock,
) -> None:
    """POST /chat maps propagated LLMConnectionError to HTTP 503."""
    mock_orchestrator.process_turn.side_effect = LLMConnectionError("Could not reach Ollama")

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 503
    assert "Could not reach Ollama" in response.json()["detail"]


def test_chat_llm_inference_maps_to_502(
    client: TestClient,
    mock_orchestrator: MagicMock,
) -> None:
    """POST /chat maps propagated LLMInferenceError to HTTP 502."""
    mock_orchestrator.process_turn.side_effect = LLMInferenceError("Ollama returned HTTP 500")

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 502
    assert "Ollama returned HTTP 500" in response.json()["detail"]


def test_chat_generic_llm_error_maps_to_500(
    client: TestClient,
    mock_orchestrator: MagicMock,
) -> None:
    """POST /chat maps propagated generic LLMServiceError to HTTP 500."""
    mock_orchestrator.process_turn.side_effect = LLMServiceError("Unexpected LLM failure")

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 500
    assert "Unexpected LLM failure" in response.json()["detail"]


def test_chat_session_initialization_error_safe_500(
    client: TestClient,
    mock_orchestrator: MagicMock,
) -> None:
    """POST /chat maps SessionInitializationError to HTTP 500 with sanitized message."""
    mock_orchestrator.process_turn.side_effect = SessionInitializationError(
        "Sensitive internal DB path /var/aura/db.sqlite: OperationalError"
    )

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Failed to initialize conversation."
    assert "Sensitive" not in detail
    assert "sqlite" not in detail
    assert "OperationalError" not in detail


def test_chat_persistence_error_safe_500(
    client: TestClient,
    mock_orchestrator: MagicMock,
) -> None:
    """POST /chat maps PersistenceError to HTTP 500 with sanitized message."""
    mock_orchestrator.process_turn.side_effect = PersistenceError(
        "Sensitive disk error on table conversation_turns: disk full"
    )

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Failed to save conversation."
    assert "Sensitive" not in detail
    assert "conversation_turns" not in detail
    assert "disk full" not in detail


def test_chat_orchestration_failure_safe_500(
    client: TestClient,
    mock_orchestrator: MagicMock,
) -> None:
    """POST /chat maps OrchestrationFailureError to HTTP 500 with sanitized message."""
    mock_orchestrator.process_turn.side_effect = OrchestrationFailureError(
        "Sensitive prompt template failure in module backend/app/memory/prompt.py"
    )

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Failed to process conversation."
    assert "Sensitive" not in detail
    assert "backend/app/memory/prompt.py" not in detail


def test_chat_integration_calls_aura_orchestrator_process_turn(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Integration test verifying route resolves AuraOrchestrator and executes process_turn end-to-end."""
    db_file = str(tmp_path / "integration_chat.db")  # type: ignore[operator]
    conn = get_db_connection(db_path=db_file)
    init_db(conn=conn)
    conn.close()

    real_memory = ConversationManager(db_path=db_file)
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.generate.return_value = "Integration test response"

    app.dependency_overrides.clear()
    app.dependency_overrides[get_memory_manager] = lambda: real_memory
    app.dependency_overrides[get_llm_service] = lambda: mock_llm

    original_process_turn = AuraOrchestrator.process_turn
    intercepted_requests: list[OrchestratorRequest] = []

    def spy_process_turn(self: AuraOrchestrator, request: OrchestratorRequest) -> OrchestratorResult:
        intercepted_requests.append(request)
        return original_process_turn(self, request)

    with patch.object(AuraOrchestrator, "process_turn", spy_process_turn):
        with TestClient(app) as test_client:
            res = test_client.post("/chat", json={"message": "Integration test message"})
            assert res.status_code == 200
            data = res.json()
            assert data["response"] == "Integration test response"
            session_id = data["session_id"]
            assert session_id

            assert len(intercepted_requests) == 1
            call_request = intercepted_requests[0]
            assert isinstance(call_request, OrchestratorRequest)
            assert call_request.message == "Integration test message"

            turns = real_memory.get_recent_turns(session_id=session_id)
            assert len(turns) == 2
            assert turns[0].content == "Integration test message"
            assert turns[1].content == "Integration test response"

    app.dependency_overrides.clear()


def test_chat_llm_service_lifecycle_closed_after_request(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verifies that the LLMService HTTP client is closed after request execution."""
    db_file = str(tmp_path / "lifecycle_chat.db")  # type: ignore[operator]
    conn = get_db_connection(db_path=db_file)
    init_db(conn=conn)
    conn.close()

    real_memory = ConversationManager(db_path=db_file)
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.generate.return_value = "Lifecycle response"

    def llm_generator() -> Generator[LLMService, None, None]:
        try:
            yield mock_llm
        finally:
            mock_llm.close()

    app.dependency_overrides.clear()
    app.dependency_overrides[get_memory_manager] = lambda: real_memory
    app.dependency_overrides[get_llm_service] = llm_generator

    with TestClient(app) as test_client:
        res = test_client.post("/chat", json={"message": "Lifecycle check"})
        assert res.status_code == 200

    mock_llm.close.assert_called_once()
    app.dependency_overrides.clear()

