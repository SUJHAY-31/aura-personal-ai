"""
Automated tests for AURA chat endpoint (POST /chat) with conversation memory.

Tests validation, session continuity, turn persistence, history injection, and error mapping.
"""

from collections.abc import Generator
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from backend.app.ai.llm import (
    LLMConnectionError,
    LLMInferenceError,
    LLMService,
    LLMServiceError,
    LLMTimeoutError,
)
from backend.app.api.chat import get_llm_service, get_memory_manager
from backend.app.database.connection import get_db_connection
from backend.app.database.schema import init_db
from backend.app.main import app
from backend.app.memory.manager import ConversationManager


@pytest.fixture
def memory_manager(tmp_path: pytest.TempPathFactory) -> ConversationManager:
    """Provide an isolated ConversationManager backed by a temporary SQLite file."""
    db_file = str(tmp_path / "test_chat_memory.db")  # type: ignore[operator]
    conn = get_db_connection(db_path=db_file)
    init_db(conn=conn)
    conn.close()
    return ConversationManager(db_path=db_file)


@pytest.fixture
def client(memory_manager: ConversationManager) -> Generator[TestClient, None, None]:
    """Provide a TestClient with memory and LLM dependency overrides."""
    app.dependency_overrides[get_memory_manager] = lambda: memory_manager
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_valid_chat_creates_session_and_persists_turns(
    client: TestClient,
    memory_manager: ConversationManager,
) -> None:
    """POST /chat without session_id creates a new session, generates reply, and saves turns."""
    mock_service = MagicMock(spec=LLMService)
    mock_service.generate.return_value = "Hello from AURA"
    app.dependency_overrides[get_llm_service] = lambda: mock_service

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Hello from AURA"
    assert "session_id" in data
    session_id = data["session_id"]
    assert len(session_id) > 0

    # Verify turns persisted in memory
    turns = memory_manager.get_recent_turns(session_id=session_id)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].content == "Hello AURA"
    assert turns[1].role == "assistant"
    assert turns[1].content == "Hello from AURA"


def test_chat_continues_existing_session_with_history(
    client: TestClient,
    memory_manager: ConversationManager,
) -> None:
    """POST /chat with existing session_id includes previous turns in prompt and appends new turns."""
    mock_service = MagicMock(spec=LLMService)
    mock_service.generate.side_effect = [
        "I can help you with programming and research.",
        "Your name is Alex.",
    ]
    app.dependency_overrides[get_llm_service] = lambda: mock_service

    # Turn 1: Introduce name
    res1 = client.post("/chat", json={"message": "My name is Alex."})
    assert res1.status_code == 200
    session_id = res1.json()["session_id"]

    # Turn 2: Ask question in same session
    res2 = client.post("/chat", json={"message": "What is my name?", "session_id": session_id})
    assert res2.status_code == 200
    assert res2.json()["session_id"] == session_id
    assert res2.json()["response"] == "Your name is Alex."

    # Verify the prompt passed to LLM for Turn 2 contains Turn 1 history
    assert mock_service.generate.call_count == 2
    second_prompt_arg = mock_service.generate.call_args_list[1][0][0]
    assert "User: My name is Alex." in second_prompt_arg
    assert "Assistant: I can help you with programming and research." in second_prompt_arg
    assert "User: What is my name?" in second_prompt_arg

    # Verify all 4 turns persisted in database
    turns = memory_manager.get_recent_turns(session_id=session_id)
    assert len(turns) == 4
    assert [t.content for t in turns] == [
        "My name is Alex.",
        "I can help you with programming and research.",
        "What is my name?",
        "Your name is Alex.",
    ]


def test_new_session_does_not_reuse_other_session_history(
    client: TestClient,
    memory_manager: ConversationManager,
) -> None:
    """A new session does not leak or reuse history from an earlier distinct session."""
    mock_service = MagicMock(spec=LLMService)
    mock_service.generate.return_value = "Acknowledged."
    app.dependency_overrides[get_llm_service] = lambda: mock_service

    # Session A
    res_a = client.post("/chat", json={"message": "Secret code is 12345."})
    assert res_a.status_code == 200
    session_a_id = res_a.json()["session_id"]

    # Session B (new session without session_id)
    res_b = client.post("/chat", json={"message": "What is the secret code?"})
    assert res_b.status_code == 200
    session_b_id = res_b.json()["session_id"]
    assert session_b_id != session_a_id

    # Check that Session B's prompt has NO history from Session A
    prompt_b = mock_service.generate.call_args_list[1][0][0]
    assert "12345" not in prompt_b


def test_whitespace_only_message_rejected(client: TestClient) -> None:
    """POST /chat with whitespace-only message returns 422 Unprocessable Entity."""
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 422


def test_chat_llm_connection_error(client: TestClient) -> None:
    """POST /chat maps LLMConnectionError to HTTP 503."""
    mock_service = MagicMock(spec=LLMService)
    mock_service.generate.side_effect = LLMConnectionError("Could not reach Ollama")
    app.dependency_overrides[get_llm_service] = lambda: mock_service

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 503
    assert "Could not reach Ollama" in response.json()["detail"]


def test_chat_llm_timeout_error(client: TestClient) -> None:
    """POST /chat maps LLMTimeoutError to HTTP 504."""
    mock_service = MagicMock(spec=LLMService)
    mock_service.generate.side_effect = LLMTimeoutError("Ollama inference timed out")
    app.dependency_overrides[get_llm_service] = lambda: mock_service

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 504
    assert "Ollama inference timed out" in response.json()["detail"]


def test_chat_llm_inference_error(client: TestClient) -> None:
    """POST /chat maps LLMInferenceError to HTTP 502."""
    mock_service = MagicMock(spec=LLMService)
    mock_service.generate.side_effect = LLMInferenceError("Ollama returned HTTP 500")
    app.dependency_overrides[get_llm_service] = lambda: mock_service

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 502
    assert "Ollama returned HTTP 500" in response.json()["detail"]


def test_chat_generic_llm_service_error(client: TestClient) -> None:
    """POST /chat maps generic LLMServiceError to HTTP 500."""
    mock_service = MagicMock(spec=LLMService)
    mock_service.generate.side_effect = LLMServiceError("Unexpected LLM failure")
    app.dependency_overrides[get_llm_service] = lambda: mock_service

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 500
    assert "Unexpected LLM failure" in response.json()["detail"]


def test_chat_memory_init_failure_safe_error(
    client: TestClient,
    memory_manager: ConversationManager,
) -> None:
    """POST /chat maps memory initialization errors to HTTP 500 with safe generic detail."""
    mock_memory = MagicMock(spec=ConversationManager)
    mock_memory.get_or_create_session.side_effect = RuntimeError("sqlite3.OperationalError: raw db error at /path/aura.db")
    app.dependency_overrides[get_memory_manager] = lambda: mock_memory

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to initialize conversation."
    # Ensure no internal path or raw error string is leaked
    assert "sqlite3" not in response.json()["detail"]
    assert "/path" not in response.json()["detail"]


def test_chat_memory_save_failure_safe_error(
    client: TestClient,
    memory_manager: ConversationManager,
) -> None:
    """POST /chat maps exchange persistence errors to HTTP 500 with safe generic detail."""
    mock_service = MagicMock(spec=LLMService)
    mock_service.generate.return_value = "Hello"
    app.dependency_overrides[get_llm_service] = lambda: mock_service

    mock_memory = MagicMock(spec=ConversationManager)
    mock_memory.get_or_create_session.return_value = MagicMock(id="s1")
    mock_memory.get_recent_turns.return_value = []
    mock_memory.add_exchange.side_effect = RuntimeError("disk write failure on table conversation_turns")
    app.dependency_overrides[get_memory_manager] = lambda: mock_memory

    response = client.post("/chat", json={"message": "Hello AURA"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to save conversation."
    # Ensure no table name or internal error is leaked
    assert "conversation_turns" not in response.json()["detail"]
    assert "disk" not in response.json()["detail"]

