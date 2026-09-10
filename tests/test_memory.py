"""
Unit tests for ConversationManager and database-backed conversation memory.
"""

from collections.abc import Generator
import sqlite3

import pytest

from backend.app.database.connection import get_db_connection
from backend.app.database.schema import init_db
from backend.app.memory.manager import ConversationManager
from backend.app.models.memory import ConversationSession, ConversationTurn


@pytest.fixture
def memory_manager(tmp_path: pytest.TempPathFactory) -> Generator[ConversationManager, None, None]:
    """Provide a ConversationManager backed by an initialized temporary SQLite file."""
    db_file = str(tmp_path / "test_memory.db")  # type: ignore[operator]
    conn = get_db_connection(db_path=db_file)
    init_db(conn=conn)
    conn.close()

    manager = ConversationManager(db_path=db_file)
    yield manager


def test_create_session_defaults(memory_manager: ConversationManager) -> None:
    """Verify session creation generates UUID and sets default title."""
    session = memory_manager.create_session()

    assert isinstance(session, ConversationSession)
    assert len(session.id) > 0
    assert session.title == "New Conversation"
    assert session.created_at is not None
    assert session.updated_at is not None


def test_create_session_custom_id_and_title(memory_manager: ConversationManager) -> None:
    """Verify session creation with custom ID and title."""
    session = memory_manager.create_session(session_id="custom-123", title="Research Notes")

    assert session.id == "custom-123"
    assert session.title == "Research Notes"


def test_get_session_existing_and_missing(memory_manager: ConversationManager) -> None:
    """Verify retrieving existing session returns model; missing returns None."""
    created = memory_manager.create_session(session_id="sess-abc", title="Test Session")

    fetched = memory_manager.get_session("sess-abc")
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.title == "Test Session"

    assert memory_manager.get_session("non-existent-id") is None
    assert memory_manager.get_session("") is None


def test_get_or_create_session(memory_manager: ConversationManager) -> None:
    """Verify get_or_create_session retrieves existing or initializes new."""
    # 1. No ID passed -> creates new
    s1 = memory_manager.get_or_create_session(default_title="Auto Session")
    assert s1.title == "Auto Session"

    # 2. Existing ID passed -> returns existing without creating new
    s2 = memory_manager.get_or_create_session(session_id=s1.id, default_title="Different Title")
    assert s2.id == s1.id
    assert s2.title == "Auto Session"

    # 3. New explicit ID passed -> creates new with that ID
    s3 = memory_manager.get_or_create_session(session_id="explicit-new-id", default_title="Explicit")
    assert s3.id == "explicit-new-id"
    assert s3.title == "Explicit"


def test_add_turns_and_role_validation(memory_manager: ConversationManager) -> None:
    """Verify adding valid turns and rejecting invalid roles or empty content."""
    session = memory_manager.create_session(session_id="s1")

    # Add user turn
    user_turn = memory_manager.add_turn(
        session_id=session.id,
        role="user",
        content="What is AURA?",
        token_count=5,
    )
    assert isinstance(user_turn, ConversationTurn)
    assert user_turn.role == "user"
    assert user_turn.content == "What is AURA?"
    assert user_turn.token_count == 5

    # Add assistant turn
    assistant_turn = memory_manager.add_turn(
        session_id=session.id,
        role="assistant",
        content="AURA is an AI personal assistant.",
    )
    assert assistant_turn.role == "assistant"
    assert assistant_turn.content == "AURA is an AI personal assistant."

    # Add system turn
    system_turn = memory_manager.add_turn(
        session_id=session.id,
        role="system",
        content="System instruction",
    )
    assert system_turn.role == "system"

    # Reject invalid role
    with pytest.raises(ValueError, match="Invalid role"):
        memory_manager.add_turn(session_id=session.id, role="invalid_role", content="msg")

    # Reject empty content
    with pytest.raises(ValueError, match="content must not be empty"):
        memory_manager.add_turn(session_id=session.id, role="user", content="   ")

    # Reject non-existent session
    with pytest.raises(ValueError, match="does not exist"):
        memory_manager.add_turn(session_id="missing-sess", role="user", content="Hello")


def test_get_recent_turns_ordering_and_limit(memory_manager: ConversationManager) -> None:
    """Verify recent turns are returned in chronological order (ASC) respecting limit."""
    session = memory_manager.create_session(session_id="s_turns")

    # Add 5 sequential turns
    for i in range(1, 6):
        role = "user" if i % 2 != 0 else "assistant"
        memory_manager.add_turn(session_id=session.id, role=role, content=f"Message {i}")

    # Fetch all with limit=10
    all_turns = memory_manager.get_recent_turns(session_id=session.id, limit=10)
    assert len(all_turns) == 5
    assert [t.content for t in all_turns] == [
        "Message 1",
        "Message 2",
        "Message 3",
        "Message 4",
        "Message 5",
    ]

    # Fetch latest 3 with limit=3 (must return Message 3, 4, 5 in ASC order)
    recent_3 = memory_manager.get_recent_turns(session_id=session.id, limit=3)
    assert len(recent_3) == 3
    assert [t.content for t in recent_3] == ["Message 3", "Message 4", "Message 5"]

    # Limit 0 or missing session
    assert memory_manager.get_recent_turns(session_id=session.id, limit=0) == []
    assert memory_manager.get_recent_turns(session_id="non-existent") == []


def test_delete_session_and_cascade(memory_manager: ConversationManager) -> None:
    """Verify deleting a session removes the session and cascades to its turns."""
    session = memory_manager.create_session(session_id="s_delete")
    memory_manager.add_turn(session_id=session.id, role="user", content="Turn 1")
    memory_manager.add_turn(session_id=session.id, role="assistant", content="Turn 2")

    assert len(memory_manager.get_recent_turns(session.id)) == 2

    # Delete session
    assert memory_manager.delete_session(session.id) is True
    assert memory_manager.get_session(session.id) is None
    assert memory_manager.get_recent_turns(session.id) == []

    # Deleting again returns False
    assert memory_manager.delete_session(session.id) is False


def test_list_sessions_ordering_and_pagination(memory_manager: ConversationManager) -> None:
    """Verify listing sessions ordered by updated_at with pagination support."""
    s1 = memory_manager.create_session(session_id="sess_1", title="Session 1")
    s2 = memory_manager.create_session(session_id="sess_2", title="Session 2")
    s3 = memory_manager.create_session(session_id="sess_3", title="Session 3")

    # Update s1 by adding a turn to make it the most recently updated
    memory_manager.add_turn(session_id=s1.id, role="user", content="New activity in session 1")

    sessions = memory_manager.list_sessions(limit=10)
    assert len(sessions) == 3
    assert sessions[0].id == "sess_1"

    # Test pagination
    paged = memory_manager.list_sessions(limit=2, offset=0)
    assert len(paged) == 2

    offset_paged = memory_manager.list_sessions(limit=2, offset=2)
    assert len(offset_paged) == 1


def test_add_exchange_success(memory_manager: ConversationManager) -> None:
    """Verify add_exchange atomically records both user and assistant turns."""
    session = memory_manager.create_session(session_id="s_exchange")

    user_turn, assistant_turn = memory_manager.add_exchange(
        session_id=session.id,
        user_content="What is Python?",
        assistant_content="Python is a programming language.",
        user_token_count=4,
        assistant_token_count=7,
    )

    assert isinstance(user_turn, ConversationTurn)
    assert user_turn.role == "user"
    assert user_turn.content == "What is Python?"
    assert user_turn.token_count == 4

    assert isinstance(assistant_turn, ConversationTurn)
    assert assistant_turn.role == "assistant"
    assert assistant_turn.content == "Python is a programming language."
    assert assistant_turn.token_count == 7

    # Verify both turns are retrieved in correct chronological order
    turns = memory_manager.get_recent_turns(session_id=session.id)
    assert len(turns) == 2
    assert turns[0].id == user_turn.id
    assert turns[1].id == assistant_turn.id


def test_add_exchange_validation(memory_manager: ConversationManager) -> None:
    """Verify add_exchange validates inputs and missing session."""
    session = memory_manager.create_session(session_id="s_val")

    with pytest.raises(ValueError, match="user_content must not be empty"):
        memory_manager.add_exchange(session.id, user_content="   ", assistant_content="Reply")

    with pytest.raises(ValueError, match="assistant_content must not be empty"):
        memory_manager.add_exchange(session.id, user_content="Hello", assistant_content="   ")

    with pytest.raises(ValueError, match="does not exist"):
        memory_manager.add_exchange("non-existent-session", user_content="Hello", assistant_content="Hi")


def test_add_exchange_rollback_on_failure(memory_manager: ConversationManager) -> None:
    """Verify add_exchange rolls back the user turn if assistant insertion fails."""
    session = memory_manager.create_session(session_id="s_rollback")

    # Force a primary key collision on assistant turn ID with an existing turn
    user_turn, _ = memory_manager.add_exchange(
        session.id,
        user_content="Turn 1",
        assistant_content="Turn 2",
        assistant_turn_id="colliding_id",
    )

    # Attempt second exchange reusing the same colliding assistant_turn_id
    with pytest.raises(sqlite3.IntegrityError):
        memory_manager.add_exchange(
            session.id,
            user_content="Turn 3 (should rollback)",
            assistant_content="Turn 4",
            assistant_turn_id="colliding_id",
        )

    # Verify only the first exchange (2 turns) exists and Turn 3 was rolled back
    turns = memory_manager.get_recent_turns(session.id)
    assert len(turns) == 2
    assert [t.content for t in turns] == ["Turn 1", "Turn 2"]

