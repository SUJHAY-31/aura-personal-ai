"""
Unit tests for AURA database connection, schema initialization, constraints, and FTS5 synchronization.
"""

from collections.abc import Generator
import sqlite3

import pytest

from backend.app.database.connection import get_db_connection, get_db_context
from backend.app.database.schema import init_db


@pytest.fixture
def in_memory_db() -> Generator[sqlite3.Connection, None, None]:
    """Provide an initialized in-memory SQLite database connection."""
    conn = get_db_connection(db_path=":memory:")
    init_db(conn=conn)
    yield conn
    conn.close()


def test_tables_created(in_memory_db: sqlite3.Connection) -> None:
    """Verify that all required tables and virtual tables are created."""
    cursor = in_memory_db.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow') ORDER BY name;"
    )
    table_names = {row["name"] for row in cursor.fetchall()}

    assert "conversation_sessions" in table_names
    assert "conversation_turns" in table_names
    assert "long_term_memories" in table_names
    assert "long_term_memories_fts" in table_names


def test_database_initialization_idempotency(in_memory_db: sqlite3.Connection) -> None:
    """Verify that calling init_db multiple times is safe and does not raise errors."""
    # Execute again on the already initialized database
    init_db(conn=in_memory_db)
    init_db(conn=in_memory_db)

    cursor = in_memory_db.execute(
        "SELECT COUNT(*) as count FROM sqlite_master WHERE name = 'conversation_sessions';"
    )
    assert cursor.fetchone()["count"] == 1


def test_pragmas_and_foreign_keys_enabled(in_memory_db: sqlite3.Connection) -> None:
    """Verify foreign keys and busy timeout pragmas are active."""
    fk_status = in_memory_db.execute("PRAGMA foreign_keys;").fetchone()[0]
    busy_timeout = in_memory_db.execute("PRAGMA busy_timeout;").fetchone()[0]

    assert fk_status == 1
    assert busy_timeout == 5000


def test_file_db_wal_mode(tmp_path: pytest.TempPathFactory) -> None:
    """Verify that file-based database connections enable WAL journal mode."""
    db_file = str(tmp_path / "test_wal.db")  # type: ignore[operator]
    conn = get_db_connection(db_path=db_file)
    try:
        init_db(conn=conn)
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()


def test_foreign_key_enforcement_and_cascade(in_memory_db: sqlite3.Connection) -> None:
    """Verify foreign key enforcement and cascading deletion of conversation turns."""
    # Inserting turn with non-existent session_id must fail
    with pytest.raises(sqlite3.IntegrityError):
        in_memory_db.execute(
            """
            INSERT INTO conversation_turns (id, session_id, role, content)
            VALUES ('t1', 'non_existent_session', 'user', 'Hello');
            """
        )

    # Insert valid session and turn
    in_memory_db.execute(
        "INSERT INTO conversation_sessions (id, title) VALUES ('s1', 'Session 1');"
    )
    in_memory_db.execute(
        """
        INSERT INTO conversation_turns (id, session_id, role, content)
        VALUES ('t1', 's1', 'user', 'Hello AURA');
        """
    )
    in_memory_db.commit()

    turns_count = in_memory_db.execute(
        "SELECT COUNT(*) FROM conversation_turns WHERE session_id = 's1';"
    ).fetchone()[0]
    assert turns_count == 1

    # Deleting session must cascade delete associated turns
    in_memory_db.execute("DELETE FROM conversation_sessions WHERE id = 's1';")
    in_memory_db.commit()

    remaining_turns = in_memory_db.execute(
        "SELECT COUNT(*) FROM conversation_turns WHERE session_id = 's1';"
    ).fetchone()[0]
    assert remaining_turns == 0


def test_long_term_memory_session_set_null(in_memory_db: sqlite3.Connection) -> None:
    """Verify long-term memory session_id is set to NULL on session deletion."""
    in_memory_db.execute(
        "INSERT INTO conversation_sessions (id, title) VALUES ('s1', 'Session 1');"
    )
    in_memory_db.execute(
        """
        INSERT INTO long_term_memories (id, category, fact, source, session_id)
        VALUES ('m1', 'project', 'Project name is AURA', 'explicit_user_instruction', 's1');
        """
    )
    in_memory_db.commit()

    in_memory_db.execute("DELETE FROM conversation_sessions WHERE id = 's1';")
    in_memory_db.commit()

    memory_row = in_memory_db.execute(
        "SELECT session_id FROM long_term_memories WHERE id = 'm1';"
    ).fetchone()
    assert memory_row["session_id"] is None


def test_conversation_turn_role_constraint(in_memory_db: sqlite3.Connection) -> None:
    """Verify CHECK constraint on conversation_turns.role."""
    in_memory_db.execute(
        "INSERT INTO conversation_sessions (id, title) VALUES ('s1', 'Session 1');"
    )

    # Valid roles
    for role in ("user", "assistant", "system"):
        in_memory_db.execute(
            f"INSERT INTO conversation_turns (id, session_id, role, content) VALUES ('t_{role}', 's1', '{role}', 'msg');"
        )
    in_memory_db.commit()

    # Invalid role must fail
    with pytest.raises(sqlite3.IntegrityError):
        in_memory_db.execute(
            "INSERT INTO conversation_turns (id, session_id, role, content) VALUES ('t_bad', 's1', 'invalid_role', 'msg');"
        )


def test_long_term_memory_constraints(in_memory_db: sqlite3.Connection) -> None:
    """Verify CHECK constraints on category and source for long_term_memories."""
    # Valid categories
    valid_categories = ("identity", "preference", "project", "constraint", "general")
    for cat in valid_categories:
        in_memory_db.execute(
            f"INSERT INTO long_term_memories (id, category, fact, source) VALUES ('m_{cat}', '{cat}', 'fact', 'manual_ui_entry');"
        )
    in_memory_db.commit()

    # Invalid category
    with pytest.raises(sqlite3.IntegrityError):
        in_memory_db.execute(
            "INSERT INTO long_term_memories (id, category, fact, source) VALUES ('m_bad_cat', 'unknown_cat', 'fact', 'manual_ui_entry');"
        )

    # Valid sources
    valid_sources = ("explicit_user_instruction", "extracted_from_dialog", "manual_ui_entry")
    for src in valid_sources:
        in_memory_db.execute(
            f"INSERT INTO long_term_memories (id, category, fact, source) VALUES ('m_{src}', 'general', 'fact', '{src}');"
        )
    in_memory_db.commit()

    # Invalid source
    with pytest.raises(sqlite3.IntegrityError):
        in_memory_db.execute(
            "INSERT INTO long_term_memories (id, category, fact, source) VALUES ('m_bad_src', 'general', 'fact', 'invalid_source');"
        )


def test_fts5_insert_update_delete_synchronization(in_memory_db: sqlite3.Connection) -> None:
    """Verify FTS5 virtual table synchronization on insert, update, and delete."""
    # 1. Insert memory
    in_memory_db.execute(
        """
        INSERT INTO long_term_memories (id, category, fact, source)
        VALUES ('m1', 'project', 'My primary project is AURA Personal AI Assistant', 'explicit_user_instruction');
        """
    )
    in_memory_db.commit()

    # Search for "AURA" via FTS5
    matches = in_memory_db.execute(
        "SELECT rowid, fact, category FROM long_term_memories_fts WHERE long_term_memories_fts MATCH 'AURA';"
    ).fetchall()
    assert len(matches) == 1
    assert "AURA Personal AI" in matches[0]["fact"]

    # 2. Update memory fact
    in_memory_db.execute(
        """
        UPDATE long_term_memories
        SET fact = 'My primary project is Quantum Framework'
        WHERE id = 'm1';
        """
    )
    in_memory_db.commit()

    # Old term "AURA" must no longer match
    old_matches = in_memory_db.execute(
        "SELECT * FROM long_term_memories_fts WHERE long_term_memories_fts MATCH 'AURA';"
    ).fetchall()
    assert len(old_matches) == 0

    # New term "Quantum" must match
    new_matches = in_memory_db.execute(
        "SELECT * FROM long_term_memories_fts WHERE long_term_memories_fts MATCH 'Quantum';"
    ).fetchall()
    assert len(new_matches) == 1

    # 3. Delete memory
    in_memory_db.execute("DELETE FROM long_term_memories WHERE id = 'm1';")
    in_memory_db.commit()

    deleted_matches = in_memory_db.execute(
        "SELECT * FROM long_term_memories_fts WHERE long_term_memories_fts MATCH 'Quantum';"
    ).fetchall()
    assert len(deleted_matches) == 0


def test_get_db_context_commit_and_rollback(tmp_path: pytest.TempPathFactory) -> None:
    """Verify get_db_context transactional commit on success and rollback on error."""
    db_file = str(tmp_path / "test_tx.db")  # type: ignore[operator]
    init_db(db_path=db_file)

    # Successful transaction
    with get_db_context(db_path=db_file) as conn:
        conn.execute("INSERT INTO conversation_sessions (id, title) VALUES ('s1', 'Success');")

    with get_db_context(db_path=db_file) as conn:
        count = conn.execute("SELECT COUNT(*) FROM conversation_sessions;").fetchone()[0]
        assert count == 1

    # Failed transaction (rollback)
    with pytest.raises(RuntimeError):
        with get_db_context(db_path=db_file) as conn:
            conn.execute("INSERT INTO conversation_sessions (id, title) VALUES ('s2', 'Fail');")
            raise RuntimeError("Forced error for rollback")

    with get_db_context(db_path=db_file) as conn:
        count = conn.execute("SELECT COUNT(*) FROM conversation_sessions;").fetchone()[0]
        assert count == 1
