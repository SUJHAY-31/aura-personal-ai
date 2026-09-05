"""
Database schema definitions, migrations, and initialization for AURA.

Defines tables for conversation sessions, turns, long-term memories,
indexes, and FTS5 full-text search triggers.
"""

from contextlib import nullcontext
import sqlite3

from backend.app.database.connection import get_db_connection

# Schema DDL statements
SCHEMA_DDL = """
-- 1. Conversation Sessions
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New Conversation',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 2. Short-Term Conversation Turns
CREATE TABLE IF NOT EXISTS conversation_turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES conversation_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_turns_session_created ON conversation_turns(session_id, created_at ASC);

-- 3. Long-Term User Facts & Preferences
CREATE TABLE IF NOT EXISTS long_term_memories (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL CHECK(category IN ('identity', 'preference', 'project', 'constraint', 'general')),
    fact TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('explicit_user_instruction', 'extracted_from_dialog', 'manual_ui_entry')),
    confidence REAL NOT NULL DEFAULT 1.0,
    session_id TEXT REFERENCES conversation_sessions(id) ON DELETE SET NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_memories_category ON long_term_memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON long_term_memories(updated_at DESC);

-- 4. Full-Text Search Virtual Table for Long-Term Memories
CREATE VIRTUAL TABLE IF NOT EXISTS long_term_memories_fts USING fts5(
    fact,
    category,
    content='long_term_memories',
    content_rowid='rowid'
);

-- 5. FTS5 Synchronization Triggers
CREATE TRIGGER IF NOT EXISTS trg_memories_ai AFTER INSERT ON long_term_memories BEGIN
    INSERT INTO long_term_memories_fts(rowid, fact, category)
    VALUES (new.rowid, new.fact, new.category);
END;

CREATE TRIGGER IF NOT EXISTS trg_memories_ad AFTER DELETE ON long_term_memories BEGIN
    INSERT INTO long_term_memories_fts(long_term_memories_fts, rowid, fact, category)
    VALUES ('delete', old.rowid, old.fact, old.category);
END;

CREATE TRIGGER IF NOT EXISTS trg_memories_au AFTER UPDATE ON long_term_memories BEGIN
    INSERT INTO long_term_memories_fts(long_term_memories_fts, rowid, fact, category)
    VALUES ('delete', old.rowid, old.fact, old.category);
    INSERT INTO long_term_memories_fts(rowid, fact, category)
    VALUES (new.rowid, new.fact, new.category);
END;
"""


def init_db(
    conn: sqlite3.Connection | None = None,
    db_path: str | None = None,
) -> None:
    """
    Initialize database schema, tables, indexes, and triggers idempotently.

    Args:
        conn: An existing SQLite connection. If omitted, a new connection is created.
        db_path: Database path to connect to if ``conn`` is not provided.
    """
    owns_conn = conn is None
    active_conn = conn if conn is not None else get_db_connection(db_path=db_path)

    try:
        active_conn.executescript(SCHEMA_DDL)
        active_conn.commit()
    finally:
        if owns_conn:
            active_conn.close()
