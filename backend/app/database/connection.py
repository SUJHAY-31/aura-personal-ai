"""
SQLite database connection and session management.

Provides configured SQLite connections with foreign keys, WAL mode,
and busy timeouts for local concurrency safety.
"""

from collections.abc import Generator
from contextlib import contextmanager
import sqlite3

from backend.app.config import settings


def get_db_connection(db_path: str | None = None) -> sqlite3.Connection:
    """
    Create and configure a SQLite connection.

    Args:
        db_path: Path to the SQLite file or ':memory:'. Defaults to ``settings.DB_PATH``.

    Returns:
        A configured ``sqlite3.Connection`` instance with foreign keys enabled.
    """
    resolved_path = db_path if db_path is not None else settings.DB_PATH
    conn = sqlite3.connect(
        database=resolved_path,
        timeout=5.0,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row

    # Enforce foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON;")
    # Set busy timeout in milliseconds to prevent immediate locking failures
    conn.execute("PRAGMA busy_timeout = 5000;")

    # In-memory databases do not support or require WAL mode
    if resolved_path != ":memory:" and not resolved_path.startswith("file::memory:"):
        conn.execute("PRAGMA journal_mode = WAL;")

    return conn


@contextmanager
def get_db_context(
    db_path: str | None = None,
) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager providing a transactional SQLite connection.

    Automatically commits on normal block exit, rolls back on exception,
    and closes the connection upon completion.
    """
    conn = get_db_connection(db_path=db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
