"""
Conversation Manager for AURA conversation sessions and turn memory.

Persists and retrieves conversation history backed by SQLite using parameterized SQL
and transactional context managers.
"""

from __future__ import annotations

from typing import Literal
import uuid

from backend.app.config import settings
from backend.app.database.connection import get_db_context
from backend.app.models.memory import ConversationSession, ConversationTurn

VALID_ROLES = {"user", "assistant", "system"}


class ConversationManager:
    """
    Manages session lifecycles, turn logging, and short-term memory retrieval.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialize the manager.

        Args:
            db_path: Path to the SQLite database file or ':memory:'.
                     Defaults to settings.DB_PATH if omitted.
        """
        self.db_path = db_path if db_path is not None else settings.DB_PATH

    def create_session(
        self,
        session_id: str | None = None,
        title: str = "New Conversation",
    ) -> ConversationSession:
        """
        Create and persist a new conversation session.

        Args:
            session_id: Optional explicit UUID/string ID. Generates UUID4 if omitted.
            title: Human-readable title for the session.

        Returns:
            The created ConversationSession instance.
        """
        resolved_id = session_id if session_id and session_id.strip() else str(uuid.uuid4())
        resolved_title = title.strip() if title and title.strip() else "New Conversation"

        with get_db_context(db_path=self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO conversation_sessions (id, title)
                VALUES (?, ?);
                """,
                (resolved_id, resolved_title),
            )
            row = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversation_sessions
                WHERE id = ?;
                """,
                (resolved_id,),
            ).fetchone()

        if row is None:
            raise RuntimeError(f"Failed to create session with id '{resolved_id}'")

        return ConversationSession.model_validate(dict(row))

    def get_session(self, session_id: str) -> ConversationSession | None:
        """
        Retrieve a conversation session by its ID.

        Args:
            session_id: Unique identifier of the session.

        Returns:
            ConversationSession if found, None otherwise.
        """
        if not session_id or not session_id.strip():
            return None

        with get_db_context(db_path=self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversation_sessions
                WHERE id = ?;
                """,
                (session_id.strip(),),
            ).fetchone()

        if row is None:
            return None

        return ConversationSession.model_validate(dict(row))

    def get_or_create_session(
        self,
        session_id: str | None = None,
        default_title: str = "New Conversation",
    ) -> ConversationSession:
        """
        Retrieve an existing session or create a new one if not found or if ID is omitted.

        Args:
            session_id: Optional session identifier.
            default_title: Title to assign if creating a new session.

        Returns:
            Existing or newly created ConversationSession.
        """
        if session_id and session_id.strip():
            existing = self.get_session(session_id.strip())
            if existing is not None:
                return existing
            return self.create_session(session_id=session_id.strip(), title=default_title)

        return self.create_session(title=default_title)

    def add_turn(
        self,
        session_id: str,
        role: Literal["user", "assistant", "system"] | str,
        content: str,
        token_count: int = 0,
        turn_id: str | None = None,
    ) -> ConversationTurn:
        """
        Record a conversation turn for a session and update the session's timestamp.

        Args:
            session_id: ID of the owning session.
            role: Must be 'user', 'assistant', or 'system'.
            content: Message text.
            token_count: Optional token count for the turn.
            turn_id: Optional explicit turn ID; generates UUID4 if omitted.

        Returns:
            The recorded ConversationTurn instance.

        Raises:
            ValueError: If role is invalid, content is empty, or session does not exist.
        """
        normalized_role = role.strip().lower() if isinstance(role, str) else role
        if normalized_role not in VALID_ROLES:
            raise ValueError(
                f"Invalid role: '{role}'. Must be one of: {', '.join(sorted(VALID_ROLES))}"
            )

        stripped_content = content.strip()
        if not stripped_content:
            raise ValueError("content must not be empty or whitespace-only")

        resolved_turn_id = turn_id.strip() if turn_id and turn_id.strip() else str(uuid.uuid4())

        with get_db_context(db_path=self.db_path) as conn:
            session_row = conn.execute(
                "SELECT id FROM conversation_sessions WHERE id = ?;",
                (session_id,),
            ).fetchone()
            if session_row is None:
                raise ValueError(f"Session with id '{session_id}' does not exist")

            conn.execute(
                """
                INSERT INTO conversation_turns (id, session_id, role, content, token_count)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    resolved_turn_id,
                    session_id,
                    normalized_role,
                    stripped_content,
                    max(0, token_count),
                ),
            )

            conn.execute(
                """
                UPDATE conversation_sessions
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?;
                """,
                (session_id,),
            )

            row = conn.execute(
                """
                SELECT id, session_id, role, content, token_count, created_at
                FROM conversation_turns
                WHERE id = ?;
                """,
                (resolved_turn_id,),
            ).fetchone()

        if row is None:
            raise RuntimeError(f"Failed to record turn with id '{resolved_turn_id}'")

        return ConversationTurn.model_validate(dict(row))

    def add_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        user_token_count: int = 0,
        assistant_token_count: int = 0,
        user_turn_id: str | None = None,
        assistant_turn_id: str | None = None,
    ) -> tuple[ConversationTurn, ConversationTurn]:
        """
        Record a pair of user and assistant turns in a single atomic database transaction.

        Args:
            session_id: Target session identifier.
            user_content: User message text.
            assistant_content: Assistant reply text.
            user_token_count: Optional token count for user turn.
            assistant_token_count: Optional token count for assistant turn.
            user_turn_id: Optional explicit turn ID for user turn.
            assistant_turn_id: Optional explicit turn ID for assistant turn.

        Returns:
            A tuple of (user_turn, assistant_turn) ConversationTurn models.

        Raises:
            ValueError: If session does not exist or message contents are empty.
        """
        stripped_user = user_content.strip()
        if not stripped_user:
            raise ValueError("user_content must not be empty or whitespace-only")

        stripped_assistant = assistant_content.strip()
        if not stripped_assistant:
            raise ValueError("assistant_content must not be empty or whitespace-only")

        resolved_user_id = (
            user_turn_id.strip() if user_turn_id and user_turn_id.strip() else str(uuid.uuid4())
        )
        resolved_assistant_id = (
            assistant_turn_id.strip()
            if assistant_turn_id and assistant_turn_id.strip()
            else str(uuid.uuid4())
        )

        with get_db_context(db_path=self.db_path) as conn:
            session_row = conn.execute(
                "SELECT id FROM conversation_sessions WHERE id = ?;",
                (session_id,),
            ).fetchone()
            if session_row is None:
                raise ValueError(f"Session with id '{session_id}' does not exist")

            conn.execute(
                """
                INSERT INTO conversation_turns (id, session_id, role, content, token_count)
                VALUES (?, ?, 'user', ?, ?);
                """,
                (
                    resolved_user_id,
                    session_id,
                    stripped_user,
                    max(0, user_token_count),
                ),
            )

            conn.execute(
                """
                INSERT INTO conversation_turns (id, session_id, role, content, token_count)
                VALUES (?, ?, 'assistant', ?, ?);
                """,
                (
                    resolved_assistant_id,
                    session_id,
                    stripped_assistant,
                    max(0, assistant_token_count),
                ),
            )

            conn.execute(
                """
                UPDATE conversation_sessions
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?;
                """,
                (session_id,),
            )

            user_row = conn.execute(
                """
                SELECT id, session_id, role, content, token_count, created_at
                FROM conversation_turns
                WHERE id = ?;
                """,
                (resolved_user_id,),
            ).fetchone()

            assistant_row = conn.execute(
                """
                SELECT id, session_id, role, content, token_count, created_at
                FROM conversation_turns
                WHERE id = ?;
                """,
                (resolved_assistant_id,),
            ).fetchone()

        if user_row is None or assistant_row is None:
            raise RuntimeError("Failed to record conversation exchange")

        return (
            ConversationTurn.model_validate(dict(user_row)),
            ConversationTurn.model_validate(dict(assistant_row)),
        )

    def get_recent_turns(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[ConversationTurn]:
        """
        Fetch the most recent turns for a session in chronological order (oldest to newest).

        Args:
            session_id: Target session identifier.
            limit: Maximum number of recent turns to retrieve.

        Returns:
            List of ConversationTurn objects ordered chronologically (ASC).
        """
        if not session_id or not session_id.strip() or limit <= 0:
            return []

        with get_db_context(db_path=self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, session_id, role, content, token_count, created_at
                FROM (
                    SELECT id, session_id, role, content, token_count, created_at, rowid
                    FROM conversation_turns
                    WHERE session_id = ?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC, rowid ASC;
                """,
                (session_id.strip(), limit),
            )
            rows = cursor.fetchall()

        return [ConversationTurn.model_validate(dict(r)) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and cascade delete all its associated turns.

        Args:
            session_id: ID of the session to delete.

        Returns:
            True if the session existed and was deleted; False otherwise.
        """
        if not session_id or not session_id.strip():
            return False

        with get_db_context(db_path=self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_sessions WHERE id = ?;",
                (session_id.strip(),),
            )
            return cursor.rowcount > 0

    def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ConversationSession]:
        """
        List conversation sessions ordered by most recently updated.

        Args:
            limit: Maximum number of sessions to return.
            offset: Number of records to skip.

        Returns:
            List of ConversationSession objects.
        """
        if limit <= 0:
            return []

        with get_db_context(db_path=self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversation_sessions
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ? OFFSET ?;
                """,
                (limit, max(0, offset)),
            )
            rows = cursor.fetchall()

        return [ConversationSession.model_validate(dict(r)) for r in rows]
