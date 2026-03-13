"""Event logging and row helpers."""

import json
import logging
import sqlite3

logger = logging.getLogger("claude-memory")


def _row_get(row: sqlite3.Row, key: str, default=None):
    """Safe accessor for sqlite3.Row — returns default if column doesn't exist."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def _log_event(
    db: sqlite3.Connection,
    memory_id: str,
    event_type: str,
    query: str = None,
    session_id: str = None,
    co_memory_ids: list = None,
    similarity_score: float = None,
    context: dict = None,
):
    """Append an event to the memory_events log.

    If session_id is not provided, auto-detects from the current Claude Code session.
    """
    if session_id is None:
        from memory.session import get_session_id
        session_id = get_session_id()
    db.execute(
        """INSERT INTO memory_events
           (memory_id, event_type, query, session_id, co_memory_ids,
            similarity_score, context)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            memory_id,
            event_type,
            query,
            session_id,
            json.dumps(co_memory_ids) if co_memory_ids else None,
            similarity_score,
            json.dumps(context) if context else None,
        ),
    )
