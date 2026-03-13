"""Session ID detection — discovers the current Claude Code session UUID."""

import logging
from pathlib import Path

logger = logging.getLogger("claude-memory")

# Module-level session state
_current_session_id: str | None = None


def detect_session_id() -> str | None:
    """Find the current session ID from the most recent JSONL transcript file.

    Claude Code writes session transcripts to ~/.claude/projects/<project>/<session-uuid>.jsonl.
    The filename IS the session UUID. We find the newest one.
    """
    global _current_session_id
    if _current_session_id is not None:
        return _current_session_id

    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        return None

    # Find the most recently modified .jsonl across all project dirs
    newest: Path | None = None
    newest_mtime = 0.0
    for jsonl in projects_dir.glob("*/*.jsonl"):
        mt = jsonl.stat().st_mtime
        if mt > newest_mtime:
            newest_mtime = mt
            newest = jsonl

    if newest is None:
        return None

    session_id = newest.stem  # filename without .jsonl
    # Validate it looks like a UUID (8-4-4-4-12 hex)
    if len(session_id) == 36 and session_id.count("-") == 4:
        _current_session_id = session_id
        logger.info(f"Detected session: {session_id[:8]}...")
        return session_id

    return None


def get_session_id() -> str | None:
    """Return the cached session ID, or try detection if not yet cached."""
    if _current_session_id is not None:
        return _current_session_id
    return detect_session_id()


def reset_session_id():
    """Clear cached session ID (for testing)."""
    global _current_session_id
    _current_session_id = None
