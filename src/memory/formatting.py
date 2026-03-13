"""Memory formatting helpers for display."""

import json
import sqlite3


def format_memory_compact(row: sqlite3.Row) -> str:
    """One-line compact format for startup_load."""
    themes = json.loads(row["themes"]) if row["themes"] else []
    if len(themes) > 5:
        themes_str = f" ({', '.join(themes[:5])}, ...)"
    elif themes:
        themes_str = f" ({', '.join(themes)})"
    else:
        themes_str = ""
    summary = row["summary"] or row["content"][:80]
    prefix = "[?]" if "question" in themes else f"[{row['category']}]"
    return f"{prefix} {summary}{themes_str}"


def format_memory_full(row: sqlite3.Row) -> str:
    """Full format for recall results."""
    themes = json.loads(row["themes"]) if row["themes"] else []
    themes_str = f" ({', '.join(themes)})" if themes else ""
    short_id = row["id"][:8]
    lines = [
        f"**[{row['category']}]** {row['summary'] or '(no summary)'}",
        f"  {row['content']}",
        f"  ID: {short_id}{themes_str}",
    ]
    return "\n".join(lines)


def format_memory_pending(row: sqlite3.Row, idx: int) -> str:
    """Numbered format for pending review."""
    themes = json.loads(row["themes"]) if row["themes"] else []
    themes_str = f" ({', '.join(themes)})" if themes else ""
    return (
        f"{idx}. [{row['category']}] p{row['base_priority']}{themes_str}\n"
        f"   {row['content'][:200]}\n"
        f"   ID: {row['id'][:8]} | Source: {row['source']} | Created: {row['created_at'][:10]}"
    )
