"""Write path — memory insertion into all three tables."""

import json
import sqlite3
from datetime import datetime, timezone

from memory.embeddings import count_tokens
from memory.fts import _themes_for_fts
from memory.themes import normalize_themes
from memory.vectors import serialize_f32


def _insert_memory(
    db: sqlite3.Connection,
    memory_id: str,
    content: str,
    summary: str,
    category: str,
    themes_json: str,
    priority: int,
    source: str,
    status: str,
    embedding: list[float],
    decay_rate: float = None,
    layer: str = "detail",
    generated_from: list[str] = None,
    confidence: float = None,
    flags_json: str = "[]",
    session_id: str = None,
):
    """Insert a memory into all three tables."""
    # Safety net: normalize themes for write paths that bypass remember()
    themes_parsed = json.loads(themes_json) if themes_json else []
    themes_parsed = normalize_themes(themes_parsed, content=content)
    themes_json = json.dumps(themes_parsed)

    now = datetime.now(timezone.utc).isoformat()
    token_count = count_tokens(content)

    # Auto-summary for short content
    if not summary and token_count < 50:
        summary = content.split(".")[0] if "." in content else content[:80]

    gen_from_json = json.dumps(generated_from) if generated_from else "[]"

    db.execute(
        """INSERT INTO memories
           (id, content, summary, category, themes, base_priority, token_count,
            created_at, last_accessed, access_count, status, source, layer, metadata,
            decay_rate, generated_from, confidence, flags, session_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, '{}', ?, ?, ?, ?, ?)""",
        (memory_id, content, summary, category, themes_json, priority,
         token_count, now, now, status, source, layer, decay_rate, gen_from_json,
         confidence, flags_json, session_id),
    )

    # Get integer rowid for sqlite-vec
    db.execute(
        "INSERT INTO memory_rowid_map (memory_id) VALUES (?)", (memory_id,)
    )
    vec_rowid = db.execute(
        "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (memory_id,)
    ).fetchone()["rowid"]

    # Insert embedding
    db.execute(
        "INSERT INTO memory_vec (rowid, embedding) VALUES (?, ?)",
        (vec_rowid, serialize_f32(embedding)),
    )

    # Insert into FTS5 (summary + themes only — content is for vector search)
    themes_fts = _themes_for_fts(themes_json)
    db.execute(
        "INSERT INTO memory_fts (rowid, summary, themes) VALUES (?, ?, ?)",
        (vec_rowid, summary or "", themes_fts),
    )
