"""FTS5 query sanitization and helpers."""

import json
import re

from memory.constants import DATA_DIR

__all__ = ["KNOWN_PHRASES", "sanitize_fts_query", "update_fts", "delete_fts"]

# Default known phrases — overridden by DATA_DIR/known_phrases.json if present
_DEFAULT_KNOWN_PHRASES = {
    "memory system", "memory server", "sleep pipeline", "recall feedback",
    "startup load",
}


def _load_known_phrases() -> set[str]:
    """Load known phrases from config file, falling back to defaults."""
    phrases_file = DATA_DIR / "known_phrases.json"
    if phrases_file.exists():
        try:
            loaded = json.loads(phrases_file.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                return set(loaded)
        except (json.JSONDecodeError, OSError):
            pass
    return set(_DEFAULT_KNOWN_PHRASES)


KNOWN_PHRASES = _load_known_phrases()


def sanitize_fts_query(query: str) -> str:
    """Sanitize a query string for FTS5 MATCH with phrase awareness.

    Preserves explicitly quoted phrases. Auto-detects known multi-word phrases.
    Remaining tokens are treated as single keywords. Groups are AND-joined (<=3)
    or OR-joined (4+) for balanced precision/breadth.
    """
    groups = []  # each entry is a string: either '"phrase"' or '"word"'

    # 1. Extract explicitly quoted phrases
    remaining = query
    explicit_phrases = re.findall(r'"([^"]+)"', remaining)
    for phrase in explicit_phrases:
        phrase = phrase.strip()
        if phrase:
            groups.append(f'"{phrase}"')
    # Remove quoted segments from remaining text
    remaining = re.sub(r'"[^"]*"', " ", remaining)

    # 2. Auto-detect known multi-word phrases in remaining text
    remaining_lower = remaining.lower()
    consumed_spans = []  # (start, end) spans consumed by known phrases
    for phrase in sorted(KNOWN_PHRASES, key=len, reverse=True):  # longest first
        idx = remaining_lower.find(phrase)
        if idx >= 0:
            end = idx + len(phrase)
            # Check no overlap with already-consumed spans
            if not any(s <= idx < e or s < end <= e for s, e in consumed_spans):
                groups.append(f'"{phrase}"')
                consumed_spans.append((idx, end))
                # Blank out consumed portion
                remaining = remaining[:idx] + " " * len(phrase) + remaining[end:]
                remaining_lower = remaining.lower()

    # 3. Tokenize remaining words
    for token in remaining.split():
        # Strip FTS5 special chars, skip single-char tokens
        cleaned = re.sub(r'["\'\*\(\)\+\^]', "", token).strip()
        if len(cleaned) > 1:
            if f'"{cleaned}"' not in groups and f'"{cleaned.lower()}"' not in groups:
                groups.append(f'"{cleaned}"')

    if not groups:
        return '""'

    # Deduplicate (preserve order)
    seen = set()
    unique = []
    for g in groups:
        gl = g.lower()
        if gl not in seen:
            seen.add(gl)
            unique.append(g)

    # <=3 groups: AND for precision; 4+: OR for breadth
    joiner = " AND " if len(unique) <= 3 else " OR "
    return joiner.join(unique)


def update_fts(db, memory_id: str) -> bool:
    """Re-sync memory_fts row from current memories data. Returns True if updated."""
    rowid_row = db.execute(
        "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    if not rowid_row:
        return False
    rid = rowid_row["rowid"]
    mem = db.execute(
        "SELECT summary, themes FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    if not mem:
        return False
    db.execute("DELETE FROM memory_fts WHERE rowid = ?", (rid,))
    db.execute(
        "INSERT INTO memory_fts (rowid, summary, themes) VALUES (?, ?, ?)",
        (rid, mem["summary"] or "", _themes_for_fts(mem["themes"])),
    )
    return True


def delete_fts(db, memory_id: str) -> bool:
    """Remove memory_fts row. Returns True if deleted."""
    rowid_row = db.execute(
        "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    if not rowid_row:
        return False
    db.execute("DELETE FROM memory_fts WHERE rowid = ?", (rowid_row["rowid"],))
    return True


def _themes_for_fts(themes_json: str) -> str:
    """Convert themes JSON array to space-separated string for FTS5 indexing."""
    if not themes_json:
        return ""
    try:
        themes_list = json.loads(themes_json) if isinstance(themes_json, str) else themes_json
        if isinstance(themes_list, list) and themes_list:
            return " ".join(str(t) for t in themes_list)
    except (json.JSONDecodeError, TypeError):
        pass
    return ""
