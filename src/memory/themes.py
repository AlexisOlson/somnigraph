"""Theme normalization — variant mapping, content-aware expansion, bulk operations."""

import json

from memory.constants import DATA_DIR
from memory.fts import _themes_for_fts

__all__ = [
    "THEME_VARIANTS", "CONTENT_THEME_PHRASES", "normalize_themes",
    "add_theme_mapping", "normalize_all_themes", "split_theme_on_memories",
]

# ---------------------------------------------------------------------------
# Theme variant mappings — loaded from config file or small defaults
# ---------------------------------------------------------------------------

_DEFAULT_THEME_VARIANTS = {
    "recall_feedback": "recall-feedback",
    "startup_load": "startup-load",
    "decisions": "decision",
}

LEARNED_MAPPINGS_PATH = DATA_DIR / "theme_mappings.json"
_learned_mappings: dict | None = None


def _load_theme_variants() -> dict[str, str]:
    """Load theme variants from config file, falling back to defaults."""
    variants_file = DATA_DIR / "theme_variants.json"
    if variants_file.exists():
        try:
            loaded = json.loads(variants_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_DEFAULT_THEME_VARIANTS)


THEME_VARIANTS = _load_theme_variants()

# ---------------------------------------------------------------------------
# Content-aware theme phrases — loaded from config file or small defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONTENT_PHRASES = {
    "memory system": "memory-system",
    "sleep pipeline": "sleep-pipeline",
}


def _load_content_phrases() -> dict[str, str]:
    """Load content theme phrases from config file, falling back to defaults."""
    phrases_file = DATA_DIR / "content_phrases.json"
    if phrases_file.exists():
        try:
            loaded = json.loads(phrases_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_DEFAULT_CONTENT_PHRASES)


CONTENT_THEME_PHRASES = _load_content_phrases()


# ---------------------------------------------------------------------------
# Mapping resolution
# ---------------------------------------------------------------------------


def _get_all_mappings() -> dict[str, str]:
    """Return combined THEME_VARIANTS + learned mappings from disk."""
    global _learned_mappings
    if _learned_mappings is None:
        if LEARNED_MAPPINGS_PATH.exists():
            try:
                _learned_mappings = json.loads(LEARNED_MAPPINGS_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                _learned_mappings = {}
        else:
            _learned_mappings = {}
    return {**THEME_VARIANTS, **_learned_mappings}


def add_theme_mapping(from_theme: str, to_theme: str):
    """Persist a learned theme mapping."""
    global _learned_mappings
    if _learned_mappings is None:
        _get_all_mappings()
    _learned_mappings[from_theme] = to_theme
    LEARNED_MAPPINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEARNED_MAPPINGS_PATH.write_text(
        json.dumps(_learned_mappings, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _auto_hyphenate(theme: str) -> str:
    """Convert underscores/spaces to hyphens (the canonical separator)."""
    if "_" in theme or " " in theme:
        return theme.replace("_", "-").replace(" ", "-")
    return theme


def normalize_themes(themes_list: list[str], content: str = "") -> list[str]:
    """Normalize variant themes to canonical forms. Optionally auto-expand from content. Deduplicates."""
    mappings = _get_all_mappings()
    result = [mappings.get(t, _auto_hyphenate(t)) for t in themes_list]

    # Content-aware auto-expansion
    if content:
        content_lower = content.lower()
        for phrase, tag in CONTENT_THEME_PHRASES.items():
            if phrase in content_lower and tag not in result:
                result.append(tag)

    seen = set()
    return [t for t in result if not (t in seen or seen.add(t))]


def normalize_all_themes(db=None) -> int:
    """Normalize themes across all active/pending memories. Returns count of updated rows.

    Called during sleep consolidation or standalone cleanup.
    """
    from memory.db import get_db

    own_db = db is None
    if own_db:
        db = get_db()
    try:
        rows = db.execute(
            "SELECT id, themes FROM memories WHERE status IN ('active', 'pending')"
        ).fetchall()

        updated = 0
        for row in rows:
            themes = json.loads(row["themes"]) if row["themes"] else []
            normalized = normalize_themes(themes)
            if normalized != themes:
                db.execute(
                    "UPDATE memories SET themes = ? WHERE id = ?",
                    (json.dumps(normalized), row["id"]),
                )
                # Update FTS5 index
                rowid_row = db.execute(
                    "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (row["id"],)
                ).fetchone()
                if rowid_row:
                    db.execute(
                        "UPDATE memory_fts SET themes = ? WHERE rowid = ?",
                        (_themes_for_fts(json.dumps(normalized)), rowid_row["rowid"]),
                    )
                updated += 1

        if updated:
            db.commit()
        return updated
    finally:
        if own_db:
            db.close()


def split_theme_on_memories(compound: str, components: list[str], db=None) -> int:
    """Replace compound theme with components on all memories that have it."""
    from memory.db import get_db

    own_db = db is None
    if own_db:
        db = get_db()
    try:
        rows = db.execute(
            "SELECT id, themes FROM memories WHERE status IN ('active', 'pending')"
        ).fetchall()
        updated = 0
        for row in rows:
            themes = json.loads(row["themes"]) if row["themes"] else []
            if compound in themes:
                themes.remove(compound)
                for c in components:
                    if c not in themes:
                        themes.append(c)
                themes = normalize_themes(themes)
                db.execute("UPDATE memories SET themes = ? WHERE id = ?",
                           (json.dumps(themes), row["id"]))
                rowid_row = db.execute(
                    "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (row["id"],)
                ).fetchone()
                if rowid_row:
                    db.execute("UPDATE memory_fts SET themes = ? WHERE rowid = ?",
                               (_themes_for_fts(json.dumps(themes)), rowid_row["rowid"]))
                updated += 1
        if updated:
            db.commit()
        return updated
    finally:
        if own_db:
            db.close()
