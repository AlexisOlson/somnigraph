"""Database initialization, schema management, and ID resolution."""

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import sqlite_vec

from memory.constants import DATA_DIR, EMBEDDING_DIM
from memory.fts import _themes_for_fts

logger = logging.getLogger("claude-memory")

_schema_lock = threading.Lock()
_schema_initialized = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DB_PATH = DATA_DIR / "memory.db"

__all__ = ["get_db", "DB_PATH", "DATA_DIR"]

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def get_db() -> sqlite3.Connection:
    """Open database connection with sqlite-vec loaded."""
    global _schema_initialized
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")
    db.execute("PRAGMA foreign_keys=ON")
    db.row_factory = sqlite3.Row
    if not _schema_initialized:
        with _schema_lock:
            if not _schema_initialized:
                _init_schema(db)
                _schema_initialized = True
    return db


# ---------------------------------------------------------------------------
# ID resolution
# ---------------------------------------------------------------------------


def _resolve_id(db: sqlite3.Connection, prefix: str, status_filter: str = None) -> tuple[str | None, str | None]:
    """Resolve a short ID prefix to a full UUID.

    Returns (full_id, error). On success error is None; on failure full_id is None
    and error explains why (ambiguous match or no match).
    """
    if len(prefix) >= 32:
        return prefix, None  # already full (or near-full)
    like = f"{prefix}%"
    if status_filter:
        rows = db.execute(
            "SELECT id FROM memories WHERE id LIKE ? AND status = ?", (like, status_filter)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id FROM memories WHERE id LIKE ?", (like,)
        ).fetchall()
    if len(rows) == 1:
        return rows[0]["id"], None
    if len(rows) > 1:
        matches = ", ".join(r["id"][:8] for r in rows[:5])
        return None, f"Ambiguous ID prefix '{prefix}' matches {len(rows)} memories: {matches}"
    return None, f"No memory found matching prefix: {prefix}"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def _backup_if_needed(db: sqlite3.Connection, db_path: Path):
    """Create a backup before running migrations, if any are pending."""
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"memory_{timestamp}.db"
    # Use SQLite online backup API for consistency
    dest = sqlite3.connect(str(backup_path))
    db.backup(dest)
    dest.close()
    # Keep only last 3 backups
    backups = sorted(backup_dir.glob("memory_*.db"))
    for old in backups[:-3]:
        old.unlink()
    logger.info(f"Pre-migration backup: {backup_path.name}")


def _init_schema(db: sqlite3.Connection):
    """Create tables if they don't exist. Includes sleep-skill schema extensions."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            summary TEXT,
            category TEXT NOT NULL DEFAULT 'semantic',
            themes TEXT DEFAULT '[]',
            base_priority INTEGER NOT NULL DEFAULT 5,
            token_count INTEGER,
            created_at TEXT NOT NULL,
            last_accessed TEXT NOT NULL,
            access_count INTEGER DEFAULT 0,  -- legacy, see phased columns below
            startup_count INTEGER DEFAULT 0,
            recall_count INTEGER DEFAULT 0,
            reflect_count INTEGER DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active',
            superseded_by TEXT REFERENCES memories(id),
            source TEXT,
            layer TEXT DEFAULT 'detail',
            metadata TEXT DEFAULT '{}',
            -- Sleep skill fields
            valid_from TEXT,
            valid_until TEXT,
            generated_from TEXT DEFAULT '[]',
            last_sleep_processed TEXT,
            use_count INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_memories_category
            ON memories(category, status);
        CREATE INDEX IF NOT EXISTS idx_memories_priority
            ON memories(base_priority, last_accessed)
            WHERE status = 'active';
        CREATE INDEX IF NOT EXISTS idx_memories_layer
            ON memories(layer) WHERE status = 'active';
        CREATE INDEX IF NOT EXISTS idx_memories_pending
            ON memories(created_at) WHERE status = 'pending';
        CREATE INDEX IF NOT EXISTS idx_memories_status
            ON memories(status);

        CREATE TABLE IF NOT EXISTS memory_rowid_map (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL UNIQUE REFERENCES memories(id)
        );

        -- Memory edges (relationship graph, for sleep skill)
        CREATE TABLE IF NOT EXISTS memory_edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES memories(id),
            target_id TEXT NOT NULL REFERENCES memories(id),
            edge_type TEXT NOT NULL,
            note TEXT,
            created_at TEXT,
            created_by TEXT DEFAULT 'sleep'
        );

        CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_edges_type ON memory_edges(edge_type);

        -- Sleep log
        CREATE TABLE IF NOT EXISTS sleep_log (
            id TEXT PRIMARY KEY,
            started_at TEXT,
            completed_at TEXT,
            mode TEXT DEFAULT 'standard',
            memories_processed INTEGER,
            relationships_found INTEGER,
            summaries_refreshed INTEGER,
            gestalt_refreshed INTEGER DEFAULT 0,
            memories_pruned INTEGER,
            memories_dormanted INTEGER DEFAULT 0,
            contradictions_flagged INTEGER,
            fast_path_count INTEGER DEFAULT 0,
            full_pipeline_count INTEGER DEFAULT 0,
            gaps_found TEXT DEFAULT '[]',
            energy_before TEXT DEFAULT '{}',
            energy_after TEXT DEFAULT '{}',
            per_memory_changes TEXT DEFAULT '[]',
            report TEXT
        );

        -- Retrieval & lifecycle event log (append-only)
        CREATE TABLE IF NOT EXISTS memory_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            query TEXT,
            session_id TEXT,
            co_memory_ids TEXT DEFAULT '[]',
            similarity_score REAL,
            context TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );

        CREATE INDEX IF NOT EXISTS idx_events_memory
            ON memory_events(memory_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_events_type
            ON memory_events(event_type, created_at);
        CREATE INDEX IF NOT EXISTS idx_events_session
            ON memory_events(session_id)
            WHERE session_id IS NOT NULL;
    """)

    # sqlite-vec virtual table (must be outside executescript)
    # EMBEDDING_DIM is set at creation time — changing backend won't alter an existing table.
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
            embedding float[{EMBEDDING_DIM}] distance_metric=cosine
        )
    """)

    # FTS5 virtual table — indexes only curated metadata (summary + themes)
    # Content is handled by vector search; FTS5 owns keyword precision
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            summary, themes, tokenize='porter unicode61'
        )
    """)

    # Migrations
    cols = {r[1] for r in db.execute("PRAGMA table_info(memories)").fetchall()}

    # Backup before migrations if any are pending
    if cols and ("confidence" not in cols or "flags" not in cols):
        _backup_if_needed(db, DB_PATH)

    # Drop deprecated counter columns (replaced by memory_events)
    for col in ("useful_count", "not_useful_count"):
        if col in cols:
            db.execute(f"ALTER TABLE memories DROP COLUMN {col}")

    # Add per-memory decay rate column
    if "decay_rate" not in cols:
        db.execute("ALTER TABLE memories ADD COLUMN decay_rate REAL DEFAULT NULL")

    # Add shadow-load column (Phase 1.2 — continuous suppression signal)
    if "shadow_load" not in cols:
        db.execute("ALTER TABLE memories ADD COLUMN shadow_load REAL DEFAULT 0.0")

    # Add topic_id column (REM taxonomy-based clustering)
    if "topic_id" not in cols:
        db.execute("ALTER TABLE memories ADD COLUMN topic_id TEXT DEFAULT NULL")

    # Add flags column (retention immunity tags: "pinned", "keep")
    if "flags" not in cols:
        db.execute("ALTER TABLE memories ADD COLUMN flags TEXT DEFAULT '[]'")

    # Add session_id column to memories
    if "session_id" not in cols:
        db.execute("ALTER TABLE memories ADD COLUMN session_id TEXT DEFAULT NULL")

    # Add confidence column to memories
    if "confidence" not in cols:
        db.execute("ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 0.5")
        # Backfill based on source
        db.execute("UPDATE memories SET confidence = 0.6 WHERE source = 'correction' AND confidence = 0.5")
        db.execute("UPDATE memories SET confidence = 0.7 WHERE layer IN ('gestalt', 'summary') AND confidence = 0.5")
        db.execute("UPDATE memories SET confidence = 0.3 WHERE status = 'pending' AND confidence = 0.5")

    # Edge schema v2 migration
    edge_cols = {r[1] for r in db.execute("PRAGMA table_info(memory_edges)").fetchall()}

    if "linking_context" not in edge_cols:
        db.execute("ALTER TABLE memory_edges ADD COLUMN linking_context TEXT")
    if "linking_embedding" not in edge_cols:
        db.execute("ALTER TABLE memory_edges ADD COLUMN linking_embedding BLOB")
    if "flags" not in edge_cols:
        db.execute("ALTER TABLE memory_edges ADD COLUMN flags TEXT DEFAULT '[]'")
    if "features" not in edge_cols:
        db.execute("ALTER TABLE memory_edges ADD COLUMN features TEXT DEFAULT '{}'")
    if "weight" not in edge_cols:
        db.execute("ALTER TABLE memory_edges ADD COLUMN weight REAL DEFAULT 0.5")

    # Refresh edge_cols after potential column additions
    edge_cols = {r[1] for r in db.execute("PRAGMA table_info(memory_edges)").fetchall()}

    # Backfill: convert edge_type → flags + linking_context (idempotent per-row)
    if "edge_type" in edge_cols and "flags" in edge_cols:
        type_to_flags = {
            "derived_from": '["derivation"]',
            "hard_contradiction": '["contradiction"]',
            "soft_contradiction": '["contradiction"]',
            "evolved_from": '["revision"]',
            "temporal_evolution": '["revision"]',
        }
        for etype, flags_json in type_to_flags.items():
            db.execute(
                "UPDATE memory_edges SET flags = ? "
                "WHERE edge_type = ? AND (flags IS NULL OR flags = '[]')",
                (flags_json, etype),
            )
        # Copy note → linking_context for edges that have notes
        db.execute(
            "UPDATE memory_edges SET linking_context = note "
            "WHERE note IS NOT NULL AND note != '' "
            "AND (linking_context IS NULL OR linking_context = '')"
        )

    # Add REM-phase columns to sleep_log
    sleep_cols = {r[1] for r in db.execute("PRAGMA table_info(sleep_log)").fetchall()}
    if "shadow_updates" not in sleep_cols:
        db.execute("ALTER TABLE sleep_log ADD COLUMN shadow_updates INTEGER DEFAULT 0")
    if "health_risk" not in sleep_cols:
        db.execute("ALTER TABLE sleep_log ADD COLUMN health_risk REAL DEFAULT 0.0")

    # Migrate feedback events: 3-dimension integers → utility float + durability float
    _migrated = db.execute(
        "SELECT COUNT(*) FROM memory_events WHERE event_type = 'feedback' AND context IS NOT NULL "
        "AND context LIKE '%impact%' AND context NOT LIKE '%utility%'"
    ).fetchone()[0]
    if _migrated:
        rows = db.execute(
            "SELECT id, context FROM memory_events WHERE event_type = 'feedback' AND context IS NOT NULL "
            "AND context LIKE '%impact%' AND context NOT LIKE '%utility%'"
        ).fetchall()
        for row in rows:
            try:
                ctx = json.loads(row["context"])
            except (json.JSONDecodeError, TypeError):
                continue
            if "utility" not in ctx and "impact" in ctx:
                utility = ctx["impact"] / 3.0
                new_ctx = {"utility": round(utility, 4)}
                dur = ctx.get("durability")
                if dur == "enduring":
                    new_ctx["durability"] = 1.0
                elif dur == "stale":
                    new_ctx["durability"] = -1.0
                if ctx.get("reason"):
                    new_ctx["reason"] = ctx["reason"]
                db.execute(
                    "UPDATE memory_events SET context = ? WHERE id = ?",
                    (json.dumps(new_ctx), row["id"]),
                )
        logger.info(f"Migrated {_migrated} feedback events to utility format")

    # Phased access count migration: split access_count into startup/recall/reflect
    if "startup_count" not in cols:
        db.execute("ALTER TABLE memories ADD COLUMN startup_count INTEGER DEFAULT 0")
        db.execute("ALTER TABLE memories ADD COLUMN recall_count INTEGER DEFAULT 0")
        db.execute("ALTER TABLE memories ADD COLUMN reflect_count INTEGER DEFAULT 0")
        # Move polluted access_count to startup_count (dominant source), zero access_count
        db.execute("UPDATE memories SET startup_count = access_count, access_count = 0")
        logger.info("Migrated access_count to phased columns (startup/recall/reflect)")

    # FTS5 themes column migration
    fts_cols = {r[1] for r in db.execute("PRAGMA table_info(memory_fts)").fetchall()}
    if "themes" not in fts_cols:
        db.execute("DROP TABLE memory_fts")
        db.execute("""
            CREATE VIRTUAL TABLE memory_fts USING fts5(
                summary, themes, tokenize='porter unicode61'
            )
        """)
        # Repopulate from source-of-truth tables
        rows = db.execute("""
            SELECT mrm.rowid, m.summary, m.themes
            FROM memory_rowid_map mrm
            JOIN memories m ON mrm.memory_id = m.id
            WHERE m.status IN ('active', 'pending')
        """).fetchall()
        for row in rows:
            db.execute(
                "INSERT INTO memory_fts (rowid, summary, themes) VALUES (?, ?, ?)",
                (row[0], row[1] or "", _themes_for_fts(row[2])),
            )
        logger.info(f"Migrated FTS5 table: added themes column, repopulated {len(rows)} rows")

    # FTS5 content column removal — drop content from FTS5, keep only summary + themes
    # Content is searched by vector; FTS5 owns keyword precision over curated metadata
    if "content" in fts_cols:
        db.execute("DROP TABLE memory_fts")
        db.execute("""
            CREATE VIRTUAL TABLE memory_fts USING fts5(
                summary, themes, tokenize='porter unicode61'
            )
        """)
        rows = db.execute("""
            SELECT mrm.rowid, m.summary, m.themes
            FROM memory_rowid_map mrm
            JOIN memories m ON mrm.memory_id = m.id
            WHERE m.status IN ('active', 'pending')
        """).fetchall()
        for row in rows:
            db.execute(
                "INSERT INTO memory_fts (rowid, summary, themes) VALUES (?, ?, ?)",
                (row[0], row[1] or "", _themes_for_fts(row[2])),
            )
        logger.info(f"Migrated FTS5: removed content column, repopulated {len(rows)} rows")

    db.commit()
