"""Graph helpers — edge creation, related memory search, shadow load, temporal evolution."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from memory.constants import RRF_K, RRF_VEC_WEIGHT
from memory.fts import sanitize_fts_query
from memory.vectors import serialize_f32


def _source_confidence_modifier(source: str) -> float:
    """Return a confidence modifier based on memory source.

    Higher = more trusted. Applied as a multiplier to classification confidence.
    """
    modifiers = {
        "manual": 1.0,
        "correction": 0.95,
        "session": 0.9,
        "journal": 0.9,
        "auto": 0.7,
    }
    return modifiers.get(source or "session", 0.8)


def _find_related_memories(
    db: sqlite3.Connection, memory_id: str, content: str, limit: int = 5
) -> list[dict]:
    """Find memories related to the given one via hybrid retrieval (vector + FTS5 + RRF).

    Uses the stored embedding — no embed_text() API call needed.
    Excludes the source memory itself. Only searches active memories.
    Returns list of {"row": sqlite3.Row, "rrf_score": float, "vec_distance": float}.
    """
    # Get vec rowid for this memory
    vec_map = db.execute(
        "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    if not vec_map:
        return []

    # --- Vector search using stored embedding (no API call) ---
    vec_results = db.execute(
        """
        SELECT rowid, distance FROM memory_vec
        WHERE embedding MATCH (SELECT embedding FROM memory_vec WHERE rowid = ?)
        AND k = ?
        ORDER BY distance
        """,
        (vec_map["rowid"], limit * 3),
    ).fetchall()

    candidates = {}  # memory_id -> {"vec_rank": int, "vec_distance": float}
    for rank, row in enumerate(vec_results):
        if row["distance"] == 0.0:
            continue  # skip self
        mapped = db.execute(
            "SELECT memory_id FROM memory_rowid_map WHERE rowid = ?",
            (row["rowid"],),
        ).fetchone()
        if mapped and mapped["memory_id"] != memory_id:
            candidates[mapped["memory_id"]] = {
                "vec_rank": rank,
                "vec_distance": row["distance"],
            }

    # --- FTS search ---
    fts_query = sanitize_fts_query(content[:200])
    try:
        fts_results = db.execute(
            "SELECT rowid, bm25(memory_fts, 5.0, 3.0) as rank FROM memory_fts "
            "WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, limit * 3),
        ).fetchall()
        for rank, row in enumerate(fts_results):
            mapped = db.execute(
                "SELECT memory_id FROM memory_rowid_map WHERE rowid = ?",
                (row["rowid"],),
            ).fetchone()
            if mapped and mapped["memory_id"] != memory_id:
                if mapped["memory_id"] not in candidates:
                    candidates[mapped["memory_id"]] = {}
                candidates[mapped["memory_id"]]["fts_rank"] = rank
    except sqlite3.OperationalError:
        pass  # FTS syntax error — fall back to vector-only

    # --- RRF fusion ---
    scored = []
    for mid, ranks in candidates.items():
        score = 0.0
        if "vec_rank" in ranks:
            score += RRF_VEC_WEIGHT / (RRF_K + ranks["vec_rank"] + 1)
        if "fts_rank" in ranks:
            score += (1.0 - RRF_VEC_WEIGHT) / (RRF_K + ranks["fts_rank"] + 1)
        scored.append((mid, score, ranks.get("vec_distance")))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Fetch full memory rows (active only)
    results = []
    for mid, score, vec_dist in scored[:limit]:
        mem = db.execute(
            "SELECT * FROM memories WHERE id = ? AND status = 'active'",
            (mid,),
        ).fetchone()
        if mem:
            results.append({
                "row": mem,
                "rrf_score": score,
                "vec_distance": vec_dist,
            })

    return results


def _check_fast_path(
    db: sqlite3.Connection, memory_row: sqlite3.Row, related_list: list[dict]
) -> Optional[str]:
    """Check if a memory can skip full LLM pipeline via schema acceleration.

    Returns the summary memory ID if fast-path applies, None otherwise.
    Requires a summary within cosine distance 0.25 and no contradiction signals.
    """
    for rel in related_list:
        row = rel["row"]
        dist = rel.get("vec_distance")
        if (
            row["layer"] == "summary"
            and dist is not None
            and dist < 0.25
        ):
            # Check for contradiction signals in the new memory
            content_lower = memory_row["content"].lower()
            neg_signals = [
                "not ", "no longer", "don't", "doesn't", "never",
                "wrong", "incorrect", "actually", "instead",
            ]
            if not any(sig in content_lower for sig in neg_signals):
                return row["id"]
    return None


def _create_edge(
    db: sqlite3.Connection,
    source_id: str,
    target_id: str,
    linking_context: str = "",
    linking_embedding: list[float] | None = None,
    flags: list[str] | None = None,
    created_by: str = "sleep",
    features: dict | None = None,
    # Legacy params (still accepted, mapped to new schema)
    edge_type: str = "",
    note: str = "",
) -> str:
    """Create a relationship edge between two memories.

    Dedup logic:
    - Structural edges (any flag): dedup on (pair, flag) — at most one per flag per pair.
    - Context edges (no flags): dedup on (pair, exact linking_context match).
    Returns edge ID, or empty string if duplicate.
    """
    # Legacy mapping: edge_type → flags + linking_context
    _LEGACY_FLAG_MAP = {
        "derived_from": ["derivation"],
        "hard_contradiction": ["contradiction"],
        "soft_contradiction": ["contradiction"],
        "evolved_from": ["revision"],
        "temporal_evolution": ["revision"],
        "supports": [],
        "related": [],
        "contextual": [],
    }
    if edge_type and flags is None:
        flags = _LEGACY_FLAG_MAP.get(edge_type, [])
    if note and not linking_context:
        linking_context = note

    resolved_flags = flags or []
    flags_json = json.dumps(resolved_flags)
    features_json = json.dumps(features) if features else "{}"

    # Dedup check
    if resolved_flags:
        # Structural edge: dedup on (pair, flag) for each flag
        for flag in resolved_flags:
            like_pattern = f'%"{flag}"%'
            existing = db.execute(
                "SELECT id FROM memory_edges "
                "WHERE flags LIKE ? AND "
                "((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?))",
                (like_pattern, source_id, target_id, target_id, source_id),
            ).fetchone()
            if existing:
                return ""
    else:
        # Context edge: dedup on (pair, exact linking_context)
        if linking_context:
            existing = db.execute(
                "SELECT id FROM memory_edges "
                "WHERE linking_context = ? AND "
                "((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?))",
                (linking_context, source_id, target_id, target_id, source_id),
            ).fetchone()
        else:
            # No flags, no context — dedup on (pair, edge_type) for legacy compat
            existing = db.execute(
                "SELECT id FROM memory_edges "
                "WHERE edge_type = ? AND "
                "((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?))",
                (edge_type or "", source_id, target_id, target_id, source_id),
            ).fetchone()
        if existing:
            return ""

    # Serialize linking_embedding if provided
    linking_emb_blob = serialize_f32(linking_embedding) if linking_embedding else None

    edge_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO memory_edges "
        "(id, source_id, target_id, edge_type, note, created_at, created_by, "
        " linking_context, linking_embedding, flags, features, weight) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (edge_id, source_id, target_id, edge_type, note, now, created_by,
         linking_context, linking_emb_blob, flags_json, features_json, 0.5),
    )
    return edge_id


def _handle_temporal_evolution(
    db: sqlite3.Connection,
    existing_row: sqlite3.Row,
    new_row: sqlite3.Row,
    note: str,
) -> dict:
    """Handle temporal evolution: old fact becomes time-bounded, not deleted.

    Sets valid_until on the older memory, creates an evolved_from edge.
    Returns a dict describing the changes made.
    """
    if not existing_row["valid_until"]:
        db.execute(
            "UPDATE memories SET valid_until = ? WHERE id = ?",
            (new_row["created_at"], existing_row["id"]),
        )
    _create_edge(db, new_row["id"], existing_row["id"],
                 edge_type="evolved_from", note=note, flags=["revision"], created_by="sleep")

    return {
        "type": "temporal_evolution",
        "old_id": existing_row["id"],
        "new_id": new_row["id"],
        "note": note,
    }


def _compute_shadow_load(
    db: sqlite3.Connection, memory_id: str, related: list[dict]
) -> float:
    """How much newer similar memories suppress this one. Returns float 0-1.

    Shadow potential: phi(newer|older) = max(0, (cos_sim - threshold) / (delta - epsilon))
    Shadow load: Phi = max shadow potential across all neighbors newer than this memory.
    """
    mem = db.execute(
        "SELECT created_at FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    if not mem:
        return 0.0

    max_shadow = 0.0
    DELTA = 0.3      # cosine similarity must be > (1 - DELTA) = 0.7 to shadow
    EPSILON = 0.05   # sharpness of the transition curve

    for rel in related:
        row, vec_dist = rel["row"], rel.get("vec_distance")
        if vec_dist is None or row["created_at"] <= mem["created_at"]:
            continue  # only newer memories can shadow older ones
        cos_sim = 1.0 - vec_dist
        if cos_sim < (1 - DELTA):
            continue
        phi = min(1.0, max(0.0, (cos_sim - (1 - DELTA)) / (DELTA - EPSILON)))
        max_shadow = max(max_shadow, phi)

    return max_shadow
