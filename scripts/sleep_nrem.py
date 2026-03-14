# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6", "openai>=2.0.0", "tiktoken>=0.7.0", "mcp[cli]>=1.2.0"]
# ///
"""
Sleep NREM pipeline — standalone script for memory consolidation.

Processes recent memories to detect relationships, classify contradictions,
handle temporal evolution, and produce a diagnostic report.

Usage: uv run scripts/sleep_nrem.py [standard|deep]
"""

import json
import os
import random
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Import helpers from memory package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from memory.constants import DATA_DIR
from memory import (
    get_db,
    count_tokens,
    embed_text,
    embed_batch,
    serialize_f32,
    _find_related_memories,
    _check_fast_path,
    _create_edge,
    _handle_temporal_evolution,
    _source_confidence_modifier,
    _compute_shadow_load,
    MAX_EDGES_PER_CYCLE,
    MAX_PRIORITY_BOOST_PER_CYCLE,
)

# Logging — timestamped log file + live progress log for tail -f
LOG_DIR = DATA_DIR / "sleep_logs"
PROGRESS_LOG = DATA_DIR / "sleep_progress.log"
_log_file = None  # set in _init_log()


def _init_log(mode: str):
    """Create a timestamped log file for this run. Returns the path."""
    global _log_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = LOG_DIR / f"nrem_{mode}_{ts}.log"
    # Clear the live progress log for tail -f compatibility
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG.write_text("", encoding="utf-8")
    return _log_file


def log(msg: str):
    """Print to stdout and append to both the run log and the live progress log."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(msg, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH_TOKEN_LIMIT = 40_000  # Max tokens per classification batch

CONTRADICTION_SYSTEM = """\
You are a memory relationship classifier for a personal knowledge system.
Given two memories, classify their relationship into exactly one of these types:

- none: No meaningful tension or connection worth tracking
- duplicate: Same information in different words — redundant, one should be removed
- supports: The new memory reinforces or provides evidence for the existing one
- hard_contradiction: Direct logical conflict — both cannot be true simultaneously
- soft_contradiction: Partial disagreement — mostly compatible but with a point of tension
- contextual: Both are true in different contexts (e.g., different projects, time periods, domains)
- temporal_evolution: A belief, preference, or fact that has changed over time — not an error, but growth
- related: Meaningfully connected but none of the above types fit precisely

Be conservative:
- Most apparent conflicts are temporal evolution or contextual differences, not hard contradictions
- Only use hard_contradiction when the memories genuinely cannot both be true in any context
- Prefer "none" over "related" if the connection is superficial
- "supports" requires genuine evidential reinforcement, not just topical overlap
- "duplicate" means truly redundant — if one adds meaningful detail the other lacks, use "supports" instead

Downstream effects: duplicate deletes the lower-priority memory, hard/soft contradiction penalizes
the older memory's confidence, temporal_evolution marks it time-bounded, supports boosts both.
Contextual and none/related create no penalties or boosts. Confidence scores below 0.3 (after
source-quality adjustment) are discarded — calibrate accordingly.

Respond with a JSON object for each pair. No markdown, no commentary — JSON only."""

CLASSIFY_TEMPLATE = """\
EXISTING MEMORY (stored {target_date}, source: {target_source}):
[{target_category}] {target_content}
Themes: {target_themes}

NEW MEMORY (stored {source_date}, source: {source_source}):
[{source_category}] {source_content}
Themes: {source_themes}"""

CLASSIFY_TEMPLATE_DEEP = """\
MEMORY A (stored {target_date}, source: {target_source}):
[{target_category}] {target_content}
Themes: {target_themes}

MEMORY B (stored {source_date}, source: {source_source}):
[{source_category}] {source_content}
Themes: {source_themes}"""

# Empty MCP config — no servers needed for pure classification
EMPTY_MCP_CONFIG = json.dumps({"mcpServers": {}})

# Isolated workspace for claude -p subprocesses. A fake git repo boundary
# prevents CLAUDE.md discovery via directory traversal, and --setting-sources
# "local" excludes user/project settings (hooks, memory instructions, etc.).
_SUBPROCESS_CWD = DATA_DIR / "subprocess_workspace"


def _ensure_subprocess_workspace():
    """Create an isolated workspace with a git boundary to block CLAUDE.md loading."""
    git_dir = _SUBPROCESS_CWD / ".git"
    head_file = git_dir / "HEAD"
    if not head_file.exists():
        git_dir.mkdir(parents=True, exist_ok=True)
        head_file.write_text("ref: refs/heads/main\n")


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------

def gather(db, mode: str, limit: int = 0) -> dict:
    """Gather memories needing sleep processing and their related memories."""
    last_sleep_row = db.execute(
        "SELECT completed_at FROM sleep_log ORDER BY completed_at DESC LIMIT 1"
    ).fetchone()
    last_sleep = last_sleep_row["completed_at"] if last_sleep_row else None

    if mode == "deep":
        memories = db.execute(
            "SELECT * FROM memories WHERE status = 'active' ORDER BY created_at DESC"
        ).fetchall()
    else:
        memories = db.execute(
            "SELECT * FROM memories WHERE status = 'active' "
            "AND last_sleep_processed IS NULL "
            "ORDER BY created_at DESC",
        ).fetchall()

    if limit > 0 and len(memories) > limit:
        if mode == "deep":
            memories = random.sample(memories, limit)
        else:
            memories = memories[:limit]

    if not memories:
        return {"status": "nothing_to_process", "last_sleep": last_sleep}

    fast_path_pairs = []
    classify_pairs = []
    processed_ids = []
    shadow_updates = {}  # memory_id -> shadow_load value
    pair_index = 0

    skipped_stable = 0
    for mem in memories:
        processed_ids.append(mem["id"])
        related = _find_related_memories(db, mem["id"], mem["content"], limit=5)

        if not related:
            continue

        # Skip-if-stable (deep mode): skip memories whose neighborhood hasn't
        # changed since they were last processed. A memory is stable if all its
        # neighbors were created before its last_sleep_processed timestamp.
        if mode == "deep" and mem["last_sleep_processed"]:
            newest_neighbor = max(r["row"]["created_at"] for r in related)
            if newest_neighbor < mem["last_sleep_processed"]:
                skipped_stable += 1
                continue

        # New memories cast shadows on older similar ones — update *their* shadow load
        for rel in related:
            rel_row = rel["row"]
            if rel_row["created_at"] >= mem["created_at"]:
                continue  # only older memories get shadowed
            rel_id = rel_row["id"]
            # Find neighbors of the older memory to compute its total shadow load
            rel_neighbors = _find_related_memories(db, rel_id, rel_row["content"], limit=5)
            shadow = _compute_shadow_load(db, rel_id, rel_neighbors)
            if shadow > 0.0:
                shadow_updates[rel_id] = max(shadow_updates.get(rel_id, 0.0), shadow)

        fast_target = _check_fast_path(db, mem, related)
        if fast_target:
            fast_path_pairs.append({
                "source_id": mem["id"],
                "target_id": fast_target,
                "note": "Schema-consistent (fast path)",
            })
            continue

        themes = json.loads(mem["themes"]) if mem["themes"] else []
        for rel in related[:3]:
            rel_row = rel["row"]

            # Pre-filter: skip weakly related pairs (unlikely to produce actionable edges)
            vec_dist = rel.get("vec_distance")
            if vec_dist is not None and vec_dist > 0.5:
                continue

            # Pre-filter: skip pairs that already have an edge between them
            existing_edge = db.execute(
                "SELECT 1 FROM memory_edges WHERE "
                "(source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?) "
                "LIMIT 1",
                (mem["id"], rel_row["id"], rel_row["id"], mem["id"]),
            ).fetchone()
            if existing_edge:
                continue

            rel_themes = json.loads(rel_row["themes"]) if rel_row["themes"] else []
            classify_pairs.append({
                "pair_index": pair_index,
                "source_id": mem["id"],
                "source_content": mem["content"],
                "source_category": mem["category"],
                "source_themes": ", ".join(themes),
                "source_date": mem["created_at"][:10],
                "source_source": mem["source"] or "session",
                "target_id": rel_row["id"],
                "target_content": rel_row["content"],
                "target_category": rel_row["category"],
                "target_themes": ", ".join(rel_themes),
                "target_date": rel_row["created_at"][:10],
                "target_source": rel_row["source"] or "session",
            })
            pair_index += 1

    # Track which memory IDs have classify pairs (need successful edges to be stamped)
    classify_memory_ids = set()
    for p in classify_pairs:
        classify_memory_ids.add(p["source_id"])
        classify_memory_ids.add(p["target_id"])

    return {
        "status": "ready",
        "last_sleep": last_sleep,
        "memories_count": len(memories),
        "processed_ids": processed_ids,
        "fast_path_pairs": fast_path_pairs,
        "classify_pairs": classify_pairs,
        "shadow_updates": shadow_updates,
        "classify_memory_ids": classify_memory_ids,
        "skipped_stable": skipped_stable,
    }


# ---------------------------------------------------------------------------
# Classify via claude -p
# ---------------------------------------------------------------------------

def format_pair(pair: dict, mode: str = "standard") -> str:
    """Format a single pair using the appropriate template for the mode."""
    template = CLASSIFY_TEMPLATE_DEEP if mode == "deep" else CLASSIFY_TEMPLATE
    return template.format(
        target_date=pair["target_date"],
        target_source=pair["target_source"],
        target_category=pair["target_category"],
        target_content=pair["target_content"],
        target_themes=pair["target_themes"] or "(none)",
        source_date=pair["source_date"],
        source_source=pair["source_source"],
        source_category=pair["source_category"],
        source_content=pair["source_content"],
        source_themes=pair["source_themes"] or "(none)",
    )


def build_batches(pairs: list[dict], mode: str = "standard") -> list[list[dict]]:
    """Split pairs into token-bounded batches (up to BATCH_TOKEN_LIMIT each)."""
    batches = []
    current_batch = []
    system_tokens = count_tokens(CONTRADICTION_SYSTEM)
    overhead = count_tokens(
        "Classify each of the following memory pairs. "
        "Return a JSON array with one object per pair.\n\n"
        "Return ONLY a JSON array: "
        '[{"pair_index": N, "edge_type": "...", "confidence": 0.0-1.0, '
        '"note": "brief explanation"}, ...]'
    )
    current_tokens = system_tokens + overhead

    for pair in pairs:
        pair_text = f"Pair {pair['pair_index']}:\n{format_pair(pair, mode)}\n"
        pair_tokens = count_tokens(pair_text)

        if current_batch and (current_tokens + pair_tokens) > BATCH_TOKEN_LIMIT:
            batches.append(current_batch)
            current_batch = []
            current_tokens = system_tokens + overhead

        current_batch.append(pair)
        current_tokens += pair_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def parse_json_response(response: str) -> list:
    """Try multiple strategies to extract a JSON array from the response."""
    text = response.strip()

    # Direct parse
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                result = json.loads(cleaned)
                return result if isinstance(result, list) else [result]
            except json.JSONDecodeError:
                continue

    # Bracket matching — find outermost [ ... ]
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            return result if isinstance(result, list) else [result]
        except json.JSONDecodeError:
            pass

    return []


def classify_batch(batch: list[dict], mode: str = "standard") -> list[dict]:
    """Send a batch to claude -p --model sonnet and parse the JSON response."""
    lines = [
        CONTRADICTION_SYSTEM,
        "",
        "Classify each of the following memory pairs. "
        "Return a JSON array with one object per pair.",
        "",
    ]
    for pair in batch:
        lines.append(f"Pair {pair['pair_index']}:")
        lines.append(format_pair(pair, mode))
        lines.append("")

    lines.append(
        "Return ONLY a JSON array: "
        '[{"pair_index": N, "edge_type": "...", "confidence": 0.0-1.0, '
        '"note": "brief explanation"}, ...]'
    )
    prompt_text = "\n".join(lines)

    # Build subprocess env — clear CLAUDECODE so we can launch claude
    _ensure_subprocess_workspace()
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)

    # Windows-specific subprocess flags
    kwargs = {}
    if sys.platform == "win32":
        CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = CREATE_NO_WINDOW

    claude_bin = shutil.which("claude") or "claude"
    try:
        result = subprocess.run(
            [
                claude_bin, "-p",
                "--model", "sonnet",
                "--no-session-persistence",
                "--strict-mcp-config",
                "--mcp-config", EMPTY_MCP_CONFIG,
                "--system-prompt", (
                    "You are a JSON-only output tool. Respond with the requested "
                    "JSON. No markdown fences, no commentary, no tool calls."
                ),
                "--tools", "",
                "--disable-slash-commands",
                "--setting-sources", "local",
            ],
            input=prompt_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
            cwd=str(_SUBPROCESS_CWD),
            env=env,
            **kwargs,
        )
    except subprocess.TimeoutExpired:
        print("  WARNING: classification timed out (180s)", file=sys.stderr)
        return []
    except FileNotFoundError:
        print("  ERROR: 'claude' CLI not found on PATH", file=sys.stderr)
        return []

    if result.returncode != 0:
        stderr_preview = (result.stderr or "")[:200]
        print(f"  WARNING: claude -p returned {result.returncode}: {stderr_preview}", file=sys.stderr)
        return []

    raw = result.stdout
    classifications = parse_json_response(raw)
    if not classifications:
        preview = raw[:500] if raw else "(empty)"
        print(f"  WARNING: failed to parse JSON from Sonnet response: {preview}",
              file=sys.stderr)

    return classifications


# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------

def write_results(db, results: list[dict], processed_ids: list[str],
                   shadow_updates: dict = None,
                   classify_memory_ids: set = None) -> dict:
    """Write classification results as edges and handle side effects."""
    now = datetime.now(timezone.utc).isoformat()

    edges_created = 0
    priority_boosts = 0
    contradictions_flagged = 0
    temporal_evolutions = 0
    duplicates_merged = 0
    merge_details = []
    per_memory_changes = {}

    valid_edge_types = {
        "none", "duplicate", "supports", "hard_contradiction", "soft_contradiction",
        "contextual", "temporal_evolution", "related", "evolved_from",
    }

    # Track memories deleted mid-cycle so we skip stale pairs
    deleted_this_cycle = set()

    # Track memories that participated in successful edge creation or merge
    successfully_processed = set()

    # Diagnostic counters
    _skip_no_ids = 0
    _skip_none = 0
    _skip_deleted = 0
    _skip_not_active = 0
    _skip_low_conf = 0
    _skip_edge_exists = 0
    _skip_cap = 0

    for result in results:
        edge_type = result.get("edge_type", "none")
        if edge_type not in valid_edge_types:
            edge_type = "related"

        source_id = result.get("source_id", "")
        target_id = result.get("target_id", "")
        confidence = float(result.get("confidence", 0.5))
        note = result.get("note", "")

        if not source_id or not target_id:
            _skip_no_ids += 1
            continue
        if edge_type in ("none", "related"):
            _skip_none += 1
            continue
        if source_id in deleted_this_cycle or target_id in deleted_this_cycle:
            _skip_deleted += 1
            continue

        source_mem = db.execute(
            "SELECT * FROM memories WHERE id = ? AND status = 'active'", (source_id,)
        ).fetchone()
        target_mem = db.execute(
            "SELECT * FROM memories WHERE id = ? AND status = 'active'", (target_id,)
        ).fetchone()
        if not source_mem or not target_mem:
            _skip_not_active += 1
            continue

        # Source quality weighting
        source_modifier = _source_confidence_modifier(source_mem["source"])
        weighted_confidence = confidence * source_modifier
        if weighted_confidence < 0.3:
            _skip_low_conf += 1
            continue

        # Duplicate merge — supersede the weaker/older one
        if edge_type == "duplicate":
            # Pick winner: higher priority wins; if tied, newer wins
            if source_mem["base_priority"] > target_mem["base_priority"]:
                winner, loser = source_mem, target_mem
            elif source_mem["base_priority"] < target_mem["base_priority"]:
                winner, loser = target_mem, source_mem
            elif source_mem["created_at"] >= target_mem["created_at"]:
                winner, loser = source_mem, target_mem
            else:
                winner, loser = target_mem, source_mem

            db.execute(
                "UPDATE memories SET superseded_by = ?, status = 'deleted' WHERE id = ?",
                (winner["id"], loser["id"]),
            )
            # Clean up vector, FTS, and rowid mapping for merged (deleted) memory
            vec_map = db.execute(
                "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (loser["id"],)
            ).fetchone()
            if vec_map:
                db.execute("DELETE FROM memory_vec WHERE rowid = ?", (vec_map["rowid"],))
                db.execute("DELETE FROM memory_fts WHERE rowid = ?", (vec_map["rowid"],))
                db.execute("DELETE FROM memory_rowid_map WHERE rowid = ?", (vec_map["rowid"],))
            deleted_this_cycle.add(loser["id"])
            successfully_processed.add(winner["id"])
            successfully_processed.add(loser["id"])
            duplicates_merged += 1
            loser_label = loser["summary"] or loser["content"][:60]
            winner_label = winner["summary"] or winner["content"][:60]
            merge_details.append(f"  Merged: '{loser_label}' into '{winner_label}'")
            continue

        for mid in (source_id, target_id):
            if mid not in per_memory_changes:
                per_memory_changes[mid] = {"priority_boost": 0, "edges_added": 0}

        target_changes = per_memory_changes[target_id]
        source_changes = per_memory_changes[source_id]

        if target_changes["edges_added"] >= MAX_EDGES_PER_CYCLE:
            _skip_cap += 1
            continue
        if source_changes["edges_added"] >= MAX_EDGES_PER_CYCLE:
            _skip_cap += 1
            continue

        # Temporal evolution
        if edge_type == "temporal_evolution":
            if source_mem["created_at"] <= target_mem["created_at"]:
                existing, newer = source_mem, target_mem
            else:
                existing, newer = target_mem, source_mem
            _handle_temporal_evolution(db, existing, newer, note)
            temporal_evolutions += 1
            target_changes["edges_added"] += 1
            source_changes["edges_added"] += 1
            edges_created += 1
            successfully_processed.add(source_id)
            successfully_processed.add(target_id)
            continue

        # Map classification to flags
        edge_flags = []
        if edge_type in ("hard_contradiction", "soft_contradiction"):
            edge_flags = ["contradiction"]
        elif edge_type in ("temporal_evolution", "evolved_from"):
            edge_flags = ["revision"]

        # Create edge with new schema fields
        edge_id = _create_edge(
            db, source_id, target_id,
            linking_context=note,
            flags=edge_flags if edge_flags else None,
            created_by="sleep",
            edge_type=edge_type,  # legacy compat
            note=note,            # legacy compat
        )
        if not edge_id:
            _skip_edge_exists += 1
        if edge_id:
            edges_created += 1
            target_changes["edges_added"] += 1
            source_changes["edges_added"] += 1
            successfully_processed.add(source_id)
            successfully_processed.add(target_id)

            # Priority boost for "supports"
            if (edge_type == "supports"
                    and target_changes["priority_boost"] < MAX_PRIORITY_BOOST_PER_CYCLE):
                current_priority = target_mem["base_priority"]
                if current_priority < 10:
                    boost = min(1, MAX_PRIORITY_BOOST_PER_CYCLE - target_changes["priority_boost"])
                    new_priority = min(10, current_priority + boost)
                    if new_priority > current_priority:
                        db.execute(
                            "UPDATE memories SET base_priority = ? WHERE id = ?",
                            (new_priority, target_id),
                        )
                        target_changes["priority_boost"] += boost
                        priority_boosts += 1


            # Confidence boost on both memories for support edges
            if edge_type == "supports":
                for mid in (source_id, target_id):
                    db.execute(
                        """UPDATE memories SET confidence = MIN(0.95,
                           COALESCE(confidence, 0.5) + 0.03 * (1.0 - COALESCE(confidence, 0.5)))
                           WHERE id = ?""",
                        (mid,))

            if edge_type in ("hard_contradiction", "soft_contradiction"):
                contradictions_flagged += 1
                # Reduce confidence on the older memory
                penalty = 0.10 if edge_type == "hard_contradiction" else 0.05
                older_id = source_id if source_mem["created_at"] <= target_mem["created_at"] else target_id
                db.execute(
                    "UPDATE memories SET confidence = MAX(0.1, COALESCE(confidence, 0.5) - ?) WHERE id = ?",
                    (penalty, older_id))

    # Log skip diagnostics (helps debug 0-edge runs)
    skip_parts = []
    if _skip_no_ids: skip_parts.append(f"no_ids={_skip_no_ids}")
    if _skip_none: skip_parts.append(f"none={_skip_none}")
    if _skip_deleted: skip_parts.append(f"deleted={_skip_deleted}")
    if _skip_not_active: skip_parts.append(f"not_active={_skip_not_active}")
    if _skip_low_conf: skip_parts.append(f"low_conf={_skip_low_conf}")
    if _skip_edge_exists: skip_parts.append(f"edge_exists={_skip_edge_exists}")
    if _skip_cap: skip_parts.append(f"cap_hit={_skip_cap}")
    if skip_parts:
        log(f"  Skipped: {', '.join(skip_parts)}")

    # Write shadow-load updates
    shadow_count = 0
    if shadow_updates:
        for mid, shadow_val in shadow_updates.items():
            if mid not in deleted_this_cycle:
                db.execute(
                    "UPDATE memories SET shadow_load = ? WHERE id = ?",
                    (shadow_val, mid),
                )
                shadow_count += 1

    # Commit edges/shadows (releases write lock before network calls)
    db.commit()

    # Batch-embed linking contexts for edges that have context but no embedding
    edges_to_embed = db.execute(
        "SELECT id, linking_context FROM memory_edges "
        "WHERE linking_context IS NOT NULL AND linking_context != '' "
        "AND linking_embedding IS NULL"
    ).fetchall()
    embedded_count = 0
    if edges_to_embed:
        texts = [e["linking_context"] for e in edges_to_embed]
        try:
            embeddings = embed_batch(texts)
            for edge_row, emb in zip(edges_to_embed, embeddings):
                db.execute(
                    "UPDATE memory_edges SET linking_embedding = ? WHERE id = ?",
                    (serialize_f32(emb), edge_row["id"]),
                )
                embedded_count += 1
        except Exception as e:
            log(f"  WARNING: batch embed failed ({e}), falling back to per-edge")
            for edge_row in edges_to_embed:
                try:
                    emb = embed_text(edge_row["linking_context"])
                    db.execute(
                        "UPDATE memory_edges SET linking_embedding = ? WHERE id = ?",
                        (serialize_f32(emb), edge_row["id"]),
                    )
                    embedded_count += 1
                except Exception as ex:
                    log(f"  WARNING: failed to embed edge {edge_row['id'][:8]}: {ex}")
    if embedded_count:
        log(f"  Embedded {embedded_count} edge linking contexts")

    # Mark memories as sleep-processed:
    # - Memories not in classify_memory_ids had no pairs to classify — legitimately done
    # - Memories in classify_memory_ids are only stamped if they had successful edges
    # This prevents failed classifications from permanently skipping memories
    if classify_memory_ids is None:
        classify_memory_ids = set()
    for mid in processed_ids:
        if mid in classify_memory_ids:
            if mid in successfully_processed:
                db.execute(
                    "UPDATE memories SET last_sleep_processed = ? WHERE id = ?",
                    (now, mid),
                )
        else:
            # No pairs to classify — legitimately processed
            db.execute(
                "UPDATE memories SET last_sleep_processed = ? WHERE id = ?",
                (now, mid),
            )

    db.commit()

    return {
        "edges_created": edges_created,
        "priority_boosts": priority_boosts,
        "contradictions_flagged": contradictions_flagged,
        "temporal_evolutions": temporal_evolutions,
        "duplicates_merged": duplicates_merged,
        "merge_details": merge_details,
        "per_memory_changes": per_memory_changes,
        "shadow_updates": shadow_count,
    }


# ---------------------------------------------------------------------------
# Log cycle
# ---------------------------------------------------------------------------

def log_cycle(db, mode: str, report: str, stats: dict) -> str:
    """Write a sleep cycle to the sleep_log table. Returns the log ID."""
    log_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    db.execute(
        """INSERT INTO sleep_log
           (id, started_at, completed_at, mode, memories_processed,
            relationships_found, summaries_refreshed, contradictions_flagged,
            fast_path_count, full_pipeline_count, memories_pruned,
            per_memory_changes, report)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            log_id,
            stats.get("started_at", now),
            now,
            mode,
            stats.get("memories_processed", 0),
            stats.get("relationships_found", 0),
            stats.get("summaries_refreshed", 0),
            stats.get("contradictions_flagged", 0),
            stats.get("fast_path_count", 0),
            stats.get("full_pipeline_count", 0),
            stats.get("duplicates_merged", 0),
            json.dumps(stats.get("per_memory_changes", [])),
            report,
        ),
    )

    db.commit()
    return log_id


# ---------------------------------------------------------------------------
# Entity auto-linking
# ---------------------------------------------------------------------------

def link_entities(db, processed_ids: list[str]) -> dict:
    """Link processed memories to entity memories by name matching.

    Scans each processed memory's content and summary for entity names.
    Creates edges where none exist. Returns stats dict.
    """
    # Load all entity memories
    entities = db.execute(
        "SELECT id, content, summary, themes FROM memories "
        "WHERE category = 'entity' AND status = 'active'"
    ).fetchall()

    if not entities:
        return {"entities_found": 0, "edges_created": 0}

    # Extract searchable names from each entity's summary
    # Entity summaries follow "Entity: Name — description" format
    entity_names = {}  # entity_id -> list of search terms
    for ent in entities:
        names = []
        summary = ent["summary"] or ""
        # Parse "Entity: Name —" pattern
        if summary.startswith("Entity: "):
            name_part = summary[8:]
            dash_pos = name_part.find(" — ")
            if dash_pos > 0:
                name_part = name_part[:dash_pos].strip()
            names.append(name_part.lower())

        # Also use themes (excluding "entity" tag)
        themes = json.loads(ent["themes"]) if ent["themes"] else []
        for t in themes:
            if t != "entity" and len(t) > 2:
                names.append(t.lower().replace("-", " "))

        entity_names[ent["id"]] = names

    # Load processed memories
    if not processed_ids:
        return {"entities_found": len(entities), "edges_created": 0}

    placeholders = ",".join("?" for _ in processed_ids)
    memories = db.execute(
        f"SELECT id, content, summary, category FROM memories "
        f"WHERE id IN ({placeholders}) AND status = 'active' AND category != 'entity'",
        processed_ids,
    ).fetchall()

    edges_created = 0
    for mem in memories:
        searchable = (mem["content"] + " " + (mem["summary"] or "")).lower()
        for ent_id, names in entity_names.items():
            if ent_id == mem["id"]:
                continue
            if any(name in searchable for name in names):
                # Check if edge already exists
                existing = db.execute(
                    "SELECT 1 FROM memory_edges WHERE "
                    "(source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?) "
                    "LIMIT 1",
                    (ent_id, mem["id"], mem["id"], ent_id),
                ).fetchone()
                if not existing:
                    edge_id = _create_edge(
                        db, ent_id, mem["id"],
                        linking_context=f"Entity auto-link: memory mentions entity",
                        created_by="sleep",
                    )
                    if edge_id:
                        edges_created += 1

    db.commit()
    return {"entities_found": len(entities), "edges_created": edges_created}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def correct_decay_rates(db) -> dict:
    """Correct decay rates that conflict with observed feedback utility.

    Memories with high custom decay rates (>category default * 3) but good
    real feedback utility get reset to their category default. Memories with
    consistently low utility and many feedbacks get a faster rate.

    Returns stats dict for the report.
    """
    from memory.constants import CATEGORY_DECAY_RATES, DEFAULT_DECAY_RATE

    cat_defaults = dict(CATEGORY_DECAY_RATES)
    stats = {"slowed": 0, "accelerated": 0, "details": []}

    # Find candidates: active memories with real (non-probe) feedback
    rows = db.execute("""
        SELECT m.id, m.category, m.decay_rate, m.summary
        FROM memories m
        WHERE m.status = 'active' AND m.decay_rate > 0
    """).fetchall()

    for r in rows:
        mid = r["id"]
        cat = r["category"]
        current_rate = r["decay_rate"]
        default_rate = cat_defaults.get(cat, DEFAULT_DECAY_RATE)

        # Get real feedback (excluding probe)
        fbs = db.execute("""
            SELECT json_extract(context, '$.utility') as u
            FROM memory_events
            WHERE memory_id = ? AND event_type = 'feedback'
              AND (json_extract(context, '$.reason') IS NULL
                   OR json_extract(context, '$.reason') NOT LIKE '%probe%')
        """, (mid,)).fetchall()
        real_utils = [f["u"] for f in fbs if f["u"] is not None]

        if len(real_utils) < 2:
            # Not enough evidence — only fix if rate is extremely aggressive
            if current_rate > default_rate * 5 and len(real_utils) == 0:
                db.execute("UPDATE memories SET decay_rate = ? WHERE id = ?",
                           (default_rate, mid))
                stats["slowed"] += 1
                stats["details"].append(
                    f"  {mid[:8]}: {current_rate:.4f} -> {default_rate:.4f} (no evidence, was 5x+ default)")
            continue

        mean_u = sum(real_utils) / len(real_utils)

        # Case 1: Good utility but aggressive decay — slow down
        if mean_u >= 0.25 and current_rate > default_rate * 2:
            db.execute("UPDATE memories SET decay_rate = ? WHERE id = ?",
                       (default_rate, mid))
            stats["slowed"] += 1
            stats["details"].append(
                f"  {mid[:8]}: {current_rate:.4f} -> {default_rate:.4f} (util={mean_u:.2f}, n={len(real_utils)})")

        # Case 2: Low utility with strong evidence — speed up (but not above 2x default)
        elif mean_u < 0.1 and len(real_utils) >= 4 and current_rate < default_rate * 1.5:
            faster = min(default_rate * 2, 0.05)
            if faster > current_rate * 1.3:  # meaningful change
                db.execute("UPDATE memories SET decay_rate = ? WHERE id = ?",
                           (faster, mid))
                stats["accelerated"] += 1
                stats["details"].append(
                    f"  {mid[:8]}: {current_rate:.4f} -> {faster:.4f} (util={mean_u:.2f}, n={len(real_utils)})")

    if stats["slowed"] or stats["accelerated"]:
        db.commit()

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sleep NREM pipeline")
    parser.add_argument("mode", nargs="?", default="standard", choices=["standard", "deep"])
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap number of memories to gather (0 = no limit)")
    args = parser.parse_args()
    mode = args.mode
    mem_limit = args.limit

    started_at = datetime.now(timezone.utc).isoformat()
    log_path = _init_log(mode)
    log(f"## Sleep NREM ({mode})\n")
    log(f"Log file: {log_path}")

    # Step 1: Gather
    db = get_db()
    gather_data = gather(db, mode, limit=mem_limit)

    if gather_data["status"] == "nothing_to_process":
        last = gather_data.get("last_sleep")
        msg = "Nothing to process."
        if last:
            msg += f" Last sleep: {last[:10]}"
        log(msg)
        db.close()
        return

    memories_count = gather_data["memories_count"]
    fast_path_pairs = gather_data["fast_path_pairs"]
    classify_pairs = gather_data["classify_pairs"]
    processed_ids = gather_data["processed_ids"]
    shadow_updates = gather_data.get("shadow_updates", {})
    classify_memory_ids = gather_data.get("classify_memory_ids", set())

    skipped_stable = gather_data.get("skipped_stable", 0)
    log(f"Gathered {memories_count} memories, "
        f"{len(classify_pairs)} pairs to classify, "
        f"{len(fast_path_pairs)} fast-path"
        + (f", {skipped_stable} skipped (stable)" if skipped_stable else ""))

    # Step 2: Fast-path results
    all_results = []
    for p in fast_path_pairs:
        all_results.append({
            "source_id": p["source_id"],
            "target_id": p["target_id"],
            "edge_type": "supports",
            "confidence": 0.9,
            "note": p["note"],
        })

    # Step 3: Classify via claude -p (parallel batches)
    if classify_pairs:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        batches = build_batches(classify_pairs, mode)
        log(f"Classification: {len(classify_pairs)} pairs in {len(batches)} batch(es)")

        # Each claude -p subprocess reserves ~13 GB virtual memory (Bun/JSC heap).
        # Parallel launches exhaust the Windows paging file and crash (0xC0000409).
        MAX_PARALLEL = 1
        from math import ceil, sqrt
        waves = ceil(len(batches) / MAX_PARALLEL)
        est_min = waves * 3.0
        est_err = sqrt(waves) * 1.0
        log(f"  Estimated time: {est_min:.0f} min (+/- {est_err:.0f} min), {waves} waves at ~3.0 min/wave")

        def _classify_one(batch_idx: int, batch: list[dict]) -> tuple[int, list[dict], list[dict]]:
            """Classify a single batch. Returns (batch_idx, batch, classifications)."""
            classifications = classify_batch(batch, mode)
            return batch_idx, batch, classifications

        import time as _time
        _t_classify_start = _time.monotonic()
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
            futures = {
                executor.submit(_classify_one, i, batch): i
                for i, batch in enumerate(batches)
            }
            log(f"  Launched {len(futures)} batch(es), up to {MAX_PARALLEL} in parallel")

            for future in as_completed(futures):
                batch_idx, batch, classifications = future.result()
                actionable = sum(1 for c in classifications if c.get("edge_type", "none") != "none")
                log(f"  Batch {batch_idx + 1}/{len(batches)}: {len(batch)} pairs -> {len(classifications)} results ({actionable} actionable)")

                # Map pair_index back to source/target IDs
                pair_lookup = {p["pair_index"]: p for p in batch}
                for cls in classifications:
                    pair_idx = cls.get("pair_index")
                    # Coerce to int — Sonnet often returns pair_index as a string
                    if pair_idx is not None:
                        try:
                            pair_idx = int(pair_idx)
                        except (ValueError, TypeError):
                            pass
                    if pair_idx is not None and pair_idx in pair_lookup:
                        p = pair_lookup[pair_idx]
                        cls["source_id"] = p["source_id"]
                        cls["target_id"] = p["target_id"]
                        all_results.append(cls)
                    else:
                        print(f"    WARNING: unmapped pair_index={cls.get('pair_index')!r} "
                              f"(lookup keys: {list(pair_lookup.keys())})",
                              file=sys.stderr)

        _t_classify_elapsed = _time.monotonic() - _t_classify_start
        avg_per_batch = _t_classify_elapsed / len(batches)
        log(f"  Classification done: {_t_classify_elapsed:.0f}s total, {avg_per_batch:.0f}s avg/batch ({avg_per_batch/60:.1f} min/batch)")

    # Step 4: Write results
    write_stats = write_results(db, all_results, processed_ids, shadow_updates,
                                classify_memory_ids=classify_memory_ids)

    # Step 4b: Decay rate correction
    decay_stats = correct_decay_rates(db)
    if decay_stats["slowed"] or decay_stats["accelerated"]:
        log(f"Decay correction: {decay_stats['slowed']} slowed, {decay_stats['accelerated']} accelerated")
        for detail in decay_stats["details"]:
            log(detail)

    # Step 4c: Entity auto-linking
    entity_stats = link_entities(db, processed_ids)
    if entity_stats["edges_created"]:
        log(f"Entity linking: {entity_stats['edges_created']} new edges to {entity_stats['entities_found']} entities")

    # Step 5: Build report
    report_lines = [
        f"## Sleep Report ({mode})",
        "",
        f"- Memories processed: {memories_count}",
        f"- Relationships found: {write_stats['edges_created']}",
        f"- Duplicates merged: {write_stats['duplicates_merged']}",
        f"- Fast-path pairs: {len(fast_path_pairs)}",
        f"- Priority boosts: {write_stats['priority_boosts']}",
        f"- Contradictions flagged: {write_stats['contradictions_flagged']}",
        f"- Temporal evolutions: {write_stats['temporal_evolutions']}",
        f"- Shadow-load updates: {write_stats['shadow_updates']}",
        f"- Decay corrections: {decay_stats['slowed']} slowed, {decay_stats['accelerated']} accelerated",
        f"- Entity links: {entity_stats['edges_created']} new edges to {entity_stats['entities_found']} entities",
    ]
    if write_stats["merge_details"]:
        report_lines.append("")
        report_lines.append("### Merges")
        report_lines.extend(write_stats["merge_details"])
    report = "\n".join(report_lines)

    # Step 6: Log cycle
    log_stats = {
        "started_at": started_at,
        "memories_processed": memories_count,
        "relationships_found": write_stats["edges_created"],
        "contradictions_flagged": write_stats["contradictions_flagged"],
        "duplicates_merged": write_stats["duplicates_merged"],
        "fast_path_count": len(fast_path_pairs),
        "full_pipeline_count": len(classify_pairs),
        "per_memory_changes": [
            {"id": mid, **changes}
            for mid, changes in write_stats["per_memory_changes"].items()
        ],
    }
    log_id = log_cycle(db, mode, report, log_stats)

    db.close()

    # Print report
    log("")
    log(report)
    log(f"\nLog ID: {log_id[:8]}")


if __name__ == "__main__":
    main()
