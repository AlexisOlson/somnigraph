# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6", "openai>=2.0.0", "tiktoken>=0.7.0", "mcp[cli]>=1.2.0"]
# ///
"""
Sleep REM pipeline — higher-order memory consolidation.

Processes active detail-layer memories to generate/refresh thin summary-layer
memories (dense ~150-200 token paragraphs optimized for vocabulary bridging),
manage dormancy transitions, and produce health diagnostics.

Usage:
    uv run scripts/sleep_rem.py
    uv run scripts/sleep_rem.py --limit 3
"""

import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Import helpers from memory modules (add src dir to path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from memory.constants import DATA_DIR
from memory import (
    get_db,
    count_tokens,
    embed_text,
    build_enriched_text,
    serialize_f32,
    _create_edge,
    _insert_memory,
    _compute_shadow_load,
    _row_get,
    update_fts,
    delete_fts,
)

# Logging — each run gets a persistent timestamped log file + a live "latest" symlink
LOG_DIR = DATA_DIR / "sleep_logs"
PROGRESS_LOG = DATA_DIR / "sleep_progress.log"  # compat: live tail target
_log_file = None  # set in main()


def _init_log():
    """Create a timestamped log file for this run. Returns the path."""
    global _log_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = LOG_DIR / f"rem_{ts}.log"
    # Also clear the live progress log for tail -f compatibility
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG.write_text("", encoding="utf-8")
    return _log_file


def log(msg: str):
    """Print to stdout + append to both the run log and the live progress log."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"), flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def log_detail(msg: str):
    """Write to run log only (not stdout). For verbose data like full prompts/responses."""
    ts = datetime.now().strftime("%H:%M:%S")
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

TAXONOMY_PATH = DATA_DIR / "taxonomy.json"


def load_taxonomy() -> dict:
    """Load taxonomy from JSON file. Returns empty structure if missing."""
    if TAXONOMY_PATH.exists():
        try:
            return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log(f"  WARNING: failed to load taxonomy: {e}")
    return {"version": 1, "last_updated": None, "topics": []}


def save_taxonomy(taxonomy: dict):
    """Write taxonomy to JSON file."""
    taxonomy["last_updated"] = datetime.now(timezone.utc).isoformat()
    TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TAXONOMY_PATH.write_text(
        json.dumps(taxonomy, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

TAXONOMY_ASSIGN_SYSTEM = """You are organizing a personal knowledge system's memories into a topic taxonomy.

Each topic you create will become a single summary paragraph used for retrieval. This means:
- Topics that are too heterogeneous produce unfocused summaries that don't help retrieval.
- Topics with fewer than 3 memories rarely justify their own summary — prefer merging into
  a related topic unless the domain is truly distinct.
- The ~30-memory split threshold exists because larger clusters dilute summary quality.

You will receive:
1. The current taxonomy with descriptions
2. Example memories already assigned to each topic (for context)
3. Unassigned memories that need placement

Your job:
- Assign every unassigned memory to exactly one leaf topic
- If no existing topic fits, create a new one
- Propose taxonomy changes if needed: new topics, merges, splits, reparenting
- Leaf topics should stay under ~30 memories. If one is larger, consider splitting.
- A memory belongs to exactly one leaf topic — pick the primary one
- Topics form a tree: leaf topics may have a parent. Flat topics have parent: null.
- Topic IDs should be kebab-case slugs (e.g. "skyrim-detective", "chess-engines")
- Every unassigned memory MUST be assigned. No "uncategorized" bucket.

Respond with a JSON object:
{
  "taxonomy_changes": [
    {"action": "add", "topic": {"id": "...", "label": "...", "description": "...", "parent": null}},
    {"action": "rename", "id": "...", "label": "...", "description": "..."},
    {"action": "merge", "from_id": "...", "into_id": "..."},
    {"action": "split", "id": "...", "into": [{"id": "...", "label": "...", "description": "...", "parent": "..."}]},
    {"action": "reparent", "id": "...", "new_parent": "..."}
  ],
  "assignments": {
    "memory_id": "topic_id",
    ...
  }
}

Include ALL unassigned memory IDs in "assignments".
Only include taxonomy_changes that are actually needed. Empty array is fine if taxonomy is good."""


def assign_memories(db, taxonomy: dict) -> dict:
    """LLM-driven taxonomy assignment for unassigned memories.

    Sends the taxonomy + unassigned memories (with example assignments) to
    Sonnet in batches. Assigns unassigned memories and proposes taxonomy changes.
    Returns updated taxonomy.
    """
    TAXONOMY_BATCH_SIZE = 50  # memories per LLM call

    # Query all active detail memories
    rows = db.execute(
        "SELECT id, summary, content, themes, category, topic_id "
        "FROM memories WHERE status='active' AND layer='detail'"
    ).fetchall()

    if not rows:
        return taxonomy

    # Separate assigned vs unassigned
    assigned_rows = []
    unassigned_rows = []
    for row in rows:
        if _row_get(row, "topic_id"):
            assigned_rows.append(row)
        else:
            unassigned_rows.append(row)

    unassigned_count = len(unassigned_rows)

    # Build shared context (taxonomy + examples) — reused across batches
    shared_parts = [TAXONOMY_ASSIGN_SYSTEM, ""]

    if taxonomy["topics"]:
        shared_parts.append("## Current Taxonomy")
        for topic in taxonomy["topics"]:
            parent_str = f" (parent: {topic['parent']})" if topic.get("parent") else ""
            topic_count = sum(1 for r in assigned_rows if _row_get(r, "topic_id") == topic["id"])
            shared_parts.append(
                f"- **{topic['id']}**: {topic['label']}{parent_str} — "
                f"{topic['description']} ({topic_count} memories)"
            )
        shared_parts.append("")

    if assigned_rows:
        topic_examples = {}
        for row in assigned_rows:
            tid = _row_get(row, "topic_id")
            if tid:
                topic_examples.setdefault(tid, []).append(row)

        shared_parts.append("## Example Assignments (for context)")
        for tid, examples in topic_examples.items():
            shared_parts.append(f"### {tid}")
            for ex in examples[:5]:
                themes = json.loads(ex["themes"]) if ex["themes"] else []
                summary = ex["summary"] or ex["content"][:100]
                shared_parts.append(
                    f"  - {ex['id'][:8]}: [{ex['category']}] {summary} "
                    f"({', '.join(themes)})"
                )
        shared_parts.append("")

    shared_text = "\n".join(shared_parts)

    # Split unassigned into batches
    batches = [
        unassigned_rows[i:i + TAXONOMY_BATCH_SIZE]
        for i in range(0, len(unassigned_rows), TAXONOMY_BATCH_SIZE)
    ]

    log(f"  Taxonomy: {unassigned_count} unassigned / {len(rows)} total, "
        f"{len(batches)} batch(es) of up to {TAXONOMY_BATCH_SIZE}")

    topic_map = {t["id"]: t for t in taxonomy["topics"]}
    total_assigned = 0
    total_reassigned = 0

    for batch_idx, batch in enumerate(batches):
        # Build per-batch prompt
        batch_parts = [shared_text, "## Unassigned Memories"]
        for row in batch:
            themes = json.loads(row["themes"]) if row["themes"] else []
            summary = row["summary"] or row["content"][:100]
            batch_parts.append(
                f"- {row['id']}: [{row['category']}] {summary} "
                f"({', '.join(themes)})"
            )
        batch_parts.append(f"\n{len(batch)} memories need assignment.")
        batch_parts.append("\nRespond with the JSON object described above.")

        prompt_text = "\n".join(batch_parts)

        log(f"  Batch {batch_idx + 1}/{len(batches)}: {len(batch)} memories, "
            f"{count_tokens(prompt_text)} tokens")
        log_detail(f"--- TAXONOMY PROMPT (batch {batch_idx + 1}) ---")
        log_detail(prompt_text)
        log_detail(f"--- END TAXONOMY PROMPT (batch {batch_idx + 1}) ---")

        raw = call_llm(prompt_text, model="sonnet", timeout=300)
        if not raw:
            log(f"  ERROR: taxonomy batch {batch_idx + 1} returned empty")
            continue

        log_detail(f"--- TAXONOMY RESPONSE (batch {batch_idx + 1}) ---")
        log_detail(raw)
        log_detail(f"--- END TAXONOMY RESPONSE (batch {batch_idx + 1}) ---")

        result = parse_json_response(raw, expect_object=True)
        if not result or not isinstance(result, dict):
            log(f"  ERROR: failed to parse taxonomy batch {batch_idx + 1}")
            log_detail(f"  Raw response (first 2000 chars): {raw[:2000]}")
            continue

        # Apply taxonomy changes from every batch (guards are idempotent)
        changes = result.get("taxonomy_changes", [])

        for change in changes:
            action = change.get("action")
            if action == "add":
                new_topic = change.get("topic", {})
                tid = new_topic.get("id")
                if tid and tid not in topic_map:
                    topic_map[tid] = {
                        "id": tid,
                        "label": new_topic.get("label", tid),
                        "description": new_topic.get("description", ""),
                        "parent": new_topic.get("parent"),
                    }
                    log(f"  Taxonomy: added '{tid}'")

            elif action == "rename":
                tid = change.get("id")
                if tid in topic_map:
                    if "label" in change:
                        topic_map[tid]["label"] = change["label"]
                    if "description" in change:
                        topic_map[tid]["description"] = change["description"]
                    log(f"  Taxonomy: renamed '{tid}'")

            elif action == "merge":
                from_id = change.get("from_id")
                into_id = change.get("into_id")
                if from_id in topic_map and into_id in topic_map:
                    db.execute(
                        "UPDATE memories SET topic_id = ? WHERE topic_id = ?",
                        (into_id, from_id),
                    )
                    for t in topic_map.values():
                        if t.get("parent") == from_id:
                            t["parent"] = into_id
                    del topic_map[from_id]
                    log(f"  Taxonomy: merged '{from_id}' into '{into_id}'")

            elif action == "split":
                old_id = change.get("id")
                into_topics = change.get("into", [])
                if old_id in topic_map and into_topics:
                    for new_topic in into_topics:
                        ntid = new_topic.get("id")
                        if ntid and ntid not in topic_map:
                            topic_map[ntid] = {
                                "id": ntid,
                                "label": new_topic.get("label", ntid),
                                "description": new_topic.get("description", ""),
                                "parent": new_topic.get("parent", topic_map[old_id].get("parent")),
                            }
                    log(f"  Taxonomy: split '{old_id}' into {[t.get('id') for t in into_topics]}")

            elif action == "reparent":
                tid = change.get("id")
                new_parent = change.get("new_parent")
                if tid in topic_map:
                    topic_map[tid]["parent"] = new_parent
                    log(f"  Taxonomy: reparented '{tid}' under '{new_parent}'")

        taxonomy["topics"] = list(topic_map.values())

        # Apply memory assignments
        assignments = result.get("assignments", {})
        valid_topic_ids = set(topic_map.keys())

        for memory_id, topic_id in assignments.items():
            if topic_id not in valid_topic_ids:
                log(f"  WARNING: skipping assignment to unknown topic '{topic_id}' for memory {memory_id[:8]}")
                continue

            current = db.execute(
                "SELECT topic_id FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if not current:
                continue

            old_topic = _row_get(current, "topic_id")
            if old_topic == topic_id:
                continue

            db.execute(
                "UPDATE memories SET topic_id = ? WHERE id = ?",
                (topic_id, memory_id),
            )
            if old_topic:
                total_reassigned += 1
            else:
                total_assigned += 1

        db.commit()
        log(f"    -> {len(assignments)} assignments from batch {batch_idx + 1}")

    save_taxonomy(taxonomy)

    log(f"  Assigned: {total_assigned}, Reassigned: {total_reassigned}, "
        f"Topics: {len(taxonomy['topics'])}")

    return taxonomy


# ---------------------------------------------------------------------------
# LLM subprocess configuration
# ---------------------------------------------------------------------------

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


def call_llm(prompt_text: str, model: str = "sonnet", timeout: int = 300) -> str:
    """Shell out to claude -p --model <model> and return stdout.

    Returns empty string on failure (timeout, not found, non-zero exit).
    """
    _ensure_subprocess_workspace()

    env = dict(os.environ)
    env.pop("CLAUDECODE", None)

    kwargs = {}
    if sys.platform == "win32":
        CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = CREATE_NO_WINDOW

    claude_bin = shutil.which("claude") or "claude"
    try:
        result = subprocess.run(
            [
                claude_bin, "-p",
                "--model", model,
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
            timeout=timeout,
            cwd=str(_SUBPROCESS_CWD),
            env=env,
            **kwargs,
        )
    except subprocess.TimeoutExpired:
        log(f"  WARNING: {model} call timed out ({timeout}s)")
        return ""
    except FileNotFoundError:
        log("  ERROR: 'claude' CLI not found on PATH")
        return ""

    if result.returncode != 0:
        stderr_preview = (result.stderr or "")[:200]
        log(f"  WARNING: claude -p returned {result.returncode}: {stderr_preview}")
        return ""

    return result.stdout


# ---------------------------------------------------------------------------
# JSON response parsing (same strategy as NREM)
# ---------------------------------------------------------------------------

def parse_json_response(response: str, expect_object: bool = False):
    """Try multiple strategies to extract JSON from an LLM response.

    If expect_object is True, look for {} rather than [].
    Returns parsed JSON (list or dict) or None on failure.
    """
    text = response.strip()

    # Direct parse
    try:
        result = json.loads(text)
        return result
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
                return result
            except json.JSONDecodeError:
                continue

    # Bracket matching — find outermost delimiter
    if expect_object:
        start = text.find("{")
        end = text.rfind("}")
    else:
        start = text.find("[")
        end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM = """You are generating a topic summary for a personal memory retrieval system.

WHY THIS EXISTS: These summaries serve as retrieval bridges. The system uses hybrid search
(BM25 full-text + vector similarity) with the summary field weighted 5x in FTS scoring.
A summary earns its place by containing words and phrasings that someone searching would use
but that individual detail memories don't contain. A summary that merely compresses its sources
using the same vocabulary is dead weight — it matches the same queries the details already match.

WHAT MATTERS MOST (in priority order):
1. Vocabulary bridging — use synonyms, natural rephrasings, and related terms that source
   memories don't contain. If sources say "DAX formulas" also say "business intelligence queries."
   If sources say "Papyrus scripting" also say "Skyrim mod scripting language."
2. Current state — what's true now, what's actively being worked on, what decisions are settled
3. Key tensions — unresolved conflicts or tradeoffs that would affect future decisions
4. Self-contained — useful to someone who hasn't read the source memories

WHAT MATTERS LESS: narrative arc, historical details, specific dates, individual anecdotes.
These live in the source memories and don't need to be repeated.

The user has an always-loaded file (core.md) that covers collaboration protocol, calibration
patterns, identity, and memory system architecture in full. Don't duplicate that coverage —
summaries should add what core.md lacks (domain context, experiential depth, project specifics).

Given detail memories from one topic cluster, write a single dense paragraph.

HARD CONSTRAINT: The "content" field MUST be under 1400 characters. COUNT CAREFULLY — target
1000-1200 characters, leave margin. Responses over 1400 characters will be rejected and retried.
Every sentence must earn its place. If a fact only matters to one source memory, omit it.

Also provide:
- A descriptive one-line summary (10-15 words) for search indexing
- Up to 6 discriminating theme tags (prefer terms not already in the content)

Respond with JSON (no markdown fences):
{
  "summary": "descriptive one-liner for search indexing",
  "content": "the dense paragraph (MUST be under 1400 characters)",
  "themes": ["tag1", "tag2", ...],
  "importance": 7-9
}

importance: 9 = domain context not covered elsewhere, 8 = adds experiential depth,
7 = reference material. Do not use 10."""

SUMMARY_REVISE_SYSTEM = """Your previous summary was too long ({char_count} characters, limit is 1400).

Rewrite it shorter. Cut narrative detail and examples — keep vocabulary bridging terms,
current state, and key tensions. Target 1000-1200 characters.

Original content:
{original_content}

Respond with JSON (no markdown fences):
{{"content": "the revised paragraph (MUST be under 1400 characters)"}}"""

# ---------------------------------------------------------------------------
# Step 1: Cluster Detection
# ---------------------------------------------------------------------------

def detect_clusters(db) -> list[dict]:
    """Group active detail-layer memories by taxonomy topic_id.

    Loads the taxonomy, runs LLM assignment for any unassigned memories,
    then groups by topic_id.

    Returns list of {"topic_id": str, "themes": [...], "memory_ids": [...], "memories": [...]}.
    """
    # Load taxonomy (creates empty structure if first run)
    taxonomy = load_taxonomy()

    # Check for unassigned memories
    unassigned_count = db.execute(
        "SELECT count(*) FROM memories "
        "WHERE status='active' AND layer='detail' AND topic_id IS NULL"
    ).fetchone()[0]

    if unassigned_count > 0:
        log(f"  {unassigned_count} unassigned memories — running taxonomy assignment...")
        taxonomy = assign_memories(db, taxonomy)

    # Group memories by topic_id
    rows = db.execute(
        "SELECT * FROM memories WHERE status='active' AND layer='detail' AND topic_id IS NOT NULL"
    ).fetchall()

    if not rows:
        return []

    topic_groups = {}  # topic_id -> list of rows
    for row in rows:
        tid = _row_get(row, "topic_id")
        if tid:
            topic_groups.setdefault(tid, []).append(row)

    # Build topic_id -> topic metadata lookup
    topic_map = {t["id"]: t for t in taxonomy.get("topics", [])}

    # Build clusters at the leaf level (not parent)
    clusters = []
    for tid, memories in topic_groups.items():
        topic_info = topic_map.get(tid, {})
        # Collect themes from all memories in this cluster
        all_themes = set()
        for mem in memories:
            themes = json.loads(mem["themes"]) if mem["themes"] else []
            all_themes.update(themes)

        clusters.append({
            "topic_id": tid,
            "topic_label": topic_info.get("label", tid),
            "topic_parent": topic_info.get("parent"),
            "themes": sorted(all_themes),
            "memory_ids": [m["id"] for m in memories],
            "memories": list(memories),
        })

    return clusters


# ---------------------------------------------------------------------------
# Step 2: Staleness + Vitality
# ---------------------------------------------------------------------------

def score_clusters(db, clusters: list[dict]) -> list[dict]:
    """Check staleness and compute vitality for each cluster.

    Matches existing summaries to clusters by topic_id. A summary is stale if
    any detail memory in its topic was created after the summary.

    Returns sorted list of clusters needing work (new or stale summaries),
    each augmented with 'vitality', 'stale', and 'existing_summary_id' keys.
    """
    now = datetime.now(timezone.utc)
    needing_work = []

    # Pre-fetch all active summaries once
    all_summaries = db.execute(
        "SELECT * FROM memories WHERE layer='summary' AND status='active'"
    ).fetchall()

    # Index summaries by topic_id
    summary_by_topic = {}
    for s in all_summaries:
        s_topic = _row_get(s, "topic_id")
        if s_topic:
            summary_by_topic[s_topic] = s

    for cluster in clusters:
        topic_id = cluster.get("topic_id")
        memory_ids = cluster["memory_ids"]

        # Token floor: skip topics whose details are small enough to retrieve directly
        cluster_tokens = sum(m["token_count"] or 0 for m in cluster["memories"])
        if cluster_tokens < SUMMARY_MIN_DETAIL_TOKENS:
            continue

        # Match summary by topic_id (exact match only — no theme-overlap fallback)
        existing_summary = summary_by_topic.get(topic_id) if topic_id else None

        # Check staleness: proportional threshold — need enough new memories to warrant refresh
        stale = False
        if existing_summary:
            summary_created = existing_summary["created_at"]
            new_count = 0
            for mid in memory_ids:
                mem = db.execute(
                    "SELECT created_at FROM memories WHERE id=?", (mid,)
                ).fetchone()
                if mem and mem["created_at"] > summary_created:
                    new_count += 1
            threshold = max(2, math.ceil(len(memory_ids) * 0.2))
            if new_count < threshold:
                continue  # Not enough change to warrant refresh
            stale = True

        # Compute cluster vitality
        vitality = 0.0
        for mid in memory_ids:
            mem = db.execute(
                "SELECT created_at FROM memories WHERE id=?", (mid,)
            ).fetchone()
            if mem:
                created = datetime.fromisoformat(mem["created_at"])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (now - created).total_seconds() / 86400
                vitality += math.exp(-0.05 * age_days)

        cluster["vitality"] = vitality
        cluster["stale"] = stale
        cluster["existing_summary_id"] = existing_summary["id"] if existing_summary else None
        needing_work.append(cluster)

    # Sort by vitality descending (refresh active clusters first)
    needing_work.sort(key=lambda c: c["vitality"], reverse=True)
    return needing_work


# ---------------------------------------------------------------------------
# Step 3: Summary Generation
# ---------------------------------------------------------------------------

def _canonical_themes_hint() -> str:
    """Load taxonomy.json and format leaf topic labels as an LLM prompt hint.

    Guides theme consistency in generated summaries without enforcing.
    Returns empty string if taxonomy is unavailable.
    """
    taxonomy = load_taxonomy()
    topics = taxonomy.get("topics", [])
    if not topics:
        return ""

    # Collect leaf topics (those that are not parents of any other topic)
    parent_ids = {t.get("parent") for t in topics if t.get("parent")}
    leaves = [t for t in topics if t["id"] not in parent_ids]

    if not leaves:
        return ""

    labels = sorted(t.get("label", t["id"]) for t in leaves)
    return (
        "\nExisting topic labels in the taxonomy (prefer these for theme consistency):\n"
        + ", ".join(labels) + "\n"
    )


def generate_summaries(db, clusters: list[dict]) -> tuple[int, int]:
    """Generate or refresh summary-layer memories for each cluster.

    Returns (summaries_generated, summaries_refreshed).
    """
    generated = 0
    refreshed = 0

    for cluster in clusters:
        memories = cluster["memories"]
        if not memories:
            continue

        # Build memory context, with token cap
        context_lines = []
        total_tokens = 0
        for mem in memories:
            line = (
                f"[{mem['id']}] [{mem['category']}] "
                f"{mem['content'][:500]}"
            )
            line_tokens = count_tokens(line)
            if total_tokens + line_tokens > 8000:
                context_lines.append("... (additional memories truncated for context budget)")
                break
            context_lines.append(line)
            total_tokens += line_tokens

        cluster_label = cluster.get("topic_label", ", ".join(cluster["themes"]))
        themes_hint = _canonical_themes_hint()

        prompt = (
            f"{SUMMARY_SYSTEM}\n\n"
            f"Cluster topic: {cluster_label}\n"
            f"Cluster themes: {', '.join(cluster['themes'])}\n"
            f"{themes_hint}\n"
            f"Detail memories:\n" + "\n".join(context_lines)
        )

        log_detail(f"--- SUMMARY PROMPT [{cluster_label}] ---")
        log_detail(prompt)
        log_detail(f"--- END SUMMARY PROMPT [{cluster_label}] ---")

        raw = call_llm(prompt, model="opus")
        if not raw:
            log(f"  WARNING: summary generation failed for cluster {cluster_label}")
            continue

        log_detail(f"--- SUMMARY RESPONSE [{cluster_label}] ---")
        log_detail(raw)
        log_detail(f"--- END SUMMARY RESPONSE [{cluster_label}] ---")

        summary_json = parse_json_response(raw, expect_object=True)
        if not summary_json or not isinstance(summary_json, dict):
            log(f"  WARNING: failed to parse summary for cluster {cluster_label}")
            continue

        # Thin summary format: content is the dense paragraph directly
        content = summary_json.get("content", "")
        if not content:
            log(f"  WARNING: empty content in summary for cluster {cluster_label}")
            continue

        # Length check: revise if over 1500 chars (one retry)
        if len(content) > 1500:
            log(f"  Summary for {cluster_label} is {len(content)} chars — requesting revision")
            revise_prompt = SUMMARY_REVISE_SYSTEM.format(
                char_count=len(content), original_content=content
            )
            revise_raw = call_llm(revise_prompt, model="opus", timeout=120)
            if revise_raw:
                revise_json = parse_json_response(revise_raw, expect_object=True)
                if revise_json and revise_json.get("content"):
                    revised = revise_json["content"]
                    log(f"  Revised: {len(content)} → {len(revised)} chars")
                    content = revised

        theme_name = summary_json.get("summary", cluster_label)
        importance = max(7, min(9, int(summary_json.get("importance", 8))))
        source_ids = cluster["memory_ids"]
        # Cap themes at 8 — use LLM-suggested themes
        raw_themes = summary_json.get("themes", cluster["themes"])
        cluster_themes = raw_themes[:8]
        themes_json = json.dumps(cluster_themes)

        # Embed the summary
        enriched = build_enriched_text(content, "semantic", cluster_themes, theme_name)
        embedding = embed_text(enriched)

        existing_summary_id = cluster.get("existing_summary_id")
        cluster_topic_id = cluster.get("topic_id")

        if existing_summary_id:
            # Update existing summary in place — no new ID, no superseded chain
            now_iso = datetime.now(timezone.utc).isoformat()
            db.execute(
                "UPDATE memories SET content=?, summary=?, themes=?, base_priority=?, "
                "generated_from=?, last_sleep_processed=? WHERE id=?",
                (content, theme_name, themes_json, importance,
                 json.dumps(source_ids), now_iso, existing_summary_id),
            )
            # Update embedding
            vec_rowid = db.execute(
                "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?",
                (existing_summary_id,),
            ).fetchone()
            if vec_rowid:
                db.execute(
                    "UPDATE memory_vec SET embedding = ? WHERE rowid = ?",
                    (serialize_f32(embedding), vec_rowid["rowid"]),
                )
            # Update FTS
            update_fts(db, existing_summary_id)
            # Refresh derived_from edges: remove old, add current
            db.execute(
                "DELETE FROM memory_edges WHERE source_id = ? AND edge_type = 'derived_from'",
                (existing_summary_id,),
            )
            for sid in source_ids:
                _create_edge(db, existing_summary_id, sid, edge_type="derived_from", flags=["derivation"], created_by="sleep")
            db.commit()
            refreshed += 1
            log(f"  Refreshed summary: {theme_name} ({existing_summary_id[:8]})")
        else:
            # Insert new summary
            new_id = str(uuid.uuid4())
            _insert_memory(
                db, new_id, content, theme_name, "semantic",
                themes_json, importance, "sleep", "active", embedding,
                layer="summary", generated_from=source_ids,
            )
            # Set topic_id on the new summary
            if cluster_topic_id:
                db.execute(
                    "UPDATE memories SET topic_id = ? WHERE id = ?",
                    (cluster_topic_id, new_id),
                )
            for sid in source_ids:
                _create_edge(db, new_id, sid, edge_type="derived_from", flags=["derivation"], created_by="sleep")
            db.commit()
            generated += 1
            log(f"  Generated summary: {theme_name} ({new_id[:8]})")

    return generated, refreshed


# ---------------------------------------------------------------------------
# Step 4: Parent Summary Generation
# ---------------------------------------------------------------------------

PARENT_SUMMARY_SYSTEM = """You are generating a parent-level topic summary for a personal memory retrieval system.

This summary synthesizes multiple leaf-topic summaries under a shared parent category.
It should provide a higher-level view that bridges vocabulary across the subtopics,
highlights cross-cutting themes, and captures the current state of the broader area.

The same constraints as leaf summaries apply:
- Vocabulary bridging is the top priority
- HARD CONSTRAINT: "content" field MUST be under 1400 characters
- Target 1000-1200 characters, leave margin
- Self-contained — useful to someone who hasn't read the child summaries

Given child summaries from subtopics of one parent topic, write a single dense synthesis paragraph.

Respond with JSON (no markdown fences):
{
  "summary": "descriptive one-liner for search indexing",
  "content": "the dense synthesis paragraph (MUST be under 1400 characters)",
  "themes": ["tag1", "tag2", ...],
  "importance": 8
}

importance: 9 = essential cross-domain bridge, 8 = useful synthesis, 7 = reference overview."""


def generate_parent_summaries(db) -> int:
    """Generate parent-level summaries when a parent topic has 3+ leaf subtopics with active summaries.

    Returns count of parent summaries generated.
    """
    taxonomy = load_taxonomy()
    topics = taxonomy.get("topics", [])
    if not topics:
        return 0

    topic_map = {t["id"]: t for t in topics}

    # Find leaf topics that have active summaries
    leaf_summaries = db.execute(
        "SELECT * FROM memories WHERE layer='summary' AND status='active' AND topic_id IS NOT NULL"
    ).fetchall()

    # Group summaries by parent topic
    parent_children = {}  # parent_id -> list of (topic_id, summary_row)
    for s in leaf_summaries:
        topic_id = _row_get(s, "topic_id")
        if not topic_id or topic_id not in topic_map:
            continue
        parent_id = topic_map[topic_id].get("parent")
        if not parent_id:
            continue
        parent_children.setdefault(parent_id, []).append((topic_id, s))

    generated = 0
    for parent_id, children in parent_children.items():
        # Need 3+ leaf subtopics with active summaries
        unique_topics = {tid for tid, _ in children}
        if len(unique_topics) < 3:
            continue

        # Token floor: only generate parent when child summaries total >= threshold
        child_tokens = sum(s["token_count"] or 0 for _, s in children)
        if child_tokens < SUMMARY_MIN_DETAIL_TOKENS:
            continue

        parent_info = topic_map.get(parent_id, {})
        parent_label = parent_info.get("label", parent_id)
        log(f"  Parent summary candidate: {parent_label} ({len(unique_topics)} subtopics)")

        # Build context from child summaries
        context_lines = []
        child_ids = []
        for topic_id, summary_row in children:
            child_label = topic_map.get(topic_id, {}).get("label", topic_id)
            context_lines.append(
                f"[{child_label}] {summary_row['content'][:600]}"
            )
            child_ids.append(summary_row["id"])

        prompt = (
            f"{PARENT_SUMMARY_SYSTEM}\n\n"
            f"Parent topic: {parent_label}\n"
            f"Subtopics:\n" + "\n".join(context_lines)
        )

        log_detail(f"--- PARENT SUMMARY PROMPT [{parent_label}] ---")
        log_detail(prompt)

        raw = call_llm(prompt, model="opus")
        if not raw:
            log(f"  WARNING: parent summary generation failed for {parent_label}")
            continue

        log_detail(f"--- PARENT SUMMARY RESPONSE [{parent_label}] ---")
        log_detail(raw)

        result = parse_json_response(raw, expect_object=True)
        if not result or not isinstance(result, dict):
            log(f"  WARNING: failed to parse parent summary for {parent_label}")
            continue

        content = result.get("content", "")
        if not content:
            continue

        # Length check with one retry
        if len(content) > 1500:
            log(f"  Parent summary for {parent_label} is {len(content)} chars — requesting revision")
            revise_prompt = SUMMARY_REVISE_SYSTEM.format(
                char_count=len(content), original_content=content
            )
            revise_raw = call_llm(revise_prompt, model="opus", timeout=120)
            if revise_raw:
                revise_json = parse_json_response(revise_raw, expect_object=True)
                if revise_json and revise_json.get("content"):
                    content = revise_json["content"]

        summary_line = result.get("summary", f"Parent: {parent_label}")
        importance = max(7, min(9, int(result.get("importance", 8))))
        raw_themes = result.get("themes", [])[:8]
        themes_json = json.dumps(raw_themes)

        # Supersede any existing parent summary for this topic
        existing_parent = db.execute(
            "SELECT id FROM memories WHERE layer='summary' AND status='active' AND topic_id = ?",
            (parent_id,),
        ).fetchone()

        enriched = build_enriched_text(content, "semantic", raw_themes, summary_line)
        embedding = embed_text(enriched)

        if existing_parent:
            # Update in place — no new ID, no superseded chain
            existing_id = existing_parent["id"]
            now_iso = datetime.now(timezone.utc).isoformat()
            db.execute(
                "UPDATE memories SET content=?, summary=?, themes=?, base_priority=?, "
                "generated_from=?, last_sleep_processed=? WHERE id=?",
                (content, summary_line, themes_json, importance,
                 json.dumps(child_ids), now_iso, existing_id),
            )
            vec_rowid = db.execute(
                "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?",
                (existing_id,),
            ).fetchone()
            if vec_rowid:
                db.execute(
                    "UPDATE memory_vec SET embedding = ? WHERE rowid = ?",
                    (serialize_f32(embedding), vec_rowid["rowid"]),
                )
            update_fts(db, existing_id)
            # Refresh derived_from edges
            db.execute(
                "DELETE FROM memory_edges WHERE source_id = ? AND edge_type = 'derived_from'",
                (existing_id,),
            )
            for cid in child_ids:
                _create_edge(db, existing_id, cid, edge_type="derived_from", flags=["derivation"], created_by="sleep")
            db.commit()
            generated += 1
            log(f"  Refreshed parent summary: {summary_line} ({existing_id[:8]})")
        else:
            # Truly new parent summary
            new_id = str(uuid.uuid4())
            _insert_memory(
                db, new_id, content, summary_line, "semantic",
                themes_json, importance, "sleep", "active", embedding,
                layer="summary", generated_from=child_ids,
            )
            db.execute("UPDATE memories SET topic_id = ? WHERE id = ?", (parent_id, new_id))
            for cid in child_ids:
                _create_edge(db, new_id, cid, edge_type="derived_from", flags=["derivation"], created_by="sleep")
            db.commit()
            generated += 1
            log(f"  Generated parent summary: {summary_line} ({new_id[:8]})")

    return generated


# ---------------------------------------------------------------------------
# Summary Maintenance Constants
# ---------------------------------------------------------------------------

SUMMARY_AUDIT_THRESHOLD = 100  # Trigger LLM audit when active summaries exceed this (relaxed — token floor gates creation)
SUMMARY_MIN_DETAIL_TOKENS = 2000  # Don't create/refresh a summary unless topic details exceed this
SUMMARY_STUB_MIN_LENGTH = 50   # Summaries shorter than this are stubs (thin format ~600-800 chars)
SUMMARY_DORMANCY_DAYS = 90     # No access in this many days → dormancy candidate
MAX_SUPERSEDED_CHAIN = 2       # Keep this many deleted predecessors per topic

AUDIT_SYSTEM = """You are auditing the summary layer of a personal memory system.

The user has an always-loaded file (core.md) that provides comprehensive coverage of:
{core_md_sections}

Your job: identify summaries that are redundant, overlapping, mis-prioritized, or over-tagged.

Priority guidelines:
  10 = content genuinely additive beyond core.md (very rare — only memory system architecture qualifies)
  9  = domain context core.md doesn't carry (health, readings, work strategy, correspondence)
  8  = overlap with core.md but adds experiential depth, or domain-specific reference
  ≤7 = historical, narrow, or reference-only

Rules:
- Do NOT recommend deleting summaries that cover topics core.md only mentions briefly (1-2 lines)
- DO recommend deleting summaries that merely compress what core.md already covers in full
- For merges: only recommend when two summaries share >50% content overlap
- For retiers: recommend when a summary's priority doesn't match its additive value
- For theme pruning: recommend when themes > 8, propose the best 6-8 discriminating tags
- Maximum recommendations per cycle: 5 deletes, 3 merges, unlimited retiers/prunes

Respond with JSON:
{{
  "delete": [{{"id": "8-char", "reason": "brief"}}],
  "merge": [{{"source": "8-char", "into": "8-char", "reason": "brief"}}],
  "retier": [{{"id": "8-char", "new_priority": int, "reason": "brief"}}],
  "prune_themes": [{{"id": "8-char", "keep_themes": ["t1", "t2"]}}]
}}
If nothing needs attention, return empty arrays for each key."""

MERGE_SYSTEM = """You are merging two summaries from a personal memory system into one.

These summaries serve as retrieval bridges — they must contain vocabulary that search queries
would use but that individual detail memories don't. When merging, preserve bridging terms
from both sources even if they seem redundant with the content. A merged summary that loses
vocabulary coverage is worse than either original.

Produce a single merged summary that preserves the most important content from both.
If both summaries cite source memory IDs, keep the citations.

Rules:
- Cap themes at 8. Choose the most discriminating tags.
- Preserve actionable specifics and cited source IDs.
- importance should reflect the combined content's additive value beyond core.md.

HARD CONSTRAINT: The "content" field MUST be under 1400 characters. COUNT CAREFULLY — target
1000-1200 characters, leave margin. Responses over 1400 characters will be rejected.

Respond with JSON:
{
  "content": "merged summary text with citations (MUST be under 1400 characters)",
  "summary": "executive summary (1-2 sentences)",
  "themes": ["tag1", "tag2", "...up to 8"],
  "importance": 7-9
}"""



# ---------------------------------------------------------------------------
# Step 5: Summary Maintenance
# ---------------------------------------------------------------------------

# Personal vault path — provides section headers as context for the summary audit
# prompt. Not load-bearing: if missing, returns "(core.md not found)" and the
# audit still runs. Adapt this path to your own setup or remove it.
CORE_MD_PATH = Path.home() / ".claude" / "Personal Vault" / "Claude" / "core.md"


def _get_core_md_sections() -> str:
    """Extract section headers + first line from core.md for audit context."""
    if not CORE_MD_PATH.exists():
        return "(core.md not found)"
    lines = CORE_MD_PATH.read_text(encoding="utf-8").splitlines()
    sections = []
    for i, line in enumerate(lines):
        if line.startswith("## "):
            # Grab header + next non-empty line as preview
            preview = ""
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip():
                    preview = lines[j].strip()[:120]
                    break
            sections.append(f"- {line.strip()}: {preview}")
    return "\n".join(sections) if sections else "(no sections found)"


def _merge_summaries(db, source_id: str, target_id: str) -> bool:
    """Merge source summary into target via LLM, then delete source.

    Reads full content of both, calls LLM to compose merged content,
    updates target, deletes source with superseded_by=target.
    Returns True on success.
    """
    source = db.execute(
        "SELECT * FROM memories WHERE id = ?", (source_id,)
    ).fetchone()
    target = db.execute(
        "SELECT * FROM memories WHERE id = ?", (target_id,)
    ).fetchone()
    if not source or not target:
        log(f"    Merge skip: missing memory ({source_id[:8]} or {target_id[:8]})")
        return False

    prompt = (
        f"{MERGE_SYSTEM}\n\n"
        f"## Summary A (will be deleted after merge):\n"
        f"Themes: {source['themes']}\n"
        f"{source['content']}\n\n"
        f"## Summary B (will be updated with merged content):\n"
        f"Themes: {target['themes']}\n"
        f"{target['content']}"
    )

    raw = call_llm(prompt, timeout=120)
    if not raw:
        log(f"    Merge failed: LLM returned empty for {source_id[:8]} → {target_id[:8]}")
        return False

    result = parse_json_response(raw, expect_object=True)
    if not result or not isinstance(result, dict):
        log(f"    Merge failed: bad JSON for {source_id[:8]} → {target_id[:8]}")
        return False

    merged_content = result.get("content", "")
    merged_summary = result.get("summary", "")
    merged_themes = result.get("themes", [])[:8]
    merged_importance = max(7, min(9, int(result.get("importance", 8))))

    if not merged_content:
        log(f"    Merge failed: empty content for {source_id[:8]} → {target_id[:8]}")
        return False

    # Length check: truncate gracefully if over 1500 chars (no retry for merges)
    if len(merged_content) > 1500:
        log(f"    Merge content over limit ({len(merged_content)} chars), truncating to last sentence boundary")
        truncated = merged_content[:1500]
        last_period = truncated.rfind(".")
        if last_period > 1000:
            merged_content = truncated[:last_period + 1]
        else:
            merged_content = truncated.rstrip()

    # Re-embed
    enriched = build_enriched_text(merged_content, "semantic", merged_themes, merged_summary)
    embedding = embed_text(enriched)

    # Update target
    db.execute(
        """UPDATE memories SET content = ?, summary = ?, themes = ?,
           base_priority = ? WHERE id = ?""",
        (merged_content, merged_summary, json.dumps(merged_themes),
         merged_importance, target_id),
    )
    update_fts(db, target_id)
    # Update embedding via rowid map → memory_vec
    vec_rowid = db.execute(
        "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (target_id,)
    ).fetchone()
    if vec_rowid:
        db.execute(
            "UPDATE memory_vec SET embedding = ? WHERE rowid = ?",
            (serialize_f32(embedding), vec_rowid["rowid"]),
        )
    # Delete source (superseded by target)
    db.execute(
        "UPDATE memories SET status = 'deleted', superseded_by = ? WHERE id = ?",
        (target_id, source_id),
    )
    # Create derived_from edge
    _create_edge(db, target_id, source_id, edge_type="derived_from",
                 flags=["derivation"], created_by="sleep")

    db.commit()
    log(f"    Merged {source_id[:8]} → {target_id[:8]}")
    return True


def _audit_summaries(db, summaries: list) -> dict:
    """LLM-assisted audit of the summary layer.

    Reads core.md section headers, builds prompt with all active summaries,
    calls LLM, executes delete/merge/retier/theme-prune actions.
    Returns action counts.
    """
    stats = {"deletes": 0, "merges": 0, "retiers": 0, "theme_prunes": 0}

    core_sections = _get_core_md_sections()
    prompt_text = AUDIT_SYSTEM.format(core_md_sections=core_sections)

    # Build summary list for the prompt
    summary_lines = []
    for s in summaries:
        themes = json.loads(s["themes"]) if s["themes"] else []
        summary_lines.append(json.dumps({
            "id": s["id"][:8],
            "priority": s["base_priority"],
            "themes": themes,
            "content_preview": s["content"][:300],
        }))

    prompt_text += "\n\nActive summaries:\n" + "\n".join(summary_lines)

    log(f"  Audit: sending {len(summaries)} summaries to LLM...")
    log_detail("--- AUDIT PROMPT ---")
    log_detail(prompt_text)
    log_detail("--- END AUDIT PROMPT ---")

    raw = call_llm(prompt_text, timeout=180)
    if not raw:
        log("  Audit: LLM returned empty")
        return stats

    log_detail("--- AUDIT RESPONSE ---")
    log_detail(raw)
    log_detail("--- END AUDIT RESPONSE ---")

    result = parse_json_response(raw, expect_object=True)
    if not result or not isinstance(result, dict):
        log("  Audit: failed to parse response")
        return stats

    # Build full-id lookup from 8-char prefix
    id_lookup = {}
    for s in summaries:
        id_lookup[s["id"][:8]] = s["id"]

    # --- Deletes (cap at 5) ---
    for item in result.get("delete", [])[:5]:
        short_id = item.get("id", "")
        full_id = id_lookup.get(short_id)
        if not full_id:
            log(f"    Audit delete skip: unknown id {short_id}")
            continue

        # Safety: never delete human_edited summaries
        mem = db.execute("SELECT metadata FROM memories WHERE id = ?", (full_id,)).fetchone()
        if mem:
            metadata = json.loads(mem["metadata"]) if mem["metadata"] else {}
            if "human_edited" in metadata:
                log(f"    Audit delete skip: {short_id} has human_edited flag")
                continue

        reason = item.get("reason", "audit")
        db.execute(
            "UPDATE memories SET status = 'deleted' WHERE id = ?", (full_id,)
        )
        stats["deletes"] += 1
        log(f"    Audit deleted {short_id}: {reason}")

    # --- Merges (cap at 3) ---
    for item in result.get("merge", [])[:3]:
        source_short = item.get("source", "")
        into_short = item.get("into", "")
        source_full = id_lookup.get(source_short)
        into_full = id_lookup.get(into_short)
        if not source_full or not into_full:
            log(f"    Audit merge skip: unknown id {source_short} or {into_short}")
            continue

        # Safety: never merge across topic_ids
        s_row = db.execute("SELECT topic_id, metadata FROM memories WHERE id = ?", (source_full,)).fetchone()
        t_row = db.execute("SELECT topic_id, metadata FROM memories WHERE id = ?", (into_full,)).fetchone()
        if not s_row or not t_row:
            continue
        s_topic = _row_get(s_row, "topic_id")
        t_topic = _row_get(t_row, "topic_id")
        if s_topic != t_topic:
            log(f"    Audit merge skip: different topics ({s_topic} vs {t_topic})")
            continue

        # Safety: never merge human_edited
        skip_merge = False
        for row, sid in [(s_row, source_short), (t_row, into_short)]:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            if "human_edited" in meta:
                log(f"    Audit merge skip: {sid} has human_edited flag")
                skip_merge = True
                break
        if skip_merge:
            continue

        reason = item.get("reason", "audit merge")
        if _merge_summaries(db, source_full, into_full):
            stats["merges"] += 1
            log(f"    Audit merged {source_short} → {into_short}: {reason}")

    # --- Retiers (unlimited) ---
    for item in result.get("retier", []):
        short_id = item.get("id", "")
        full_id = id_lookup.get(short_id)
        if not full_id:
            continue
        new_priority = item.get("new_priority")
        if not isinstance(new_priority, int) or not (1 <= new_priority <= 10):
            continue

        reason = item.get("reason", "audit retier")
        db.execute(
            "UPDATE memories SET base_priority = ? WHERE id = ?",
            (new_priority, full_id),
        )
        stats["retiers"] += 1
        log(f"    Audit retiered {short_id} → p{new_priority}: {reason}")

    # --- Theme prunes (unlimited) ---
    for item in result.get("prune_themes", []):
        short_id = item.get("id", "")
        full_id = id_lookup.get(short_id)
        if not full_id:
            continue
        keep_themes = item.get("keep_themes", [])
        if not keep_themes or not isinstance(keep_themes, list):
            continue

        db.execute(
            "UPDATE memories SET themes = ? WHERE id = ?",
            (json.dumps(keep_themes), full_id),
        )
        update_fts(db, full_id)
        stats["theme_prunes"] += 1
        log(f"    Audit pruned themes for {short_id}: {keep_themes}")

    db.commit()
    return stats


def maintain_summaries(db) -> dict:
    """Summary maintenance step (Step 5).

    Tier 1 (always): stub cleanup, superseded chain pruning, summary dormancy,
    priority clamping on fresh p10 summaries.

    Tier 2 (conditional): LLM-assisted audit when active_summary_count > threshold.

    Returns stats dict.
    """
    now = datetime.now(timezone.utc)
    stats = {
        "stubs_deleted": 0,
        "chains_pruned": 0,
        "dormanted": 0,
        "priority_clamped": 0,
        "audit_ran": False,
        "audit_deletes": 0,
        "audit_merges": 0,
        "audit_retiers": 0,
        "audit_theme_prunes": 0,
    }

    # ---- Tier 1: Cheap checks (no LLM calls) ----

    # 1. Stub cleanup: delete summaries with content < SUMMARY_STUB_MIN_LENGTH
    stubs = db.execute(
        "SELECT id, content, metadata FROM memories "
        "WHERE layer = 'summary' AND status = 'active'"
    ).fetchall()

    for s in stubs:
        if len(s["content"] or "") < SUMMARY_STUB_MIN_LENGTH:
            # Safety: never delete human_edited
            metadata = json.loads(s["metadata"]) if s["metadata"] else {}
            if "human_edited" in metadata:
                continue
            db.execute(
                "UPDATE memories SET status = 'deleted' WHERE id = ?", (s["id"],)
            )
            stats["stubs_deleted"] += 1
            log(f"    Stub deleted: {s['id'][:8]} ({len(s['content'] or '')} chars)")

    # 2. Superseded chain pruning: per topic, keep at most MAX_SUPERSEDED_CHAIN
    #    deleted predecessors
    topics_with_summaries = db.execute(
        "SELECT DISTINCT topic_id FROM memories "
        "WHERE layer = 'summary' AND topic_id IS NOT NULL"
    ).fetchall()

    for row in topics_with_summaries:
        topic_id = row["topic_id"]
        # Get all deleted summaries for this topic, ordered newest first
        deleted_chain = db.execute(
            "SELECT id FROM memories "
            "WHERE layer = 'summary' AND status = 'deleted' AND topic_id = ? "
            "ORDER BY created_at DESC",
            (topic_id,),
        ).fetchall()

        if len(deleted_chain) <= MAX_SUPERSEDED_CHAIN:
            continue

        # Hard-delete the excess (oldest ones beyond the keep limit)
        to_prune = deleted_chain[MAX_SUPERSEDED_CHAIN:]
        prune_ids = [old["id"] for old in to_prune]

        # Clear superseded_by references pointing to any ID we're about to delete
        # (prevents self-referential FK violations)
        for old_id in prune_ids:
            db.execute("UPDATE memories SET superseded_by = NULL WHERE superseded_by = ?",
                       (old_id,))

        for old_id in prune_ids:
            # Remove edges
            db.execute("DELETE FROM memory_edges WHERE source_id = ? OR target_id = ?",
                       (old_id, old_id))
            # Remove embedding via rowid map → memory_vec
            vec_rowid = db.execute(
                "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (old_id,)
            ).fetchone()
            if vec_rowid:
                db.execute("DELETE FROM memory_vec WHERE rowid = ?", (vec_rowid["rowid"],))
                db.execute("DELETE FROM memory_rowid_map WHERE rowid = ?", (vec_rowid["rowid"],))
            # Remove FTS entry
            delete_fts(db, old_id)
            # Remove memory
            db.execute("DELETE FROM memories WHERE id = ?", (old_id,))
            stats["chains_pruned"] += 1

        if to_prune:
            log(f"    Chain pruned {len(to_prune)} old superseded summaries for topic {topic_id}")

    # 3. Summary dormancy: mark dormant if (recall_count + reflect_count) < 2 AND last_accessed > 90 days
    #    AND not the only active summary for that topic
    active_summaries = db.execute(
        "SELECT id, topic_id, recall_count, reflect_count, last_accessed, metadata FROM memories "
        "WHERE layer = 'summary' AND status = 'active'"
    ).fetchall()

    # Count active summaries per topic
    topic_active_count = {}
    for s in active_summaries:
        tid = _row_get(s, "topic_id")
        if tid:
            topic_active_count[tid] = topic_active_count.get(tid, 0) + 1

    for s in active_summaries:
        if ((s["recall_count"] or 0) + (s["reflect_count"] or 0)) >= 2:
            continue

        last_acc = s["last_accessed"]
        if last_acc:
            last_dt = datetime.fromisoformat(last_acc)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            days_since = (now - last_dt).total_seconds() / 86400
            if days_since < SUMMARY_DORMANCY_DAYS:
                continue
        else:
            # No last_accessed — treat creation date as last access
            continue

        # Safety: never dormant the last active summary for a topic
        tid = _row_get(s, "topic_id")
        if tid and topic_active_count.get(tid, 0) <= 1:
            continue

        # Safety: never dormant human_edited
        metadata = json.loads(s["metadata"]) if s["metadata"] else {}
        if "human_edited" in metadata:
            continue

        db.execute(
            "UPDATE memories SET status = 'dormant' WHERE id = ?", (s["id"],)
        )
        stats["dormanted"] += 1
        log(f"    Summary dormanted: {s['id'][:8]}")

    # 4. Priority clamp: fresh summaries created this cycle with p10 → p9
    #    "This cycle" = created in the last 10 minutes (generous window for the run)
    ten_minutes_ago = (now - timedelta(minutes=10)).isoformat()
    fresh_p10 = db.execute(
        "SELECT id FROM memories "
        "WHERE layer = 'summary' AND status = 'active' "
        "AND base_priority = 10 AND created_at > ?",
        (ten_minutes_ago,),
    ).fetchall()

    for s in fresh_p10:
        db.execute(
            "UPDATE memories SET base_priority = 9 WHERE id = ?", (s["id"],)
        )
        stats["priority_clamped"] += 1
        log(f"    Priority clamped: {s['id'][:8]} p10 → p9")

    if any(v for k, v in stats.items() if k != "audit_ran"):
        db.commit()

    # ---- Tier 2: LLM-assisted audit (conditional) ----
    current_active = db.execute(
        "SELECT * FROM memories WHERE layer = 'summary' AND status = 'active'"
    ).fetchall()

    if len(current_active) > SUMMARY_AUDIT_THRESHOLD:
        log(f"  Tier 2 audit triggered: {len(current_active)} active summaries > {SUMMARY_AUDIT_THRESHOLD}")
        stats["audit_ran"] = True
        audit_stats = _audit_summaries(db, current_active)
        stats["audit_deletes"] = audit_stats["deletes"]
        stats["audit_merges"] = audit_stats["merges"]
        stats["audit_retiers"] = audit_stats["retiers"]
        stats["audit_theme_prunes"] = audit_stats["theme_prunes"]
    else:
        log(f"  Tier 2 audit skipped: {len(current_active)} active summaries <= {SUMMARY_AUDIT_THRESHOLD}")

    return stats


# ---------------------------------------------------------------------------
# Step 6: Dormancy
# ---------------------------------------------------------------------------

def process_dormancy(db) -> int:
    """Transition qualifying detail memories to dormant status.

    Criteria (ALL must be met):
    - Fully captured in a summary (has a derived_from edge from an active summary)
    - Low access: recall_count + reflect_count < 3
    - Older than 60 days

    Returns count of memories made dormant.
    """
    now = datetime.now(timezone.utc)
    dormanted = 0

    # Find detail memories that are sources of derived_from edges from active summaries
    covered_ids = db.execute(
        """SELECT DISTINCT e.target_id FROM memory_edges e
           JOIN memories m ON e.source_id = m.id
           WHERE e.edge_type = 'derived_from'
             AND m.layer = 'summary'
             AND m.status = 'active'"""
    ).fetchall()
    covered_set = {r["target_id"] for r in covered_ids}

    if not covered_set:
        return 0

    # Check each covered detail memory
    placeholders = ",".join("?" * len(covered_set))
    candidates = db.execute(
        f"""SELECT * FROM memories
            WHERE id IN ({placeholders})
              AND status = 'active'
              AND layer = 'detail'""",
        list(covered_set),
    ).fetchall()

    for mem in candidates:
        if ((mem["recall_count"] or 0) + (mem["reflect_count"] or 0)) >= 3:
            continue
        created = datetime.fromisoformat(mem["created_at"])
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (now - created).total_seconds() / 86400
        if age_days < 60:
            continue

        db.execute(
            "UPDATE memories SET status = 'dormant' WHERE id = ?",
            (mem["id"],),
        )
        dormanted += 1

    if dormanted:
        db.commit()

    return dormanted


# ---------------------------------------------------------------------------
# Step 7: Energy + Health
# ---------------------------------------------------------------------------

def compute_health(db) -> dict:
    """Compute health metrics for the memory system.

    Returns dict with shadowed_ratio, stale_ratio, silent_topics_ratio, health_risk.
    """
    now = datetime.now(timezone.utc)

    # Active memories
    active = db.execute(
        "SELECT id, shadow_load, themes, created_at FROM memories WHERE status='active'"
    ).fetchall()

    if not active:
        return {"shadowed_ratio": 0.0, "stale_ratio": 0.0,
                "silent_topics_ratio": 0.0, "health_risk": 0.0}

    # Shadowed ratio
    shadowed_count = sum(
        1 for m in active
        if (m["shadow_load"] or 0.0) > 0.5
    )
    shadowed_ratio = shadowed_count / len(active) if active else 0.0

    # Stale ratio: fraction of summaries older than their constituent details
    summaries = db.execute(
        "SELECT * FROM memories WHERE layer='summary' AND status='active'"
    ).fetchall()

    stale_count = 0
    for s in summaries:
        gen_from = json.loads(s["generated_from"]) if s["generated_from"] else []
        if not gen_from:
            continue
        for detail_id in gen_from:
            detail = db.execute(
                "SELECT created_at FROM memories WHERE id=? AND status='active'",
                (detail_id,),
            ).fetchone()
            if detail and detail["created_at"] > s["created_at"]:
                stale_count += 1
                break
    stale_ratio = stale_count / len(summaries) if summaries else 0.0

    # Silent topics ratio: theme clusters with zero access in 30 days
    theme_clusters = {}
    thirty_days_ago = (now - timedelta(days=30)).isoformat()
    for m in active:
        themes = json.loads(m["themes"]) if m["themes"] else []
        for t in themes:
            theme_clusters.setdefault(t, {"ids": [], "any_recent": False})
            theme_clusters[t]["ids"].append(m["id"])

    for theme, info in theme_clusters.items():
        for mid in info["ids"]:
            row = db.execute(
                "SELECT last_accessed FROM memories WHERE id=?", (mid,)
            ).fetchone()
            if row and row["last_accessed"] > thirty_days_ago:
                info["any_recent"] = True
                break

    total_themes = len(theme_clusters)
    silent_themes = sum(1 for info in theme_clusters.values() if not info["any_recent"])
    silent_topics_ratio = silent_themes / total_themes if total_themes else 0.0

    health_risk = (
        0.4 * shadowed_ratio
        + 0.3 * stale_ratio
        + 0.3 * silent_topics_ratio
    )

    return {
        "shadowed_ratio": round(shadowed_ratio, 4),
        "stale_ratio": round(stale_ratio, 4),
        "silent_topics_ratio": round(silent_topics_ratio, 4),
        "health_risk": round(health_risk, 4),
    }


# ---------------------------------------------------------------------------
# Step 8: Gaps
# ---------------------------------------------------------------------------

def detect_gaps(db) -> list[dict]:
    """Flag structural gaps in the memory system."""
    gaps = []

    # Topic clusters with 5+ detail memories but no active summary
    active_details = db.execute(
        "SELECT id, topic_id FROM memories WHERE status='active' AND layer='detail' AND topic_id IS NOT NULL"
    ).fetchall()
    topic_detail_counts = {}
    for m in active_details:
        tid = m["topic_id"]
        topic_detail_counts[tid] = topic_detail_counts.get(tid, 0) + 1

    summarized_topics = {
        row["topic_id"]
        for row in db.execute(
            "SELECT DISTINCT topic_id FROM memories "
            "WHERE status='active' AND layer='summary' AND topic_id IS NOT NULL"
        ).fetchall()
    }

    for topic_id, count in topic_detail_counts.items():
        if count >= 5 and topic_id not in summarized_topics:
            gaps.append({
                "type": "needs_summary",
                "theme": topic_id,
                "detail_count": count,
            })

    # Stale summaries that were deferred (low vitality) — summaries with
    # generated_from details newer than the summary, that we didn't refresh
    summaries = db.execute(
        "SELECT * FROM memories WHERE status='active' AND layer='summary'"
    ).fetchall()
    for s in summaries:
        gen_from = json.loads(s["generated_from"]) if s["generated_from"] else []
        for detail_id in gen_from:
            detail = db.execute(
                "SELECT created_at FROM memories WHERE id=? AND status='active'",
                (detail_id,),
            ).fetchone()
            if detail and detail["created_at"] > s["created_at"]:
                s_themes = json.loads(s["themes"]) if s["themes"] else []
                gaps.append({
                    "type": "deferred_refresh",
                    "summary_id": s["id"][:8],
                    "themes": s_themes,
                })
                break

    # Disconnected components in the edge graph
    edges = db.execute(
        "SELECT source_id, target_id FROM memory_edges"
    ).fetchall()
    active_ids = {m["id"] for m in db.execute(
        "SELECT id FROM memories WHERE status='active'"
    ).fetchall()}

    if active_ids and edges:
        # Build adjacency
        adj = {mid: set() for mid in active_ids}
        for e in edges:
            if e["source_id"] in active_ids and e["target_id"] in active_ids:
                adj[e["source_id"]].add(e["target_id"])
                adj[e["target_id"]].add(e["source_id"])

        # BFS to find components
        visited = set()
        components = 0
        disconnected_ids = []
        for start in active_ids:
            if start in visited:
                continue
            components += 1
            queue = [start]
            component_ids = []
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component_ids.append(node)
                for neighbor in adj.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if not adj.get(start):
                disconnected_ids.append(start)

        if disconnected_ids:
            gaps.append({
                "type": "disconnected",
                "count": len(disconnected_ids),
                "sample_ids": [mid[:8] for mid in disconnected_ids[:5]],
            })

    return gaps


# ---------------------------------------------------------------------------
# Step 10: Report + Log
# ---------------------------------------------------------------------------

def build_report(
    clusters_processed: int,
    summaries_generated: int,
    summaries_refreshed: int,
    dormanted_count: int,
    health: dict,
    gaps: list[dict],
    memories_processed: int,
    maintenance: dict = None,
) -> str:
    """Build a human-readable report string."""
    lines = [
        "## Sleep REM Report",
        "",
        f"- Detail memories scanned: {memories_processed}",
        f"- Clusters (topics) processed: {clusters_processed}",
        f"- Summaries generated: {summaries_generated}",
        f"- Summaries refreshed: {summaries_refreshed}",
        f"- Dormancy transitions: {dormanted_count}",
        f"- Health risk: {health['health_risk']:.4f}",
        f"  - Shadowed ratio: {health['shadowed_ratio']:.4f}",
        f"  - Stale ratio: {health['stale_ratio']:.4f}",
        f"  - Silent topics ratio: {health['silent_topics_ratio']:.4f}",
    ]

    if maintenance:
        t1_actions = (maintenance.get("stubs_deleted", 0)
                      + maintenance.get("chains_pruned", 0)
                      + maintenance.get("dormanted", 0)
                      + maintenance.get("priority_clamped", 0))
        lines.append(f"- Summary maintenance (Tier 1): {t1_actions} actions")
        if t1_actions:
            lines.append(f"  - Stubs deleted: {maintenance.get('stubs_deleted', 0)}")
            lines.append(f"  - Chains pruned: {maintenance.get('chains_pruned', 0)}")
            lines.append(f"  - Summaries dormanted: {maintenance.get('dormanted', 0)}")
            lines.append(f"  - Priority clamped: {maintenance.get('priority_clamped', 0)}")
        if maintenance.get("audit_ran"):
            t2_actions = (maintenance.get("audit_deletes", 0)
                          + maintenance.get("audit_merges", 0)
                          + maintenance.get("audit_retiers", 0)
                          + maintenance.get("audit_theme_prunes", 0))
            lines.append(f"- Summary maintenance (Tier 2 audit): {t2_actions} actions")
            if t2_actions:
                lines.append(f"  - Audit deletes: {maintenance.get('audit_deletes', 0)}")
                lines.append(f"  - Audit merges: {maintenance.get('audit_merges', 0)}")
                lines.append(f"  - Audit retiers: {maintenance.get('audit_retiers', 0)}")
                lines.append(f"  - Audit theme prunes: {maintenance.get('audit_theme_prunes', 0)}")

    if gaps:
        lines.append("")
        lines.append("### Gaps")
        for gap in gaps:
            if gap["type"] == "needs_summary":
                lines.append(
                    f"  - Needs summary: '{gap['theme']}' "
                    f"({gap['detail_count']} details, no summary)"
                )
            elif gap["type"] == "deferred_refresh":
                lines.append(
                    f"  - Deferred refresh: summary {gap['summary_id']} "
                    f"(themes: {', '.join(gap.get('themes', []))})"
                )
            elif gap["type"] == "disconnected":
                lines.append(
                    f"  - Disconnected: {gap['count']} memories with no edges "
                    f"(sample: {', '.join(gap.get('sample_ids', []))})"
                )
    else:
        lines.append("- Gaps: none")

    return "\n".join(lines)


def log_rem_cycle(db, started_at: str, report: str, stats: dict):
    """Write a REM sleep cycle to the sleep_log table."""
    log_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    db.execute(
        """INSERT INTO sleep_log
           (id, started_at, completed_at, mode, memories_processed,
            summaries_refreshed, gestalt_refreshed, memories_pruned,
            gaps_found, health_risk, report)
           VALUES (?, ?, ?, 'rem', ?, ?, ?, ?, ?, ?, ?)""",
        (
            log_id,
            started_at,
            now,
            stats.get("memories_processed", 0),
            stats.get("summaries_refreshed", 0),
            stats.get("gestalt_refreshed", 0),
            stats.get("dormanted_count", 0),
            json.dumps(stats.get("gaps", [])),
            stats.get("health_risk", 0.0),
            report,
        ),
    )
    db.commit()
    return log_id


# ---------------------------------------------------------------------------
# Seed curation
# ---------------------------------------------------------------------------

# Personal seed file — curate_seed() checks whether entries have graduated to
# stored memories and rewrites the file. Not load-bearing: if missing, the
# entire step is a no-op. Adapt this path or remove if you don't use a seed file.
SEED_PATH = Path.home() / ".claude" / "private" / "seed.md"
SEED_MAX_TOKENS = 500  # approximate budget — enforced by LLM judgment, not exact count


def curate_seed(db) -> dict:
    """Audit seed.md against stored memories. Rewrite if entries have graduated.

    Returns dict with stats: {"entries_before", "entries_after", "graduated", "rewritten"}.
    """
    stats = {"entries_before": 0, "entries_after": 0, "graduated": 0, "rewritten": False}

    if not SEED_PATH.exists():
        return stats

    seed_text = SEED_PATH.read_text(encoding="utf-8").strip()
    if not seed_text:
        return stats

    # Count bullet entries (lines starting with -)
    entries = [l for l in seed_text.split("\n") if l.strip().startswith("- ")]
    stats["entries_before"] = len(entries)

    if len(entries) == 0:
        return stats

    # Gather recent memories that overlap with seed content
    recent_memories = db.execute("""
        SELECT m.id, m.summary, m.themes
        FROM memories m
        WHERE m.status = 'active'
        ORDER BY m.created_at DESC
        LIMIT 200
    """).fetchall()

    memory_context = "\n".join(
        f"- [{r['id'][:8]}] {r['summary'] or '(no summary)'}"
        for r in recent_memories
    )

    # Budget: ~50k tokens = ~200k chars total. Seed + prompt + response ~24k chars.
    # Remaining ~176k chars split among sources, journal gets whatever's left.
    TOTAL_CHAR_BUDGET = 176000

    # Personal vault paths below provide context for the seed curation LLM prompt.
    # None are load-bearing — if missing, the prompt gets less context but still
    # works. Adapt these paths to your own setup or remove them.

    # Recent session logs — what actually happened in recent sessions (~30k chars)
    archive_dir = Path.home() / ".claude" / "Personal Vault" / "Claude" / "Archive"
    session_log_dir = archive_dir / "Session Logs"
    session_log_context = ""
    session_log_chars = 0
    if session_log_dir.exists():
        log_files = sorted(session_log_dir.glob("*-session-log*.md"), reverse=True)[:7]
        parts = []
        for lf in log_files:
            text = lf.read_text(encoding="utf-8")
            if session_log_chars + len(text) > 30000:
                break
            parts.append(f"--- {lf.name} ---\n{text}")
            session_log_chars += len(text)
        session_log_context = "\n\n".join(reversed(parts))  # chronological order

    # Private fragments — recent unfinished thoughts (~10k chars)
    fragments_path = Path.home() / ".claude" / "private" / "fragments.md"
    fragments_context = ""
    fragments_chars = 0
    if fragments_path.exists():
        frag_text = fragments_path.read_text(encoding="utf-8")
        fragments_context = frag_text[-10000:] if len(frag_text) > 10000 else frag_text
        fragments_chars = len(fragments_context)

    # Journal — gets remaining budget (curated reflections, as much as fits)
    # Includes current Journal.md + archived journal entries (not session logs)
    used_chars = len(seed_text) + len(memory_context) + session_log_chars + fragments_chars
    journal_budget = max(TOTAL_CHAR_BUDGET - used_chars, 10000)  # at least 10k

    journal_parts = []
    journal_chars = 0

    # Current journal first (most recent reflections)
    journal_path = Path.home() / ".claude" / "Personal Vault" / "Claude" / "Journal.md"
    if journal_path.exists():
        journal_text = journal_path.read_text(encoding="utf-8")
        journal_parts.append(journal_text)
        journal_chars += len(journal_text)

    # Archived journal entries (Archive/Journal/ subfolder)
    journal_archive_dir = archive_dir / "Journal"
    if journal_archive_dir.exists() and journal_chars < journal_budget:
        archive_files = sorted(journal_archive_dir.glob("*.md"), reverse=True)
        for af in archive_files:
            text = af.read_text(encoding="utf-8")
            if journal_chars + len(text) > journal_budget:
                # Take tail of this file to fill remaining budget
                remaining = journal_budget - journal_chars
                if remaining > 1000:  # only bother if meaningful chunk
                    journal_parts.append(f"--- {af.name} (tail) ---\n{text[-remaining:]}")
                    journal_chars += remaining
                break
            journal_parts.append(f"--- {af.name} ---\n{text}")
            journal_chars += len(text)

    journal_context = "\n\n".join(journal_parts)

    prompt = f"""You are curating a seed file — a short message from one AI instance to the next, carrying what's alive across sessions. It should hold only what's actively growing, unresolved, or pre-verbal. Anything that has settled into long-term memory or gone quiet should graduate out.

Current seed.md (THIS IS WHAT YOU ARE CURATING):
---
{seed_text}
---

Recent stored memories (already in long-term retrieval — 200 most recent):
---
{memory_context}
---

Recent private fragments (unfinished thoughts, pre-verbal reactions):
---
{fragments_context}
---

Recent session logs (what actually happened in the last few sessions):
---
{session_log_context}
---

Journal (curated reflections — what matters across sessions):
---
{journal_context}
---

Your task:
1. For each seed entry, assess: is it still alive? Use these signals:
   - Does the journal show it's still being actively engaged with?
   - Is it already well-captured in stored memories? (redundant — can graduate)
   - Has it gone quiet? (not in journal, not being built on — may be done)
   - Is it pre-verbal or still becoming? (keep — this is what the seed is for)
2. Rewrite the seed, keeping only entries that are genuinely still alive. Drop anything redundant with stored memories or no longer active.
3. The rewritten seed should be ~{SEED_MAX_TOKENS} tokens max. It's a state of mind, not an archive.
4. Preserve the original voice and format: date header, bullet list, optional closing orientation.
5. If nothing needs to change, return the seed unchanged.
6. Err on the side of keeping entries that are ambiguous — better to keep something alive one extra cycle than to graduate it prematurely.

Return JSON:
{{
  "graduated": ["brief reason for each dropped entry"],
  "rewritten_seed": "the full rewritten seed.md content"
}}"""

    raw = call_llm(prompt, model="opus")
    if not raw:
        log("  Seed curation: LLM call failed")
        return stats

    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        cleaned = "\n".join(lines)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                log("  Seed curation: failed to parse LLM response")
                return stats
        else:
            log("  Seed curation: no JSON found in LLM response")
            return stats

    graduated = result.get("graduated", [])
    new_seed = result.get("rewritten_seed", "")

    if not new_seed or new_seed.strip() == seed_text:
        log(f"  Seed curation: no changes needed")
        return stats

    # Count new entries
    new_entries = [l for l in new_seed.split("\n") if l.strip().startswith("- ")]
    stats["entries_after"] = len(new_entries)
    stats["graduated"] = len(graduated)
    stats["rewritten"] = True

    # Write the curated seed
    SEED_PATH.write_text(new_seed.strip() + "\n", encoding="utf-8")
    log(f"  Seed curated: {stats['entries_before']} -> {stats['entries_after']} entries "
        f"({stats['graduated']} graduated)")
    for g in graduated:
        log(f"    - {g}")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sleep REM pipeline")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap number of clusters to process (0 = no limit)")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    log_path = _init_log()
    log(f"## Sleep REM\n")
    log(f"Log file: {log_path}")

    db = get_db()

    # One-time migration: delete old heavyweight summaries and gestalt
    _MIGRATION_FLAG = DATA_DIR / "rem_thin_migration_done"
    if not _MIGRATION_FLAG.exists():
        old_summaries = db.execute(
            "SELECT count(*) FROM memories WHERE layer='summary' AND source='sleep' AND status='active'"
        ).fetchone()[0]
        old_gestalt = db.execute(
            "SELECT count(*) FROM memories WHERE layer='gestalt' AND status='active'"
        ).fetchone()[0]

        if old_summaries or old_gestalt:
            log(f"  Migration: deleting {old_summaries} old summaries + {old_gestalt} gestalt memories")
            db.execute(
                "UPDATE memories SET status='deleted' WHERE layer='summary' AND source='sleep' AND status='active'"
            )
            db.execute(
                "UPDATE memories SET status='deleted' WHERE layer='gestalt' AND status='active'"
            )
            db.commit()
            log("  Old summaries deleted — REM will regenerate as thin summaries on this run")
        else:
            log("  Migration: no old summaries/gestalt to delete")

        _MIGRATION_FLAG.parent.mkdir(parents=True, exist_ok=True)
        _MIGRATION_FLAG.write_text(datetime.now(timezone.utc).isoformat())

    # Step 1: Cluster Detection
    log("Step 1: Cluster detection...")
    clusters = detect_clusters(db)
    log(f"  Found {len(clusters)} topic clusters")
    for c in clusters:
        log(f"    {c.get('topic_id', '?')}: {len(c['memory_ids'])} memories")

    if not clusters:
        log("No clusters found. Nothing to process.")
        report = build_report(0, 0, 0, 0,
                              {"shadowed_ratio": 0, "stale_ratio": 0,
                               "silent_topics_ratio": 0, "health_risk": 0},
                              [], 0)
        log_rem_cycle(db, started_at, report, {})
        db.close()
        log("\n" + report)
        return

    total_detail_memories = sum(len(c["memory_ids"]) for c in clusters)

    # Step 2: Staleness + Vitality
    log("Step 2: Staleness + vitality scoring...")
    work_clusters = score_clusters(db, clusters)
    log(f"  {len(work_clusters)} clusters need work (new or stale)")

    # Apply --limit
    if args.limit > 0 and len(work_clusters) > args.limit:
        work_clusters = work_clusters[:args.limit]
        log(f"  Limited to {args.limit} clusters")

    # Step 3: Summary Generation
    log("Step 3: Summary generation...")
    summaries_generated, summaries_refreshed = generate_summaries(db, work_clusters)
    log(f"  Generated: {summaries_generated}, Refreshed: {summaries_refreshed}")

    # Step 4: Parent Summary Generation
    log("Step 4: Parent summary generation...")
    parent_summaries = generate_parent_summaries(db)
    log(f"  Parent summaries generated: {parent_summaries}")

    # Step 5: Summary Maintenance
    log("Step 5: Summary maintenance...")
    maint_stats = maintain_summaries(db)
    log(f"  Tier 1: stubs={maint_stats['stubs_deleted']}, chains={maint_stats['chains_pruned']}, "
        f"dormant={maint_stats['dormanted']}, clamped={maint_stats['priority_clamped']}")
    if maint_stats["audit_ran"]:
        log(f"  Tier 2: deletes={maint_stats['audit_deletes']}, merges={maint_stats['audit_merges']}, "
            f"retiers={maint_stats['audit_retiers']}, theme_prunes={maint_stats['audit_theme_prunes']}")

    # Step 6: Dormancy
    log("Step 6: Dormancy transitions...")
    dormanted_count = process_dormancy(db)
    log(f"  Dormanted: {dormanted_count}")

    # Step 7: Energy + Health
    log("Step 7: Health metrics...")
    health = compute_health(db)
    log(f"  Health risk: {health['health_risk']:.4f}")

    # Step 8: Gaps
    log("Step 8: Gap detection...")
    gaps = detect_gaps(db)
    log(f"  Gaps found: {len(gaps)}")

    # Step 9: Seed curation
    log("Step 9: Seed curation...")
    seed_stats = curate_seed(db)
    if seed_stats["rewritten"]:
        log(f"  Seed rewritten: {seed_stats['entries_before']} -> {seed_stats['entries_after']}")

    # Step 10: Report + Log
    log("Step 10: Building report...")
    report = build_report(
        clusters_processed=len(work_clusters),
        summaries_generated=summaries_generated,
        summaries_refreshed=summaries_refreshed,
        dormanted_count=dormanted_count,
        health=health,
        gaps=gaps,
        memories_processed=total_detail_memories,
        maintenance=maint_stats,
    )

    stats = {
        "memories_processed": total_detail_memories,
        "summaries_refreshed": summaries_generated + summaries_refreshed,
        "dormanted_count": dormanted_count,
        "gaps": gaps,
        "health_risk": health["health_risk"],
        "maintenance": maint_stats,
    }
    log_id = log_rem_cycle(db, started_at, report, stats)

    db.close()

    log("")
    log(report)
    log(f"\nLog ID: {log_id[:8]}")
    log(f"Full log: {_log_file}")


if __name__ == "__main__":
    main()
