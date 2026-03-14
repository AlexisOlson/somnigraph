# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6", "openai>=2.0.0", "tiktoken>=0.7.0", "mcp[cli]>=1.2.0"]
# ///
"""
Sleep Consolidate — thematic cluster consolidation.

Detects clusters of memories about the same underlying insight via taxonomy +
weight-filtered edge BFS, sends each cluster to an LLM for judgment, and
consolidates based on the LLM's decisions (keep/archive/merge/rewrite).

Pipeline position:
    uv run sleep_nrem.py deep       # edges exist
    uv run sleep_rem.py             # topics assigned
    uv run sleep_consolidate.py     # cluster consolidation  ← this
    uv run probe_recall.py          # feedback for consolidated memories

Usage:
    uv run scripts/sleep_consolidate.py [--limit N] [--dry-run]
"""

import json
import os
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports from memory package
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from memory.constants import DATA_DIR
from memory import (
    get_db,
    count_tokens,
    embed_text,
    serialize_f32,
    _create_edge,
    update_fts,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = DATA_DIR / "sleep_logs"
PROGRESS_LOG = DATA_DIR / "sleep_progress.log"
_log_file = None


def _init_log():
    global _log_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = LOG_DIR / f"consolidate_{ts}.log"
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG.write_text("", encoding="utf-8")
    return _log_file


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    print(msg, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def log_detail(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")


# ---------------------------------------------------------------------------
# Subprocess isolation (same pattern as sleep_rem.py / sleep_nrem.py)
# ---------------------------------------------------------------------------

EMPTY_MCP_CONFIG = json.dumps({"mcpServers": {}})
_SUBPROCESS_CWD = DATA_DIR / "subprocess_workspace"

EDGE_WEIGHT_THRESHOLD = 0.4
MIN_CLUSTER_SIZE = 3
CROSS_TOPIC_EDGE_MIN = 2
MAX_ARCHIVES_PER_CLUSTER = 10
MAX_CLUSTER_TOKENS = 50_000
MAX_CLUSTER_SIZE = 20  # Don't merge beyond this — LLM can't judge 93 memories well


def _ensure_subprocess_workspace():
    git_dir = _SUBPROCESS_CWD / ".git"
    head_file = git_dir / "HEAD"
    if not head_file.exists():
        git_dir.mkdir(parents=True, exist_ok=True)
        head_file.write_text("ref: refs/heads/main\n")


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------

def call_llm(prompt_text: str, system_prompt: str = None, model: str = "opus",
             timeout: int = 600) -> str:
    """Shell out to claude -p and return stdout. Empty string on failure."""
    _ensure_subprocess_workspace()

    env = dict(os.environ)
    env.pop("CLAUDECODE", None)

    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    claude_bin = shutil.which("claude") or "claude"
    cmd = [
        claude_bin, "-p",
        "--model", model,
        "--no-session-persistence",
        "--strict-mcp-config",
        "--mcp-config", EMPTY_MCP_CONFIG,
        "--system-prompt", system_prompt or (
            "You are a JSON-only output tool. Respond with the requested "
            "JSON. No markdown fences, no commentary, no tool calls."
        ),
        "--tools", "",
        "--disable-slash-commands",
        "--setting-sources", "local",
    ]

    try:
        result = subprocess.run(
            cmd, input=prompt_text, capture_output=True, text=True,
            encoding="utf-8", timeout=timeout, cwd=str(_SUBPROCESS_CWD),
            env=env, **kwargs,
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


def parse_json_response(response: str):
    """Extract JSON object from LLM response. Returns dict or None."""
    text = response.strip()

    # Direct parse
    try:
        return json.loads(text)
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
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    # Brace matching — find outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Phase 1: Within-topic BFS clustering
# ---------------------------------------------------------------------------

def _bfs_component(adj: dict[str, set[str]], start: str, visited: set[str]) -> set[str]:
    """BFS from start, returns connected component."""
    component = set()
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        component.add(node)
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                queue.append(neighbor)
    return component


def gather_clusters(db) -> list[dict]:
    """Detect thematic clusters using taxonomy + weight-filtered edge BFS.

    Phase 1: Within-topic sub-clusters at edge weight >= 0.4
    Phase 2: Merge sub-clusters sharing 2+ cross-topic edges

    Returns clusters with 3+ members, sorted by size desc.
    """
    # Ensure last_consolidated column exists
    cols = {r["name"] for r in db.execute("PRAGMA table_info(memories)").fetchall()}
    if "last_consolidated" not in cols:
        db.execute("ALTER TABLE memories ADD COLUMN last_consolidated TEXT DEFAULT NULL")
        db.commit()

    # Load all active detail-layer memories with a topic_id
    memories = db.execute(
        "SELECT * FROM memories WHERE status = 'active' AND layer = 'detail' "
        "AND topic_id IS NOT NULL"
    ).fetchall()

    if not memories:
        return []

    mem_by_id = {m["id"]: m for m in memories}

    # Group memories by topic_id
    by_topic = defaultdict(list)
    for m in memories:
        by_topic[m["topic_id"]].append(m["id"])

    # Load all edges with weight >= threshold between active memories
    edges = db.execute(
        "SELECT source_id, target_id, weight, linking_context FROM memory_edges "
        "WHERE weight >= ?",
        (EDGE_WEIGHT_THRESHOLD,),
    ).fetchall()

    # Also load ALL edges (any weight) for cross-topic merge detection
    all_edges = db.execute(
        "SELECT source_id, target_id, weight, linking_context FROM memory_edges"
    ).fetchall()

    # Build within-topic adjacency at threshold
    topic_adj = defaultdict(lambda: defaultdict(set))  # topic -> {node -> {neighbors}}
    for e in edges:
        sid, tid = e["source_id"], e["target_id"]
        if sid not in mem_by_id or tid not in mem_by_id:
            continue
        s_topic = mem_by_id[sid]["topic_id"]
        t_topic = mem_by_id[tid]["topic_id"]
        if s_topic == t_topic:
            topic_adj[s_topic][sid].add(tid)
            topic_adj[s_topic][tid].add(sid)

    # Phase 1: BFS within each topic
    sub_clusters = []  # list of sets of memory_ids
    sub_cluster_topics = []  # parallel list: topic_id for each sub-cluster

    for topic_id, member_ids in by_topic.items():
        adj = topic_adj[topic_id]
        visited = set()
        for mid in member_ids:
            if mid not in visited and mid in adj:
                component = _bfs_component(adj, mid, visited)
                if len(component) >= MIN_CLUSTER_SIZE:
                    sub_clusters.append(component)
                    sub_cluster_topics.append(topic_id)

    if not sub_clusters:
        return []

    log(f"  Phase 1: {len(sub_clusters)} within-topic sub-clusters")

    # Phase 2: Cross-topic merge
    # Build lookup: memory_id -> sub_cluster index
    mem_to_sc = {}
    for i, sc in enumerate(sub_clusters):
        for mid in sc:
            mem_to_sc[mid] = i

    # Count cross-topic edges between sub-cluster pairs
    cross_counts = defaultdict(int)  # (sc_i, sc_j) -> count
    for e in all_edges:
        sid, tid = e["source_id"], e["target_id"]
        si = mem_to_sc.get(sid)
        ti = mem_to_sc.get(tid)
        if si is not None and ti is not None and si != ti:
            pair = (min(si, ti), max(si, ti))
            cross_counts[pair] += 1

    # Union-find for merging (with size tracking to cap cluster size)
    parent = list(range(len(sub_clusters)))
    size = [len(sc) for sc in sub_clusters]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # Skip if merged cluster would exceed max size
            if size[ra] + size[rb] > MAX_CLUSTER_SIZE:
                return False
            # Union by size (larger becomes root)
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
            return True
        return False

    merge_count = 0
    # Sort by edge count descending so strongest connections merge first
    for (i, j), count in sorted(cross_counts.items(), key=lambda x: x[1], reverse=True):
        if count >= CROSS_TOPIC_EDGE_MIN:
            if union(i, j):
                merge_count += 1

    if merge_count:
        log(f"  Phase 2: {merge_count} cross-topic merges")

    # Build final clusters from union-find groups
    groups = defaultdict(set)
    for i, sc in enumerate(sub_clusters):
        root = find(i)
        groups[root].update(sc)

    # Build cluster dicts
    clusters = []
    for root, member_ids in groups.items():
        if len(member_ids) < MIN_CLUSTER_SIZE:
            continue

        cluster_memories = [mem_by_id[mid] for mid in member_ids if mid in mem_by_id]
        topic_ids = list(set(m["topic_id"] for m in cluster_memories))

        # Gather internal edges
        internal_edges = []
        for e in all_edges:
            if e["source_id"] in member_ids and e["target_id"] in member_ids:
                internal_edges.append(dict(e))

        # Rough token estimate (avoid loading tiktoken during gather — saves memory)
        total_tokens = sum(len(m["content"]) // 4 for m in cluster_memories)

        clusters.append({
            "topic_ids": topic_ids,
            "memory_ids": list(member_ids),
            "memories": cluster_memories,
            "edges": internal_edges,
            "total_tokens": total_tokens,
        })

    # Filter out clusters where all members are already consolidated and unchanged
    fresh_clusters = []
    skipped_stable = 0
    for c in clusters:
        needs_work = False
        for m in c["memories"]:
            lc = m["last_consolidated"]
            if not lc:
                needs_work = True
                break
            # If memory was created or modified after last consolidation, reprocess
            if m["created_at"] > lc:
                needs_work = True
                break
            # Check if last_sleep_processed is newer (new edges since consolidation)
            if m["last_sleep_processed"] and m["last_sleep_processed"] > lc:
                needs_work = True
                break
        if needs_work:
            fresh_clusters.append(c)
        else:
            skipped_stable += 1

    if skipped_stable:
        log(f"  Skipped {skipped_stable} stable clusters (already consolidated)")

    # Sort by size descending
    fresh_clusters.sort(key=lambda c: len(c["memory_ids"]), reverse=True)

    return fresh_clusters


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

CONSOLIDATION_SYSTEM = """\
You are a memory consolidation agent. You receive a cluster of thematically
related memories and decide which are genuinely distinct vs redundant.

REDUNDANCY TYPES:
1. True paraphrases: same info in different words — archive the weaker version
2. Journal digests mentioning an insight captured standalone elsewhere — archive digest
3. Multiple versions of an evolving understanding — keep latest, archive earlier
4. Narrative arcs (temporal sequences like ordering→shipping→arrival): NOT redundant.
   Each step has unique texture. Use "annotate" to link them into an arc, not archive.
5. Incident + codified pattern: if the incident ONLY matters as an instance of the
   pattern, archive it. If it has its own texture/story, annotate it into the arc.

ACTIONS (per memory):
- "keep": distinct and valuable, no changes
- "archive": redundant; specify which memory_id captures its info via superseded_by
- "merge_into": fold this memory's unique content into target; provide target_id + content to add
- "rewrite": this memory becomes the cluster representative with updated content incorporating archived members; provide new content + summary
- "annotate": for narrative arcs/sequences — append arc context to this memory showing where it sits in the sequence (prev/next memories, brief arc summary). Use this instead of archive when memories form a temporal arc where each step has unique texture worth preserving. Provide arc_context (text to append).

FORMATION:
If the cluster core represents a formative pattern (learned through experience, shaped
subsequent behavior), mark it formation=true on the representative memory.

SAFETY:
- Archive means GONE FROM RECALL. If someone later asks a question whose answer
  is only in a memory you archived, that's a failure. Before archiving, verify every
  retrievable fact (dates, names, specific details, emotional moments) exists in the
  keeper. If not, use merge_into to fold those facts into the keeper first.
- When unsure, keep
- source="correction" memories should almost always be kept
- PROTECTED memories (marked below) CANNOT be archived — only keep or rewrite
- Prefer merge_into over archive when the source has ANY unique detail worth recalling
- Prefer keeping 2-3 strong memories over aggressively merging to 1

Respond with a JSON object. No markdown fences, no commentary — JSON only.

Output schema:
{
  "reasoning": "Brief cluster analysis",
  "actions": [
    {
      "memory_id": "full-uuid",
      "action": "keep|archive|merge_into|rewrite|annotate",
      "superseded_by": "full-uuid of the memory that captures this one's info (REQUIRED for archive)",
      "target_id": "full-uuid of the memory to merge into (REQUIRED for merge_into)",
      "merge_content": "text to append to target (REQUIRED for merge_into)",
      "new_content": "full rewritten content (REQUIRED for rewrite)",
      "new_summary": "new summary (REQUIRED for rewrite)",
      "arc_context": "arc position text to append (REQUIRED for annotate)",
      "formation": false
    }
  ]
}"""


def _is_protected(mem) -> bool:
    """Check if a memory is protected from archival."""
    if mem["decay_rate"] is not None and mem["decay_rate"] == 0:
        return True
    if mem["source"] == "correction":
        return True
    try:
        flags_raw = mem["flags"]
    except (IndexError, KeyError):
        flags_raw = None
    flags = json.loads(flags_raw) if flags_raw else []
    if "pinned" in flags or "keep" in flags:
        return True
    return False


def prepare_prompt(db, cluster: dict) -> tuple[str, dict]:
    """Format cluster for LLM with rich context.

    Includes: cluster memories, internal edges, summary-layer memories
    for the topic(s), and other active memories in the same topic(s)
    (not in the cluster) so the LLM can judge what info exists elsewhere.

    Respects 50k token budget (rough char/4 estimate).
    Returns (prompt_text, stats_dict).
    """
    lines = []
    lines.append(f"CLUSTER ({len(cluster['memories'])} memories, "
                 f"topics: {', '.join(cluster['topic_ids'])})")
    lines.append("")

    cluster_ids = set(cluster["memory_ids"])
    token_budget = MAX_CLUSTER_TOKENS
    prompt_stats = {"cluster_tokens": 0, "summary_tokens": 0, "sibling_tokens": 0,
                    "summaries_count": 0, "siblings_count": 0, "siblings_truncated": 0}

    # --- Section 1: Cluster memories (full content) ---
    lines.append("=== CLUSTER MEMORIES (decide on these) ===")
    lines.append("")

    sorted_mems = sorted(cluster["memories"], key=lambda m: m["created_at"])
    included = []

    for mem in sorted_mems:
        mem_text = _format_memory(mem)
        mem_tokens = len(mem_text) // 4
        if token_budget - mem_tokens < 0 and included:
            lines.append(f"\n[{len(sorted_mems) - len(included)} memories omitted for budget]")
            break
        token_budget -= mem_tokens
        prompt_stats["cluster_tokens"] += mem_tokens
        included.append(mem)
        lines.append(mem_text)
        lines.append("")

    # --- Section 2: Topic summaries (summary-layer memories) ---
    topic_placeholders = ",".join("?" for _ in cluster["topic_ids"])
    summaries = db.execute(
        f"SELECT id, content, summary, topic_id FROM memories "
        f"WHERE status = 'active' AND layer = 'summary' "
        f"AND topic_id IN ({topic_placeholders})",
        cluster["topic_ids"],
    ).fetchall()

    if summaries:
        lines.append("=== TOPIC SUMMARIES (for context — do NOT act on these) ===")
        lines.append("")
        for s in summaries:
            text = f"TOPIC SUMMARY [{s['topic_id']}]: {s['content']}"
            text_tokens = len(text) // 4
            if token_budget - text_tokens < 0:
                break
            token_budget -= text_tokens
            prompt_stats["summary_tokens"] += text_tokens
            prompt_stats["summaries_count"] += 1
            lines.append(text)
            lines.append("")

    # --- Section 3: Other active memories in same topic(s) ---
    # Sorted by relevance: edges to cluster members, then feedback utility
    others_full = db.execute(
        f"SELECT * FROM memories WHERE status = 'active' AND layer = 'detail' "
        f"AND topic_id IN ({topic_placeholders})",
        cluster["topic_ids"],
    ).fetchall()

    non_cluster = [m for m in others_full if m["id"] not in cluster_ids]

    if non_cluster:
        # Count edges from each sibling to cluster members (single query)
        edge_counts = defaultdict(int)
        cid_ph = ",".join("?" for _ in cluster_ids)
        cid_list = list(cluster_ids)
        rows = db.execute(
            f"SELECT source_id, target_id FROM memory_edges "
            f"WHERE source_id IN ({cid_ph}) OR target_id IN ({cid_ph})",
            cid_list + cid_list,
        ).fetchall()
        sibling_ids = {m["id"] for m in non_cluster}
        for r in rows:
            if r["source_id"] in cluster_ids and r["target_id"] in sibling_ids:
                edge_counts[r["target_id"]] += 1
            elif r["target_id"] in cluster_ids and r["source_id"] in sibling_ids:
                edge_counts[r["source_id"]] += 1

        # Get feedback utility per sibling
        sibling_utility = {}
        for m in non_cluster:
            fbs = db.execute(
                "SELECT json_extract(context, '$.utility') as u "
                "FROM memory_events WHERE memory_id = ? AND event_type = 'feedback'",
                (m["id"],),
            ).fetchall()
            utils = [f["u"] for f in fbs if f["u"] is not None]
            sibling_utility[m["id"]] = sum(utils) / len(utils) if utils else 0.0

        # Sort: most edges to cluster first, then highest utility
        non_cluster.sort(
            key=lambda m: (edge_counts.get(m["id"], 0), sibling_utility.get(m["id"], 0.0)),
            reverse=True,
        )
    if non_cluster:
        lines.append("=== OTHER MEMORIES IN SAME TOPIC(S) (for context — do NOT act on these) ===")
        lines.append("These show what info already exists outside the cluster.")
        lines.append("")
        for m in non_cluster:
            text = _format_memory(m)
            text_tokens = len(text) // 4
            if token_budget - text_tokens < 0:
                prompt_stats["siblings_truncated"] = len(non_cluster) - non_cluster.index(m)
                lines.append(f"  ... and {prompt_stats['siblings_truncated']} more (budget exhausted)")
                break
            token_budget -= text_tokens
            prompt_stats["sibling_tokens"] += text_tokens
            prompt_stats["siblings_count"] += 1
            lines.append(text)
            lines.append("")

    # --- Section 4: Internal edges ---
    if cluster["edges"]:
        lines.append("=== INTERNAL EDGES ===")
        for e in cluster["edges"]:
            ctx = (e.get("linking_context") or "")[:100]
            lines.append(f"  {e['source_id'][:8]} <-> {e['target_id'][:8]} "
                         f"(weight={e.get('weight', '?')}) {ctx}")
        lines.append("")

    lines.append("OUTPUT: Respond with ONLY a JSON object. No prose, no markdown fences, no commentary.")
    lines.append("Use FULL memory UUIDs (not 8-char prefixes) in memory_id, superseded_by, and target_id fields.")
    lines.append('{"reasoning": "...", "actions": [{"memory_id": "full-uuid", "action": "keep|archive|merge_into|rewrite|annotate", ...}, ...]}')

    return "\n".join(lines), prompt_stats


def _format_memory(mem) -> str:
    """Format a single memory for the cluster prompt."""
    protected = _is_protected(mem)
    themes = json.loads(mem["themes"]) if mem["themes"] else []
    try:
        flags_raw = mem["flags"]
    except (IndexError, KeyError):
        flags_raw = None
    flags = json.loads(flags_raw) if flags_raw else []

    header = f"MEMORY {mem['id']}"
    if protected:
        header += " [PROTECTED — cannot be archived]"

    lines = [
        header,
        f"  Created: {mem['created_at'][:10]} | Category: {mem['category']} | "
        f"Source: {mem['source'] or 'session'} | Priority: {mem['base_priority']}",
    ]
    if themes:
        lines.append(f"  Themes: {', '.join(themes)}")
    if flags:
        lines.append(f"  Flags: {', '.join(flags)}")
    if mem["decay_rate"] is not None:
        lines.append(f"  Decay rate: {mem['decay_rate']}")
    if mem["summary"]:
        lines.append(f"  Summary: {mem['summary']}")
    lines.append(f"  Content: {mem['content']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM judgment
# ---------------------------------------------------------------------------

def judge_cluster(db, cluster: dict) -> dict | None:
    """Send cluster to LLM for consolidation judgment. Returns parsed actions or None."""
    prompt, pstats = prepare_prompt(db, cluster)

    trunc = f", {pstats['siblings_truncated']} truncated" if pstats['siblings_truncated'] else ""
    log(f"  Prompt: ~{pstats['cluster_tokens']}t cluster, "
        f"~{pstats['summary_tokens']}t summaries ({pstats['summaries_count']}), "
        f"~{pstats['sibling_tokens']}t siblings ({pstats['siblings_count']}{trunc})")
    log_detail(f"PROMPT for cluster ({len(cluster['memories'])} mems):\n{prompt}")

    raw = call_llm(prompt, system_prompt=CONSOLIDATION_SYSTEM, timeout=600)
    if not raw:
        return None

    log_detail(f"RESPONSE:\n{raw}")

    judgment = parse_json_response(raw)
    if not judgment or not isinstance(judgment, dict):
        log(f"  WARNING: failed to parse judgment JSON")
        return None

    if "actions" not in judgment:
        log(f"  WARNING: judgment missing 'actions' key")
        return None

    return judgment


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

def archive_memory(db, memory_id: str, superseded_by: str):
    """Set deleted, clean vec/FTS/rowid, create derivation edge."""
    db.execute(
        "UPDATE memories SET status = 'deleted', superseded_by = ? WHERE id = ?",
        (superseded_by, memory_id),
    )

    # Clean up vec, FTS, and rowid mapping
    vec_map = db.execute(
        "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    if vec_map:
        db.execute("DELETE FROM memory_vec WHERE rowid = ?", (vec_map["rowid"],))
        db.execute("DELETE FROM memory_fts WHERE rowid = ?", (vec_map["rowid"],))
        db.execute("DELETE FROM memory_rowid_map WHERE rowid = ?", (vec_map["rowid"],))

    # Create derivation edge from superseding -> archived
    _create_edge(
        db, superseded_by, memory_id,
        linking_context="consolidated: archived as redundant",
        flags=["derivation"],
        created_by="consolidate",
    )


def rewrite_memory(db, memory_id: str, new_content: str, new_summary: str):
    """Update content/summary, re-embed, update FTS."""
    db.execute(
        "UPDATE memories SET content = ?, summary = ?, token_count = ? WHERE id = ?",
        (new_content, new_summary, count_tokens(new_content), memory_id),
    )

    # Re-embed
    embedding = embed_text(new_content)
    vec_map = db.execute(
        "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    if vec_map:
        db.execute(
            "UPDATE memory_vec SET embedding = ? WHERE rowid = ?",
            (serialize_f32(embedding), vec_map["rowid"]),
        )

    # Update FTS
    update_fts(db, memory_id)


def tag_formation(db, memory_id: str):
    """Set decay_rate=0, add 'formation' theme."""
    db.execute("UPDATE memories SET decay_rate = 0 WHERE id = ?", (memory_id,))

    # Add 'formation' to themes
    row = db.execute("SELECT themes FROM memories WHERE id = ?", (memory_id,)).fetchone()
    themes = json.loads(row["themes"]) if row and row["themes"] else []
    if "formation" not in themes:
        themes.append("formation")
        db.execute(
            "UPDATE memories SET themes = ? WHERE id = ?",
            (json.dumps(themes), memory_id),
        )
        update_fts(db, memory_id)


def execute_actions(db, cluster: dict, judgment: dict, dry_run: bool = False,
                    verbose: bool = False) -> dict:
    """Apply keep/archive/merge_into/rewrite per memory.
    Returns stats {archived, merged, rewritten, formations, kept, skipped}."""
    stats = {"archived": 0, "merged": 0, "rewritten": 0, "formations": 0,
             "kept": 0, "rejected": 0, "blocked": 0, "details": []}

    mem_by_id = {m["id"]: m for m in cluster["memories"]}
    # Build prefix lookup: LLM often returns 8-char prefixes instead of full UUIDs
    prefix_to_full = {}
    for full_id in mem_by_id:
        prefix_to_full[full_id[:8]] = full_id
        prefix_to_full[full_id] = full_id  # full ID maps to itself

    def resolve_id(raw_id: str) -> str:
        """Resolve a full UUID or 8-char prefix to the full UUID."""
        if not raw_id:
            return ""
        return prefix_to_full.get(raw_id, prefix_to_full.get(raw_id[:8], ""))

    actions = judgment.get("actions", [])
    archive_count = 0

    for action_item in actions:
        mid = resolve_id(action_item.get("memory_id", ""))
        action = action_item.get("action", "keep")
        formation = action_item.get("formation", False)

        if mid not in mem_by_id:
            stats["rejected"] += 1
            continue

        mem = mem_by_id[mid]

        # Verify memory is still active
        live = db.execute(
            "SELECT status FROM memories WHERE id = ? AND status = 'active'", (mid,)
        ).fetchone()
        if not live:
            stats["blocked"] += 1
            stats["details"].append(f"  BLOCKED {mid[:8]}: no longer active")
            continue

        reason = action_item.get("reason", "")

        if action == "keep":
            stats["kept"] += 1
            if verbose:
                label = mem["summary"] or mem["content"][:80]
                stats["details"].append(f"  KEEP {mid[:8]}: {label}")
                if reason:
                    stats["details"].append(f"       reason: {reason}")
            if formation and not dry_run:
                tag_formation(db, mid)
                stats["formations"] += 1
                stats["details"].append(f"  FORMATION {mid[:8]}: {(mem['summary'] or '')[:60]}")

        elif action == "archive":
            superseded_by = resolve_id(action_item.get("superseded_by", ""))
            # Fallback: LLM sometimes puts the target ID in "reason" text instead
            if not superseded_by or superseded_by not in mem_by_id:
                for candidate_id in mem_by_id:
                    if candidate_id[:8] in reason or candidate_id in reason:
                        if candidate_id != mid:  # don't supersede self
                            superseded_by = candidate_id
                            break
            if not superseded_by or superseded_by not in mem_by_id:
                stats["rejected"] += 1
                stats["details"].append(f"  REJECTED archive {mid[:8]}: invalid superseded_by")
                if verbose and reason:
                    stats["details"].append(f"       reason: {reason}")
                continue
            if _is_protected(mem):
                stats["blocked"] += 1
                stats["details"].append(f"  BLOCKED archive {mid[:8]}: protected")
                continue
            if archive_count >= MAX_ARCHIVES_PER_CLUSTER:
                stats["blocked"] += 1
                stats["details"].append(f"  BLOCKED archive {mid[:8]}: max archives reached")
                continue

            label = mem["summary"] or mem["content"][:60]
            sup_label = mem_by_id[superseded_by]["summary"] or mem_by_id[superseded_by]["content"][:60]
            stats["details"].append(f"  ARCHIVE {mid[:8]}: {label}")
            stats["details"].append(f"       -> superseded by {superseded_by[:8]}: {sup_label}")
            if verbose and reason:
                stats["details"].append(f"       reason: {reason}")
            if not dry_run:
                archive_memory(db, mid, superseded_by)
            stats["archived"] += 1
            archive_count += 1

        elif action == "merge_into":
            target_id = resolve_id(action_item.get("target_id", ""))
            merge_content = action_item.get("merge_content", "")
            if not target_id or target_id not in mem_by_id:
                stats["rejected"] += 1
                stats["details"].append(f"  REJECTED merge {mid[:8]}: invalid target_id")
                continue
            if not merge_content:
                stats["rejected"] += 1
                stats["details"].append(f"  REJECTED merge {mid[:8]}: no merge_content")
                continue
            if _is_protected(mem):
                stats["blocked"] += 1
                stats["details"].append(f"  BLOCKED merge {mid[:8]}: protected")
                continue
            if archive_count >= MAX_ARCHIVES_PER_CLUSTER:
                stats["blocked"] += 1
                stats["details"].append(f"  BLOCKED merge {mid[:8]}: max archives reached")
                continue

            label = mem["summary"] or mem["content"][:60]
            target_label = mem_by_id[target_id]["summary"] or mem_by_id[target_id]["content"][:60]
            stats["details"].append(f"  MERGE {mid[:8]} -> {target_id[:8]}: {label}")
            stats["details"].append(f"       into: {target_label}")
            if verbose:
                preview = merge_content[:200].replace("\n", " ")
                stats["details"].append(f"       adding: {preview}")
                if reason:
                    stats["details"].append(f"       reason: {reason}")

            if not dry_run:
                # Append merge_content to target
                target_mem = db.execute(
                    "SELECT content FROM memories WHERE id = ?", (target_id,)
                ).fetchone()
                if target_mem:
                    new_content = target_mem["content"] + "\n\n" + merge_content
                    rewrite_memory(db, target_id, new_content,
                                   mem_by_id[target_id]["summary"] or "")
                # Archive source
                archive_memory(db, mid, target_id)

            stats["merged"] += 1
            archive_count += 1

        elif action == "rewrite":
            new_content = action_item.get("new_content", "")
            new_summary = action_item.get("new_summary", "")
            if not new_content:
                stats["rejected"] += 1
                stats["details"].append(f"  REJECTED rewrite {mid[:8]}: no new_content")
                continue

            label = mem["summary"] or mem["content"][:60]
            stats["details"].append(f"  REWRITE {mid[:8]}: {label}")
            if new_summary:
                stats["details"].append(f"       new summary: {new_summary}")
            if verbose:
                preview = new_content[:300].replace("\n", " ")
                stats["details"].append(f"       new content: {preview}...")
                if reason:
                    stats["details"].append(f"       reason: {reason}")

            if not dry_run:
                rewrite_memory(db, mid, new_content, new_summary or mem["summary"] or "")

                # Create derivation edges from rewritten memory to all other cluster members
                for other_mid in mem_by_id:
                    if other_mid != mid:
                        _create_edge(
                            db, mid, other_mid,
                            linking_context="consolidated: rewritten as cluster representative",
                            flags=["derivation"],
                            created_by="consolidate",
                        )

            stats["rewritten"] += 1
            if formation and not dry_run:
                tag_formation(db, mid)
                stats["formations"] += 1
                stats["details"].append(f"  FORMATION {mid[:8]}")

        elif action == "annotate":
            arc_context = action_item.get("arc_context", "")
            if not arc_context:
                stats["rejected"] += 1
                stats["details"].append(f"  REJECTED annotate {mid[:8]}: no arc_context")
                continue

            label = mem["summary"] or mem["content"][:60]
            stats["details"].append(f"  ANNOTATE {mid[:8]}: {label}")
            if verbose:
                stats["details"].append(f"       arc: {arc_context[:200]}")
                if reason:
                    stats["details"].append(f"       reason: {reason}")

            if not dry_run:
                # Append arc context to content
                current = db.execute(
                    "SELECT content FROM memories WHERE id = ?", (mid,)
                ).fetchone()
                if current:
                    new_content = current["content"] + "\n\n" + arc_context
                    db.execute(
                        "UPDATE memories SET content = ?, token_count = ? WHERE id = ?",
                        (new_content, count_tokens(new_content), mid),
                    )
                    # Re-embed with arc context included
                    embedding = embed_text(new_content)
                    vec_map = db.execute(
                        "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (mid,)
                    ).fetchone()
                    if vec_map:
                        db.execute(
                            "UPDATE memory_vec SET embedding = ? WHERE rowid = ?",
                            (serialize_f32(embedding), vec_map["rowid"]),
                        )
                    update_fts(db, mid)

            stats["annotated"] = stats.get("annotated", 0) + 1
            if formation and not dry_run:
                tag_formation(db, mid)
                stats["formations"] += 1
                stats["details"].append(f"  FORMATION {mid[:8]}")

    return stats


# ---------------------------------------------------------------------------
# Post-run integrity checks
# ---------------------------------------------------------------------------

def verify_integrity(db) -> list[str]:
    """Check for integrity issues after consolidation. Returns list of warnings."""
    warnings = []

    # Active memory with superseded_by pointing to another active memory
    bad = db.execute(
        "SELECT m1.id, m1.superseded_by FROM memories m1 "
        "JOIN memories m2 ON m1.superseded_by = m2.id "
        "WHERE m1.status = 'active' AND m2.status = 'active' "
        "AND m1.superseded_by IS NOT NULL"
    ).fetchall()
    for row in bad:
        warnings.append(f"Active memory {row['id'][:8]} has superseded_by "
                        f"pointing to active {row['superseded_by'][:8]}")

    # Orphan edges (both endpoints deleted)
    orphans = db.execute(
        "SELECT e.id FROM memory_edges e "
        "JOIN memories m1 ON e.source_id = m1.id "
        "JOIN memories m2 ON e.target_id = m2.id "
        "WHERE m1.status = 'deleted' AND m2.status = 'deleted'"
    ).fetchall()
    if orphans:
        warnings.append(f"{len(orphans)} edges with both endpoints deleted")

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    import time as _time

    parser = argparse.ArgumentParser(description="Sleep Consolidate — thematic cluster consolidation")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max clusters to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without executing")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed per-action output (reasons, content previews)")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    log_path = _init_log()
    log("## Sleep Consolidate\n")
    log(f"Log file: {log_path}")
    if args.dry_run:
        log("MODE: dry-run (no changes will be made)")

    db = get_db()

    # Step 1: Gather clusters
    log("Gathering clusters...")
    clusters = gather_clusters(db)
    if not clusters:
        log("No clusters found (3+ members at weight >= 0.4). Nothing to consolidate.")
        db.close()
        return

    total_mems = sum(len(c["memory_ids"]) for c in clusters)
    log(f"Found {len(clusters)} clusters ({total_mems} total memories)")
    for i, c in enumerate(clusters):
        log(f"  Cluster {i+1}: {len(c['memory_ids'])} members, "
            f"topics={c['topic_ids']}, ~{c['total_tokens']} tokens")

    if args.limit > 0:
        clusters = clusters[:args.limit]
        log(f"Limited to {len(clusters)} clusters")

    # Step 2: Process each cluster
    totals = {"archived": 0, "merged": 0, "rewritten": 0, "annotated": 0,
              "formations": 0, "kept": 0, "rejected": 0, "blocked": 0, "failed": 0}
    all_details = []

    _t_start = _time.monotonic()
    for i, cluster in enumerate(clusters):
        log(f"\nCluster {i+1}/{len(clusters)} "
            f"({len(cluster['memory_ids'])} members, topics={cluster['topic_ids']})...")

        judgment = judge_cluster(db, cluster)
        if judgment is None:
            log(f"  FAILED: no judgment returned")
            totals["failed"] += 1
            continue

        reasoning = judgment.get("reasoning", "")
        if reasoning:
            log(f"  Reasoning: {reasoning[:120]}")

        stats = execute_actions(db, cluster, judgment, dry_run=args.dry_run,
                                verbose=args.verbose)

        for key in ("archived", "merged", "rewritten", "annotated", "formations", "kept", "rejected", "blocked"):
            totals[key] += stats[key]

        for detail in stats["details"]:
            log(detail)
            all_details.append(detail)

        log(f"  Result: {stats['archived']} archived, {stats['merged']} merged, "
            f"{stats['rewritten']} rewritten, {stats.get('annotated', 0)} annotated, "
            f"{stats['kept']} kept, {stats['formations']} formations")

        if not args.dry_run:
            # Stamp all cluster members as consolidated
            now = datetime.now(timezone.utc).isoformat()
            for mid in cluster["memory_ids"]:
                db.execute(
                    "UPDATE memories SET last_consolidated = ? "
                    "WHERE id = ? AND status = 'active'",
                    (now, mid),
                )
            db.commit()

    elapsed = _time.monotonic() - _t_start

    # Step 3: Integrity check
    if not args.dry_run:
        warnings = verify_integrity(db)
        if warnings:
            log("\nIntegrity warnings:")
            for w in warnings:
                log(f"  {w}")

    # Step 4: Log sleep cycle
    if not args.dry_run and (totals["archived"] or totals["merged"] or totals["rewritten"]):
        now = datetime.now(timezone.utc).isoformat()
        log_id = str(uuid.uuid4())
        report = (
            f"Consolidate: {len(clusters)} clusters, "
            f"{totals['archived']} archived, {totals['merged']} merged, "
            f"{totals['rewritten']} rewritten, {totals['formations']} formations"
        )
        db.execute(
            """INSERT INTO sleep_log
               (id, started_at, completed_at, mode, memories_processed,
                relationships_found, summaries_refreshed, contradictions_flagged,
                fast_path_count, full_pipeline_count, memories_pruned,
                per_memory_changes, report)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (log_id, started_at, now, "consolidate",
             total_mems, 0, 0, 0, 0, len(clusters),
             totals["archived"] + totals["merged"],
             json.dumps([]), report),
        )
        db.commit()

    db.close()

    # Step 5: Report
    log(f"\n## Consolidation Report")
    log(f"- Clusters processed: {len(clusters)}")
    log(f"- Archived: {totals['archived']}")
    log(f"- Merged: {totals['merged']}")
    log(f"- Rewritten: {totals['rewritten']}")
    log(f"- Annotated: {totals['annotated']}")
    log(f"- Formations tagged: {totals['formations']}")
    log(f"- Kept: {totals['kept']}")
    log(f"- Rejected: {totals['rejected']} (schema errors)")
    log(f"- Blocked: {totals['blocked']} (safety catches)")
    log(f"- Failed clusters: {totals['failed']}")
    log(f"- Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    if args.dry_run:
        log("\n(dry-run — no changes were made)")


if __name__ == "__main__":
    main()
