# Claude Cognitive -- Source Analysis

*Phase 13, 2026-03-05. Analysis of GMaN1911/claude-cognitive.*

## 1. Architecture Overview

**Language:** Python 3 (stdlib only -- zero external dependencies)

**Paradigm:** Not a store/recall memory system. This is an attention-based context router that dynamically injects documentation files into Claude Code's system prompt via hooks. Files are classified by relevance score into HOT/WARM/COLD tiers, with different injection strategies per tier.

**File Structure (v1.x production):**
```
scripts/
  context-router-v2.py     # Core: attention scoring + context injection
  history.py                # CLI: query attention_history.jsonl
  pool-auto-update.py       # Hook: extract coordination signals from conversation
  pool-loader.py            # Hook: inject pool state at session start
  pool-extractor.py         # Hook: extract explicit pool blocks on Stop
  pool-query.py             # CLI: query pool state
  usage_tracker.py          # v1.2 prototype: learn from file access patterns
  usage-track-stop.py       # v1.2 prototype: Stop hook for usage tracking
  add-usage-tracking-hook.py # v1.2 installer script
```

**v2.0 (release candidate, separate directory):** Replaces manual keywords.json with auto-discovered DAG via a separate `hologram-cognitive` library. Uses content-addressed coordinates, edge-weighted injection, and BFS hop decay. Not yet production -- lives in `v2.0/` folder as docs/examples only (the actual hologram engine is a separate repo).

**How it fits into Claude Code:** Three hooks registered in `hooks-config.json`:
- `UserPromptSubmit` -> context-router-v2.py + pool-auto-update.py
- `SessionStart` -> pool-loader.py
- `Stop` -> pool-extractor.py

The router reads from stdin (JSON with `prompt` field), computes attention scores, reads .md files from a `.claude/` directory, and prints tiered context to stdout. Claude Code injects this output into the conversation.

---

## 2. Attention-Based Context Router

### Scoring Formula

The attention system is a three-phase update per turn:

```python
# Phase 1: DECAY all scores
for path in state["scores"]:
    score[path] *= get_decay_rate(path)  # multiplicative decay

# Phase 2: KEYWORD ACTIVATION (binary -- any match sets score to 1.0)
for path, keywords in KEYWORDS.items():
    if any(kw in prompt_lower for kw in keywords):
        score[path] = 1.0  # KEYWORD_BOOST = 1.0

# Phase 3: CO-ACTIVATION (additive boost, capped at 1.0)
for activated_path in directly_activated:
    for related_path in CO_ACTIVATION[activated_path]:
        score[related_path] = min(1.0, score[related_path] + 0.35)

# Phase 4: PINNED FILE FLOOR
for pinned in PINNED_FILES:
    score[pinned] = max(score[pinned], 0.35)  # WARM_THRESHOLD + 0.1
```

Key observation: keyword activation is binary (exact substring match in lowercased prompt), not weighted. A match instantly sets score to 1.0 regardless of previous score. This is intentionally simple -- no TF-IDF, no embeddings, no fuzzy matching.

### HOT / WARM / COLD Classification

| Tier | Score Range | Injection | Max Files | Token Cost |
|------|-------------|-----------|-----------|------------|
| HOT  | >= 0.8      | Full file content | 4 | 2,500-5,000 tokens/file |
| WARM | 0.25 - 0.8  | First 25 lines (headers) | 8 | 200-500 tokens/file |
| COLD | < 0.25      | Nothing (evicted) | unlimited | 0 |

Hard ceiling: `MAX_TOTAL_CHARS = 25,000` (~6.25K tokens). If a HOT file would exceed the budget, it's demoted to WARM.

### Decay Function

Multiplicative exponential decay per turn. Rate varies by file path prefix:

```python
DECAY_RATES = {
    "systems/":       0.85,  # Infrastructure: slow decay (half-life ~4.3 turns)
    "modules/":       0.70,  # Code: fast decay (half-life ~2.0 turns)
    "integrations/":  0.80,  # APIs: medium decay
    "docs/":          0.75,
    "default":        0.70
}
```

Half-life formula: `h = ln(0.5) / ln(rate)`. So a module at score 1.0 with no re-activation crosses WARM threshold (0.25) in about 4 turns, and crosses COLD threshold after ~5 turns. Systems files persist ~8 turns before going COLD.

This is NOT memory decay -- there's no long-term storage being degraded. It's attention decay: how quickly a file becomes irrelevant when the conversation moves to other topics. The file itself is unchanged; only the injection score decays.

### Co-Activation

Hardcoded or configured graph of related files. When file A is directly activated by keywords, all files in `CO_ACTIVATION[A]` get a +0.35 boost (capped at 1.0). This is one-hop spreading activation -- no multi-hop propagation in v1.x.

The co-activation graph is manually curated in `keywords.json` or hardcoded in the script. Example: activating `systems/orin.md` co-activates `integrations/pipe-to-orin.md`, `modules/t3-telos.md`, and `modules/ppe-anticipatory-coherence.md`.

### Keyword Activation

`keywords.json` maps file paths to trigger words:

```json
{
  "keywords": {
    "modules/auth.md": ["authentication", "login", "token", "oauth"],
    "systems/backend.md": ["backend", "server", "api"]
  },
  "co_activation": { ... },
  "pinned": ["systems/architecture.md"]
}
```

Matching is `any(kw in prompt.lower() for kw in keywords)` -- simple substring containment. No word boundaries, no stemming. "auth" would match "authentication" but also "author".

Resolution order: project-local `.claude/keywords.json` -> global `~/.claude/keywords.json` -> hardcoded defaults in script.

### Content Injection Strategy

HOT files: `full_path.read_text()` -- entire file dumped into output.

WARM files: First `WARM_HEADER_LINES = 25` lines, with truncation marker:
```
... [WARM: Content truncated, mention to expand] ...
```

This relies on documentation files having structured headers (role, host, topology table, health checks) in the first 25 lines. The project provides templates enforcing this structure with a `<!-- WARM CONTEXT ENDS ABOVE THIS LINE -->` marker.

Output format includes a status header:
```
ATTENTION STATE [Turn N]
Hot: X | Warm: Y | Cold: Z
Total chars: N / 25,000
```

---

## 3. Multi-Instance Pool Coordination

### State Sharing

Pool state is an append-only JSONL file: `.claude/pool/instance_state.jsonl` (project-local) or `~/.claude/pool/instance_state.jsonl` (global fallback).

Each instance is identified by the `CLAUDE_INSTANCE` environment variable (e.g., A, B, C, D). Instances don't share attention state -- only coordination signals.

### Pool Entry Schema

```json
{
  "id": "uuid4",
  "timestamp": 1735145600,          // Unix epoch seconds
  "source_instance": "A",
  "session_id": "abc12345",         // Truncated to 8 chars
  "action": "completed|blocked|signaling|claimed|health",
  "topic": "Auth token refresh fix",
  "summary": "Fixed race condition...",  // Max 200 chars
  "relevance": {                     // Per-instance relevance scores
    "A": 0.12,
    "B": 0.0,
    "C": 0.88,
    "D": 0.0
  },
  "affects": "auth.py, session_handler.py",
  "blocks": "Session management refactor can proceed",
  "ttl": 3600,                       // 1 hour default
  "raw_hash": "sha256[:16]"         // Dedup hash
}
```

### Task/Blocker Detection

**Explicit mode:** Claude includes a fenced `pool` block in its response. The Stop hook (`pool-extractor.py`) parses it using regex: `` ```pool\n(.*?)``` ``. Fields are key: value pairs.

**Implicit mode (auto-update):** `pool-auto-update.py` runs on every `UserPromptSubmit`, reads the last assistant response from the transcript, and applies regex heuristics:

- Completion patterns: `(fixed|resolved|completed|deployed|merged|pushed)\s+(.{10,80})`
- Blocker patterns: `(blocked by|cannot|unable to|waiting for)\s+(.{10,80})`
- Signaling patterns: `(discovered|found|noticed)\s+(.{10,80})`

5-minute cooldown between implicit updates to prevent spam.

### Duplicate Prevention

Relevance filtering on load: each pool entry has per-instance relevance scores computed by keyword matching against hardcoded domain definitions:

```python
domains = {
    "A": ["pipeline", "orchestration", "routing", ...],
    "B": ["visual", "image", "clip", "llava", ...],
    "C": ["inference", "oracle", "transformer", ...],
    "D": ["edge", "hailo", "jetson", "npu", ...]
}
```

`relevance = min(matches / len(keywords) * 2, 1.0)`. Entries with relevance < 0.3 are filtered out on load (unless they have a `blocks` field). TTL is 1 hour -- entries expire on read (lazy deletion, no cleanup daemon).

For actual dedup of work, the system relies on visibility: when Instance B starts a session, it sees "[A] completed: Auth bug fix" and Claude can inform the user. There's no lock/claim enforcement -- it's advisory.

### Auto vs Manual Mode

- **Manual:** User gets Claude to include a `pool` block; Stop hook extracts it. Requires user discipline.
- **Auto:** Runs on every prompt, scans previous assistant response for completion/blocker patterns. 5-min cooldown. Lower precision (regex heuristics) but zero user effort.

Both modes write to the same JSONL file. Auto mode was added later (pool-auto-update.py) to handle long-running sessions where explicit pool blocks are impractical.

---

## 4. Hooks Integration

### UserPromptSubmit (runs on every user message)

1. **context-router-v2.py**: Reads JSON from stdin with `prompt` field. Loads attention state, applies decay/activation/co-activation, builds tiered output, prints to stdout. Output is injected into Claude's context.

2. **pool-auto-update.py**: Reads last assistant response from transcript JSONL. Checks for explicit pool blocks first, then tries implicit signal detection. Writes to pool file with cooldown.

### SessionStart (runs when session begins)

**pool-loader.py**: Loads recent pool entries (< 1 hour, relevance >= 0.3 or own entries). Formats as compact summary or full detail (controlled by `POOL_COMPACT` env var). Prints to stdout for injection.

### Stop (runs when session ends)

**pool-extractor.py**: Reads last assistant response from transcript. Extracts explicit `pool` blocks only (no implicit detection). Writes to pool file. Outputs `{"suppressOutput": true}` to avoid cluttering transcript.

### Error Handling

All hooks fail silently -- they catch exceptions, write to error log files, and exit 0. This prevents a broken hook from blocking Claude Code. Defensive but means failures are invisible unless you check logs.

---

## 5. Storage & State Schema

### attn_state.json

```json
{
  "scores": {
    "systems/orin.md": 0.85,
    "modules/pipeline.md": 0.49,
    "integrations/pipe-to-orin.md": 0.35,
    ...
  },
  "turn_count": 47,
  "last_update": "2025-12-31T18:43:21.123456"
}
```

Location: `.claude/attn_state.json` (project-local) or `~/.claude/attn_state.json` (global).

Initialized with all keywords.json file paths at score 0.0. Persists across sessions (not cleared on SessionStart). This means attention carries over between sessions -- if you were working on auth yesterday, auth.md starts with a decayed-but-nonzero score today.

### attention_history.jsonl

Append-only JSONL, one entry per turn:

```json
{
  "turn": 47,
  "timestamp": "2025-12-31T18:43:21Z",
  "instance_id": "A",
  "prompt_keywords": ["refactor", "ppe", "routing", "tier"],
  "activated": ["ppe-anticipatory-coherence.md"],
  "hot": ["ppe-anticipatory-coherence.md", "t3-telos.md"],
  "warm": ["orin.md", "pipeline.md"],
  "cold_count": 12,
  "transitions": {
    "to_hot": ["ppe-anticipatory-coherence.md"],
    "to_warm": ["orin.md"],
    "to_cold": ["img-to-asus.md"]
  },
  "total_chars": 18420
}
```

Location: `~/.claude/attention_history.jsonl`. Retention: 30 days (configurable but not auto-pruned in code -- the MAX_HISTORY_DAYS constant exists but no pruning logic is implemented).

### Pool State (instance_state.jsonl)

See schema in Section 3. Append-only JSONL. Lazy TTL expiration on read. No compaction/pruning implemented.

### Usage Tracking State (v1.2 prototype)

- `usage_stats.json`: Per-file injection/access/edit counts
- `usage_history.jsonl`: Per-turn injection and access records
- `keyword_weights.json`: Learned keyword weight adjustments
- `learning_progress.txt`: Human-readable log of weight changes

---

## 6. Token Economics

### How Reduction Is Achieved

The system claims 64-95% token reduction. The mechanism is straightforward: instead of loading all documentation files every turn, only HOT files (max 4, full content) and WARM files (max 8, 25-line headers) are injected.

**Budget math:**
- Hard ceiling: 25,000 chars (~6,250 tokens at 4 chars/token)
- Typical HOT file: 5,000-10,000 chars
- Typical WARM header: 500-1,000 chars
- Max injection: 4 HOT * 10K + 8 WARM * 1K = 48K, but capped at 25K

**Without the router:** All docs loaded = ~80-120K chars
**With the router:** 15-25K chars per turn

The 64-95% range depends on:
- Codebase size (more docs = higher savings)
- Conversation focus (narrow topic = fewer files activated)
- Turn number (early turns may load more; later turns decay unused files)

### Injection Budgets

| Budget Parameter | Value |
|-----------------|-------|
| MAX_TOTAL_CHARS | 25,000 |
| MAX_HOT_FILES | 4 |
| MAX_WARM_FILES | 8 |
| WARM_HEADER_LINES | 25 |

v2.0 (hologram) changes the budget split to 80/20: 80% for full content (Tier 1), 20% for headers (Tier 2), with a third tier (file paths only) for awareness.

---

## 7. Comparison to claude-memory

### What claude-cognitive does that we don't

| Capability | claude-cognitive | claude-memory |
|-----------|-----------------|---------------|
| Dynamic context injection per turn | Yes (hook-based, every UserPromptSubmit) | Partial (startup_load, then on-demand recall) |
| Attention decay | Yes (multiplicative per-turn decay) | No (memories persist until forgotten) |
| Co-activation / spreading activation | Yes (related files boost together) | No (edges exist but not used for auto-boost) |
| Multi-instance coordination | Yes (pool system) | No |
| Token budget enforcement | Yes (hard ceiling with tier demotion) | No explicit budget |
| Usage-based learning | Prototype (v1.2) | No |

### What we do that they don't

| Capability | claude-memory | claude-cognitive |
|-----------|---------------|-----------------|
| Semantic memory storage | Yes (embeddings + recall) | No (only routes pre-written docs) |
| Memory creation from conversation | Yes (remember tool) | No (docs must be pre-written) |
| Consolidation / reflection | Yes (consolidate, reflect) | No |
| Sleep cycle processing | Yes (sleep_gather/write) | No |
| Forgetting / pruning | Yes (forget, TTL) | No (docs are permanent) |
| Cross-session knowledge transfer | Yes (memories persist) | Partial (attention state persists but no new knowledge) |

### Fundamental Difference

claude-cognitive is a **context router** -- it selects which pre-existing documentation files to show Claude. It doesn't create or modify those files. The "memory" is the documentation itself, maintained by humans.

claude-memory is a **knowledge store** -- it creates, retrieves, and manages memories from conversation. The memory content is generated and maintained by the system.

These are complementary, not competing. claude-cognitive solves "which of my existing docs should Claude see right now?" while claude-memory solves "what has Claude learned that should persist?"

### Analogies to our system

- **Attention scoring** is analogous to our recall relevance ranking. Both answer "what's most relevant right now?" but our system uses embedding similarity while theirs uses keyword matching + decay.
- **startup_load** is roughly analogous to their SessionStart hook, but their system loads pool coordination state while ours loads recent memories.
- **HOT/WARM/COLD tiers** map loosely to our shadow-load pattern: we inject key memories at session start (like HOT) and make others available on demand (like WARM recall). We don't have COLD -- our equivalent is "not recalled."
- **Co-activation** is conceptually similar to our memory edges, but theirs triggers automatically while ours are passive relationships.

---

## 8. Worth Stealing

### Ranked by value and transferability

1. **Co-activation for recall (HIGH value, MEDIUM effort)**
   When a memory is recalled, automatically boost related memories' relevance in the current session. Our edge graph already exists -- we just don't use it for automatic boosting. Implementation: when `recall` returns results, check edges and apply a score boost to connected memories. This would make multi-memory retrieval more contextually coherent.

2. **Attention decay for session context (HIGH value, LOW effort)**
   Track which memories were recalled this session and apply decay to their relevance scores turn-over-turn. Memories recalled early but not referenced again should fade from "top of mind." This doesn't change stored memories -- it's a session-local relevance modifier. Would improve recall ranking within long sessions.

3. **Tiered injection at startup (MEDIUM value, LOW effort)**
   Our startup_load currently has a binary choice: inject or don't. We could tier it: inject full content for most-relevant memories, summaries for next tier, nothing for the rest. Would reduce startup token cost while maintaining awareness.

4. **Hard token budget for injected context (MEDIUM value, LOW effort)**
   We don't enforce a ceiling on how much memory content gets injected. Adding a `MAX_INJECTION_CHARS` with graceful demotion (full -> summary -> skip) would prevent context bloat in memory-heavy sessions.

5. **Usage tracking for recall quality (MEDIUM value, HIGH effort)**
   Their v1.2 prototype tracks which injected files Claude actually uses (via tool call analysis). Applying this to memories -- tracking which recalled memories Claude references in its response -- could inform future recall ranking. The signal is noisy but valuable over time.

6. **Pool coordination pattern (LOW value for us, HIGH effort)**
   Multi-instance coordination isn't relevant to our current architecture (single-user, single-instance). But the pattern of JSONL append-only state sharing with TTL expiration and relevance filtering is clean and could be useful if we ever support team memory or multi-agent scenarios.

### Ideas specifically for recall ranking

- **Keyword boost as a first-pass filter:** Before embedding similarity, check if the query contains exact keywords from memory metadata. This is computationally cheap and could pre-filter candidates.
- **Decay-weighted recall:** Apply a recency decay to embedding similarity scores. A memory that was relevant 50 turns ago should rank lower than an equally-similar memory from 5 turns ago, even if embedding distance is the same.

---

## 9. Not Worth It

1. **Replacing embeddings with keyword matching.** Their v1.x system is entirely keyword-based -- no semantic understanding. This works for routing a small set of known documentation files but would be terrible for a memory system where content is dynamic and unpredictable. Their own v2.0 roadmap acknowledges this and plans semantic relevance.

2. **Pre-written documentation as "memory."** Their system requires humans to write and maintain `.claude/` documentation files with specific header structures. This is a documentation management system, not a memory system. We create memories from conversation automatically.

3. **The v2.0 hologram DAG discovery system.** Interesting technically (content-addressed coordinates, 6 discovery strategies, toroidal topology) but it solves a problem we don't have. We don't need to discover relationships between files -- our memories are atomic with explicit edges.

4. **The hardcoded domain/instance model for relevance.** Their pool relevance scoring uses hardcoded keyword lists per instance type (A = pipeline, B = visual, etc.). This is brittle and project-specific. Not transferable.

5. **The garret_sutherland.py experimental file.** An amusing personality/lore module with sleep debt tracking and "watcher" mechanics. Not relevant to context management.

6. **Structured header format requirement.** Their WARM tier depends on documentation files having standardized headers in the first 25 lines. This is a reasonable constraint for documentation routing but irrelevant for a memory system where content structure varies.
