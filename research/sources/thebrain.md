# TheBrain (Advenire Consulting) Analysis

*Generated 2026-03-28 by Opus agent reading local clone*

---

## 1. Architecture Overview

**Repo**: https://github.com/Advenire-Consulting/thebrain
**Stars**: ~22 (as of 2026-03-28)
**Forks**: 3
**License**: MIT
**Language**: JavaScript (Node.js 18+), ~6,700 lines across ~90 JS files (includes tests)
**Created**: 2026-03-15, 11 commits to date
**Dependency**: better-sqlite3 (sole runtime dependency)
**Description**: "A Claude Code plugin that makes your AI remember, navigate, and learn." Cognitive layer organized around brain-region metaphors.

**Delivery mechanism**: Claude Code plugin (not an MCP server). Uses hooks (`SessionStart`, `PreToolUse`, `PostToolUse`) and slash commands (`/hello`, `/wrapup`, `/continue`, `/dopamine`, `/oxytocin`). Context is injected via `additional_context` in the SessionStart hook output, not via tool definitions. The brain tools (hippocampus queries, CC2 search) are invoked as Bash commands, not as registered MCP tools.

**Module organization** (brain-region metaphor):
- `hippocampus/` — Codebase spatial index: file imports/exports, identifier search, blast radius, DIR files, term DB
- `cerebral-cortex-v2/` — Conversation recall: JSONL transcript indexing, FTS5 search, decision detection, window-based reading
- `hypothalamus/` — Safety hooks: pre-edit/pre-bash path classification, blast radius warnings, sensitivity blocking
- `dlpfc/` — Working memory: file-level heat tracking, exponential decay, context notes, cluster detection
- `scripts/` — Behavioral system: lessons (dopamine), forces (oxytocin), prefrontal-live.md generation
- `hooks/` — Runtime wiring: session-start loader, hypothalamus hook, post-edit hook (term index + dlPFC bump)
- `commands/` — Slash command markdown definitions

**Data**: All stored under `~/.claude/brain/`. Multiple SQLite databases: `signals.db` (lessons + forces), `working-memory.db` (file heat), `recall.db` (conversation index), `terms.db` (identifier index). Generated markdown files (`prefrontal-live.md`, `dlpfc-live.md`) are loaded into context at session start.

---

## 2. Memory Model

TheBrain has no unified memory store. Instead, it has four distinct memory subsystems, each with its own storage and lifecycle:

### Behavioral Memory (signals.db)

Two types of behavioral data:

**Lessons** (via `/dopamine`): Stored with brain_file (amygdala, nucleus-accumbens, prefrontal, hippocampus), domain, title, entry text, polarity (positive/negative), and a weight counter (confirmation_count). New lessons start at weight 50. Each `/dopamine` reinforcement adds 50, capping at 100. Weight determines enforcement tier:
- 75-100 = Rule (always loaded, non-negotiable)
- 50-74 = Inclination (always loaded, overridable with reasoning)
- 25-49 = Awareness (on-demand only)
- <25 = Data (background, not surfaced)

**Forces** (via `/oxytocin`): Relational dynamics with a score (0-100). Always-on forces (75+) shape every interaction; planning-mode forces (50-74) activate during design discussions.

Both compile into `prefrontal-live.md`, loaded at session start. This is the "learning centers for positive and negative lessons" mentioned in the Reddit discussion — amygdala stores pain points, nucleus accumbens stores good patterns. The mechanism is simple (weight counter + tier thresholds) but the UX is thoughtful: the structured `/dopamine` command walks through moment reconstruction, lesson discussion, categorization, and collaborative entry crafting before storing.

No decay on behavioral data. Lessons persist indefinitely unless manually deactivated.

### Conversation Recall (CC2 recall.db)

JSONL session transcripts are indexed into "windows" (segments bounded by session start/end or context compaction). Each window stores:
- Term heatmaps (user vs assistant, with line numbers and counts)
- File references (from tool calls)
- Project frequencies
- Decision markers (detected via Read-discussion-Write/Edit heuristic)
- Summaries (mechanical from terms, optionally overwritten by Claude-authored PFC entries)

FTS5 search with cluster-scored queries. Trust decay: `1 / (1 + daysSince * 0.1)`. No vector search — purely term-based.

Dynamic stopword filter: noise terms flagged during search usage are auto-promoted after 5 flags without a relevant hit. A relevant hit resets the noise count. This is a simple but effective feedback mechanism for search quality.

### Working Memory (dlpfc, working-memory.db)

File-level heat tracking with exponential decay:
- Edit: +1.0, Read: +0.3, Referenced in conversation: +0.5
- All scores multiplied by 0.8 at each wrapup (opt-in)
- Files above 2.0 = "hot" (summary + context_note loaded), 1.0-2.0 = "warm" (summary only), <1.0 = cold

Co-occurrence clusters track files that appear together across sessions (all pairwise combinations per session, sorted alphabetically as JSON key). Generated output caps at 15 files and 3 clusters per project, targeting 200-400 tokens.

Context notes are Claude-authored during `/wrapup` — volatile, describing what was being done with the file. Summaries are stable, seeded from DIR file purpose lines.

### Codebase Index (hippocampus)

Not a memory system per se, but a persistent structural index. DIR files map imports/exports/routes/schemas per project. Term database indexes every identifier across all projects for cross-project `--find` queries. Incrementally updated via PostToolUse hook on every edit.

---

## 3. Retrieval Pipeline

There is no unified retrieval pipeline. Each subsystem has its own:

**CC2 search**: Cluster-scored term search. Each argument is a cluster of OR-terms. Scoring: `(clusterScoreSum + projectBoost + fileBoost) * trustDecay`. User terms weighted 2x assistant terms. Project boost via hippocampus alias resolution. File boost for direct path matches. Trust decay is `1/(1 + days * 0.1)`. FTS5 candidate retrieval, then in-memory scoring. No vector search, no embedding, no reranking.

**Hippocampus queries**: Direct SQLite lookups. `--find` queries the FTS5 term index. `--blast-radius` traverses import graphs from DIR files. `--map` reads DIR files and returns purpose + description for each file.

**Working memory**: Simple score-sorted query of file_heat table, filtered by threshold.

**Behavioral**: All rules/inclinations above threshold are loaded into context at session start. No search needed — the full behavioral ruleset is always present.

---

## 4. Write Path

**Conversation indexing**: Triggered by `/wrapup` (mechanical step). Scanner reads JSONL files, detects window boundaries (session start, compaction markers), writes to windows.json. Extractor reads JSONL within each window, produces term heatmaps, file references, project frequencies. Decision detector walks JSONL looking for Read-then-Write patterns with overlapping file paths.

**Behavioral learning**: `/dopamine` command walks through a structured 6-step process: reconstruct the moment (review last 10-15 messages), discuss the lesson, categorize against existing domains, craft the entry collaboratively, store with weight increment, confirm. `/oxytocin` follows a similar pattern for relational forces. Both are LLM-mediated (Claude interprets and structures) but store mechanically (SQLite insert via helper script).

**Working memory**: Automatic via hooks. PreToolUse Read hook bumps file heat +0.3. PostToolUse Edit/Write hook bumps +1.0, also incrementally updates term index and DIR file entry. Context notes written by Claude during `/wrapup`.

**Codebase index**: Full rebuild via `scan.js` (at wrapup), incremental updates via post-edit hook (during session). Descriptions are preserved across re-scans; purpose lines are regenerated from metadata.

---

## 5. Comparison to Somnigraph

### What they have that we don't

| Feature | TheBrain | Somnigraph equivalent |
|---------|----------|----------------------|
| **Blast radius / dependency graph** | Import/export graph per project, warns before editing high-dependency files | No codebase awareness — Somnigraph is content-agnostic |
| **Pre-edit safety hooks** | Hypothalamus blocks edits to databases, .env files; warns on high-blast-radius files | No hook system |
| **File heat tracking** | Per-file scores with exponential decay, context notes, cluster detection | No file-level working memory |
| **Behavioral weight tiers** | Lessons graduate from data → awareness → inclination → rule via reinforcement | Priorities exist but are flat (1-10 scale, no promotion mechanism) |
| **Decision detection** | Read→discuss→Write heuristic identifies decision points in transcripts | No transcript analysis |
| **Conversation window indexing** | Indexes JSONL transcripts with term heatmaps, compaction boundary detection | Session logs exist but are not indexed |
| **Onboarding questionnaire** | 9-question setup personalizes behavioral defaults | No onboarding — CLAUDE.md snippet is the entry point |
| **Cross-project navigation** | `--find` searches identifiers across all registered workspaces | Single-DB, no cross-project awareness |
| **Incremental index updates** | PostToolUse hook updates term index and DIR entry after every edit | No codebase indexing |

### What we have that they don't

| Feature | Somnigraph | TheBrain equivalent |
|---------|-----------|---------------------|
| **Vector search** | sqlite-vec embeddings, hybrid BM25+vector retrieval | No embeddings, no vector search |
| **Learned reranker** | LightGBM 26-feature model, trained on 1032 human-judged queries | No ML, no reranking |
| **Graph-based retrieval** | Typed edges (support, contradict, evolve, derive), PPR traversal, Hebbian co-retrieval | No graph structure |
| **Biological decay** | Per-memory configurable decay rates, three tiers (episodic/semantic/procedural) | Fixed 0.8 decay only on file heat; behavioral data has no decay |
| **Sleep consolidation** | LLM-driven NREM/REM pipeline (duplicate detection, contradiction resolution, entity extraction) | No consolidation — data accumulates |
| **Retrieval feedback loop** | EWMA + UCB exploration, per-memory utility tracking | Dynamic stopwords (noise/relevant counts) on conversation search only |
| **RRF fusion** | Multi-signal fusion with configurable weights | Single-signal scoring per subsystem |
| **Content-level memory** | Stores semantic content with categories, themes, priorities | Stores behavioral rules and file references, not semantic content |
| **Embedding-based similarity** | Deduplication at 0.9 similarity, similarity search | No similarity computation |

### Architectural trade-offs

| Dimension | TheBrain | Somnigraph |
|-----------|---------|------------|
| **Delivery** | Claude Code plugin (hooks + slash commands) | MCP server (tool definitions) |
| **Scope** | Codebase-aware + behavioral + conversation recall | Content-agnostic semantic memory |
| **ML dependency** | None — all heuristic | LightGBM reranker, OpenAI embeddings |
| **Storage** | 4 separate SQLite DBs + generated markdown | Single SQLite DB + sqlite-vec |
| **Retrieval model** | Term-based FTS5, no vectors | Hybrid BM25 + vector + graph + RRF |
| **Learning mechanism** | Explicit (`/dopamine` reinforcement) | Implicit (retrieval feedback, Hebbian co-retrieval) |
| **External APIs** | None | OpenAI embeddings API |
| **Token awareness** | Explicit focus on token economy (README quantifies savings) | No token-level awareness |

---

## 6. Worth Adopting?

### File heat tracking (dlPFC concept)

**Idea**: Track which files Claude reads and edits, build heat scores with exponential decay, load hot files with context notes at session start.

**Assessment**: Interesting but not directly applicable. Somnigraph is content-agnostic — it doesn't know about files, projects, or codebases. The dlPFC concept addresses a real problem (cold-start re-orientation) but solves it at the wrong layer for Somnigraph. The analogous mechanism in Somnigraph is retrieval frequency and Hebbian co-retrieval, which operate on memory content rather than file paths.

The cluster detection (file co-occurrence tracking) is a simplified version of what Somnigraph's Hebbian edges already capture, but applied to files rather than memories.

**Verdict**: Not adoptable as-is. The heat decay math is trivial (score * 0.8 per session) and less sophisticated than Somnigraph's per-memory configurable decay rates.

### Blast radius pre-write hook warnings

**Idea**: Before Claude edits a file, check how many other files import it and warn (or block) accordingly. Classify files by sensitivity (databases, .env files block; high-dependency files warn).

**Assessment**: This is genuinely useful for a code-editing tool but entirely orthogonal to Somnigraph's purpose. Somnigraph is a memory system, not a code navigation or safety tool. The hypothalamus pattern (classify-before-act with severity levels) could theoretically apply to memory operations (e.g., warn before forgetting a high-connectivity memory node), but the complexity isn't justified — Somnigraph's forget operation already requires explicit intent.

**Verdict**: Not relevant. Different problem domain.

### Behavioral weight tiers with reinforcement

**Idea**: Lessons start at weight 50 and graduate through tiers (data → awareness → inclination → rule) via explicit reinforcement. Two `/dopamine` reinforcements promote a pattern to a "rule" that's always enforced.

**Assessment**: Somnigraph's priority system (1-10) with decay serves a similar purpose but with continuous rather than discrete tiers. The TheBrain approach is more legible to users (clear tier labels, explicit reinforcement ceremony) but less nuanced (only 4 tiers, no decay, no automatic demotion). The structured `/dopamine` UX (5-step collaborative lesson crafting) is well-designed but requires LLM mediation for every storage event — Somnigraph's `remember()` is a direct tool call.

The "two reinforcements to become a rule" mechanic is simple but effective for a plugin that relies on user-visible behavioral rules. It doesn't translate well to Somnigraph where the equivalent would be retrieval feedback already handling this implicitly.

**Verdict**: The UX design is thoughtful but the mechanism is simpler than what Somnigraph already has. The tier labels (Rule/Inclination/Awareness/Data) are a nice presentation pattern but not an architectural improvement.

### Dynamic stopword filter

**Idea**: During CC2 search, flag noise terms. After 5 noise flags without a relevant hit, auto-promote to stopword list. A single relevant hit resets the noise count and removes promotion.

**Assessment**: This is a miniature version of Somnigraph's retrieval feedback loop, applied to a single dimension (term relevance in conversation search). The asymmetric reset (one relevant hit cancels five noise flags) is a sensible design choice. Somnigraph's EWMA + UCB system operates at the memory level rather than the term level, but the principle of using retrieval feedback to improve future retrieval quality is shared.

**Verdict**: Already covered by Somnigraph's feedback loop at a more general level.

### Decision detection heuristic

**Idea**: Walk JSONL transcripts looking for Read→discussion→Write/Edit patterns to identify decision points. Store decision markers with terms, file anchors, and continuation links across compaction boundaries.

**Assessment**: Novel approach to extracting structure from conversation transcripts. Somnigraph doesn't index transcripts at all — its memories are explicitly stored via `remember()`. The decision detection pattern could be useful for a hypothetical "session analysis" feature, but Somnigraph's design philosophy is explicit memory creation, not implicit extraction from transcripts.

**Verdict**: Interesting pattern, not applicable to Somnigraph's architecture.

---

## 7. Worth Watching

- **Conversation indexing maturity**: CC2's window-based JSONL indexing with decision detection is a different approach to session recall than Somnigraph's explicit memory creation. If TheBrain demonstrates that implicit extraction from transcripts produces useful recall, it could validate (or invalidate) Somnigraph's design choice of requiring explicit `remember()` calls.

- **Plugin ecosystem adoption**: TheBrain is designed as a Claude Code plugin, not an MCP server. If Claude Code's plugin system gains traction, it may validate the hook-based approach (pre/post tool use interception) over MCP's tool-definition approach for memory systems.

- **Behavioral learning convergence**: The `/dopamine` + `/oxytocin` system is currently simple (weight counter + tier thresholds). If it evolves toward more sophisticated behavioral modeling (temporal weighting, domain-specific decay, contradiction detection), it would be worth comparing against Somnigraph's category/priority/decay system.

---

## 8. Key Claims

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Saves tokens" — detailed per-feature token savings tables in README | Token estimates are plausible for the scenarios described but are theoretical (no measured benchmarks, "~" estimates throughout). The cumulative cost model (tokens persist in context across all turns) is correct and well-explained. | **Plausible** — the math is directionally right but unvalidated |
| "Learns how you work" via dopamine/oxytocin | Behavioral lessons persist in signals.db and are loaded at session start. Reinforcement mechanism exists. But "learning" is entirely user-initiated — no automatic pattern detection, no feedback loop. | **Plausible** — stores and surfaces explicit user feedback; "learns" is generous for what is essentially a weighted rule store |
| "File heat map based granularity" | dlPFC implementation exists and works (exponential decay, edit/read/reference weights, context notes, cluster detection). Heat scores determine what loads at session start. | **Verified** — code implements the described mechanism |
| "Blast radius pre-write hook warnings" | Hypothalamus hook fires on every Edit/Write/Bash call, classifies paths against hippocampus DIR data, warns or blocks based on sensitivity and dependency count. | **Verified** — code implements the described mechanism |
| "Learning centers for positive and negative lessons" | Amygdala (pain points) and nucleus accumbens (good patterns) are brain_file categories in the lessons table. Weight-based tier promotion exists. | **Verified** — but the brain-region naming is metaphorical; the mechanism is a weighted tag store |
| Cross-project identifier search | Term DB indexes identifiers across all registered workspaces. `--find` queries work cross-project. | **Verified** — FTS5 index with per-project scoping |
| "Single dependency — just Node.js 18+" | package.json shows only better-sqlite3 as a runtime dependency. | **Verified** |
| Token savings of 100,000-300,000 cumulative tokens per session | Theoretical estimate based on assumed usage patterns and turn counts. No measurement methodology. | **Aspirational** — plausible ceiling but no empirical validation |

---

## 9. Relevance to Somnigraph

**Rating: Low**

TheBrain and Somnigraph solve different problems with different architectures. TheBrain is a codebase navigation and behavioral rule system delivered as a Claude Code plugin. Somnigraph is a semantic memory system with learned retrieval, graph-based knowledge representation, and biological decay.

The overlap is minimal:
- Both use SQLite (but for entirely different schemas and purposes)
- Both have a concept of decay (but TheBrain's is trivial — a fixed 0.8 multiplier on file heat)
- Both store behavioral patterns (but via fundamentally different mechanisms — explicit `/dopamine` ceremonies vs. implicit retrieval feedback)

TheBrain's strengths (codebase awareness, blast radius analysis, pre-edit safety hooks, token economy awareness) are orthogonal to Somnigraph's domain. Its weaknesses relative to Somnigraph (no vector search, no ML, no graph, no consolidation, no embedding) reflect a deliberate design choice: zero external API dependencies, pure heuristic approach.

The most interesting aspect for Somnigraph is the *negative* comparison: TheBrain demonstrates what a memory-adjacent system looks like without embeddings, without a learned reranker, and without graph structure. The conversation recall (CC2) relies entirely on FTS5 term matching with cluster scoring — no semantic similarity at all. This would make an interesting control condition if anyone were comparing retrieval approaches, but it doesn't offer patterns that would improve Somnigraph's already more sophisticated pipeline.

The behavioral system (`/dopamine` + `/oxytocin`) is well-designed UX but architecturally simpler than Somnigraph's category/priority/decay/feedback system. The onboarding questionnaire is a nice touch that Somnigraph doesn't have, but Somnigraph's CLAUDE.md snippet serves a different purpose (guiding LLM usage of tools vs. personalizing behavioral rules).
