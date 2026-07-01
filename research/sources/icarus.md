# icarus - Git-native, provenance-first memory for coding agents (versioning/supersession, not retrieval)

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Icarus (`esaradev/icarus-memory-infra`, v0.3, MIT, Python 3.10+) is "agent coherence infrastructure": local-first, markdown-native memory whose purpose is to stop coding agents from using stale facts, repeating failed attempts, and contradicting prior decisions. It is a *versioning + provenance* layer, not a retrieval-quality system. No decay, no learned ranker, no graph, no consolidation.

### Storage & Schema
- **Plain markdown files on disk.** Entries at `<root>/YYYY/MM/icarus-<12hex>.md`, YAML frontmatter + markdown body. No SQLite, no binary index — lookups `rglob` the disk (`store.py:_find_path`). Explicitly designed to `git add` the memory alongside code.
- **Atomic writes** via tmp-file + `os.replace` (`_layers.atomic_write_text`), `os.fsync` before replace. Crash-safe.
- **Rich Pydantic `Entry` schema (~23 fields)** (`schema.py`): `id`, `agent`, `platform`, `timestamp`, `type`, `summary` (≤200 char), `body`, `project_id`, `session_id`, `status` (open/in-progress/closed), `assigned_to`, `review_of`, `revises`, `training_value`, `evidence[]`, `source_tool`, `verified`, `contradicted_by`, `artifact_paths[]`, `verification_log[]`, `lifecycle`, `superseded_by`, `supersedes[]`.
- **Two orthogonal state axes** (a genuinely clean design point): `verified` (unverified→verified→contradicted→rolled_back) is *trust/provenance*; `lifecycle` (active/superseded) is *freshness*. They live in separate fields so a fact can be unverified-and-active or verified-and-superseded (`schema.py:13-17`).
- **`EvidencePointer`**: kind (file/url/fabric_ref/tool_output/message), `ref`, `excerpt` (≤500 char), and an optional **SHA256 `hash`** of the source excerpt (validated to 64 hex chars).

### Memory Types
Three layers (README, evidence file):
1. **Working memory** — per-task scratch (observations, attempts, hypotheses), cleared at task end.
2. **Session archive** — per-agent history; failed attempts stay *private to the agent* that tried them.
3. **Wiki** — shared markdown source of truth, organized by topic; `end_session(..., promote_to_wiki=[...])` promotes a finding to the shared layer.

### Write Path
Explicit, human/agent-driven. `memory_write` / `IcarusMemory.write()` create entries; `write_with_supersession()` atomically marks the old entry `lifecycle=superseded` and sets `superseded_by`/`supersedes`. **No auto-extraction, no dedup, no salience/quality gating, no content preprocessing.** `generate_id()` is `secrets.token_hex(6)` — purely random, so there is *no content-hash dedup* (evidence file confirms). Write-time validation (`validation.py`, per DESIGN.md): referential integrity on `revises`/`review_of`/`fabric_ref` evidence; `verified='verified'` cannot be set on initial write, only via `verify()`; `type='review'` requires `review_of`; etc. Verification is manual (`memory_verify`, `memory_contradict`).

### Retrieval
`retrieval.py`. Filtering happens **before** scoring (status_filter, min_verified, lifecycle, agent, project_id, type).
- **Default `keyword` mode**: token-overlap scoring — `sum(term_counts)/len(haystack) + 0.1*len(matched_terms)` over `summary + body` (`_keyword_score`). Cheap, no model deps.
- **`hybrid`/`auto` mode** (opt-in `[embeddings]` extra): BM25 (`rank_bm25.BM25Okapi`) + `BAAI/bge-small-en-v1.5` cosine, fused via **RRF with k=60** (`_hybrid_rank`, lines 261-266). Embeddings cached per entry keyed on `(model, mtime, size)`.
- **Distinctive final sort**: `key = (_VERIFIED_ORDER[verified], -score)` — a **hard trust-tier bucket** where verified entries always rank above unverified regardless of relevance, and contradicted/rolled_back sink to the bottom (lines 174-177, and DESIGN.md "Recall"). `audit_search()` is a separate path that *includes* contradicted/rolled-back/superseded for forensics.

### Consolidation / Processing
**None** in the sleep/offline sense. The only LLM use is on-demand: `BriefingGenerator` (`briefing.py`) compiles a pre-task briefing from wiki pages + same-agent sessions + failed attempts + recent supersessions, calling gpt-4o-mini (`call_openai_json`, capped at $0.05/briefing) with a deterministic template fallback. Briefings are cached 1h, keyed on wiki+archive version. This is *compilation of existing facts*, not narrative generation or extraction.

### Lifecycle Management
This is Icarus's real substance:
- **Supersession** — non-destructive; old facts marked `superseded`, never overwritten. `_recent_superseded` (30-day window) surfaces them in briefings so agents don't reuse stale facts.
- **Non-destructive rollback** (`rollback.py`) — `plan_rollback` walks the `revises` chain backward to the first `verified` ancestor, marks intermediates `rolled_back` (appending `VerificationRecord`s), and writes a new `type=rollback` entry pointing at the verified ancestor. Optional `cascade=True` also taints transitive descendants found via a reverse-`revises` index (`lineage._find_descendants`, with cycle detection). No file ever deleted.
- **Lineage** — `lineage()` BFS-walks `revises` + `review_of` chains for full ancestry.
- **No time-based decay / forgetting.** All lifecycle transitions are explicit.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Provenance is first-class | `EvidencePointer` with kind/ref/excerpt/SHA256 hash; write-time referential validation | Validated (real code, well-tested) |
| Stale facts marked superseded, never overwritten | `write_with_supersession`, `lifecycle` field, `_recent_superseded` in briefing | Validated |
| Non-destructive rollback to last verified ancestor | `rollback.py` walk + taint + cascade, dedicated tests (`test_rollback_cascade.py`) | Validated |
| Verified entries outrank unverified in recall | `retrieval.py:174-177` sort key `(verified_bucket, -score)` | Validated |
| Hybrid retrieval | BM25 + bge-small cosine, RRF k=60 — but opt-in extra, keyword is default | Validated; retrieval is basic/untuned |
| "Persistent, sourced memory" prevents repeated dead ends | Failed attempts archived per-agent, surfaced in briefing | Plausible; no benchmark, no eval |
| Original comparison-table claims (webUi, entities, decay, dedup, autoExtract, quality refine, 5 platform supports) | Absent in code | **Debunked by evidence file** |

**No end-to-end QA benchmark exists.** No LoCoMo, no retrieval metrics, no ablations. Comparison to Somnigraph's 85.1% LoCoMo is not possible — different problem class entirely.

---

## Relevance to Somnigraph

### What icarus does that Somnigraph doesn't
- **Provenance grounding with source hashing.** Every fact can carry an `EvidencePointer` (source file/url + excerpt + SHA256). Somnigraph memories are ungrounded free text — no `db.py`/`tools.py` field ties a memory to the artifact it came from, and nothing detects when that source changed. This is the write-path-grounding gap corroborated across ByteRover/agentmemory (see `ai-memory-comparison.md`, `agentmemory.md`).
- **A trust/verification tier applied at ranking time.** `retrieval.py` hard-sorts verified > unverified > contradicted. Somnigraph's `reranker.py`/`scoring.py` have no notion of a memory being *disputed* or *verified*; a contradicted fact ranks purely on relevance. Somnigraph detects contradiction edges during sleep but does not demote a contradicted memory in live recall.
- **Explicit non-destructive revision lineage.** `revises`/`supersedes`/`review_of` chains + rollback-with-cascade. Somnigraph has `valid_from`/`valid_until` and sleep-detected `revision`/`evolves` edges, but no operator-facing rollback and no cascade-taint of derived facts.
- **Per-agent isolation of failed attempts.** Multi-agent data isolation (private archives, shared wiki). Somnigraph is single-user.

### What Somnigraph does better
- **Retrieval quality**, decisively. Somnigraph: RRF k=14 Bayesian-tuned + 26-feature LightGBM reranker (NDCG 0.7958) + PPR graph expansion + explicit feedback loop (Spearman 0.70). Icarus: token-overlap by default, untuned RRF k=60, no reranker, no feedback, no graph.
- **Offline consolidation** (`sleep_nrem.py`/`sleep_rem.py`) — Icarus has nothing analogous; its only "processing" is on-demand briefing compilation.
- **Decay / lifecycle automation** — Somnigraph's per-category exponential decay vs Icarus's fully-manual transitions.
- **Measured benchmarks** — Somnigraph has LoCoMo QA + multi-hop failure analysis; Icarus has none.

---

## Worth Stealing (ranked)

### 1. Verified/contradicted trust tier as a hard ranking bucket (Low)
**What**: A per-memory trust axis (`verified`/`contradicted`) that, at recall time, buckets results *above* relevance score so a disputed memory can never outrank an undisputed one, and a contradicted memory sinks below everything (`retrieval.py:174-177`).
**Why**: Somnigraph already *detects* `contradicts` edges during NREM sleep but does nothing with that signal at live-recall time — a contradicted/superseded fact still surfaces on pure relevance. A trust bucket turns an already-computed sleep signal into a live safety rail.
**How**: In `scoring.py`/`reranker.py`, add a coarse pre-sort bucket derived from sleep-detected contradiction/supersession state (or a boolean reranker feature at high weight). Cheap; reuses existing edge data.

### 2. Source-artifact hashing for staleness detection (Medium)
**What**: Attach to a memory a pointer to its source (file path/url) plus a SHA256 of the grounding excerpt (`EvidencePointer.hash`). When the source's current hash diverges, the memory is a staleness candidate.
**Why**: Directly addresses the write-path-grounding gap that ByteRover/agentmemory independently flag as the actual LoCoMo-leader differentiator. Somnigraph memories are ungrounded, so it cannot tell when a remembered fact's basis has changed.
**How**: Optional `evidence`/`source_hash` columns in `db.py`; a sleep-phase check that re-hashes referenced artifacts and flags divergence for REM gap analysis. Additive, opt-in.

### 3. Recent-supersession surfacing in a pre-task hint (Low-Medium)
**What**: `_recent_superseded` (30-day window) proactively injects "these facts were recently superseded — don't reuse them" into the pre-task briefing (`briefing.py:155-164`).
**Why**: Same shape as the `docs/proposals/proactive-injection.md` design — surface a signal before the agent asks. Recently-invalidated facts are exactly the high-value negative hint the proactive floor could carry.
**How**: When proactive injection ships, add a "recently invalidated" channel sourced from sleep-detected supersession/`valid_until` transitions, distinct from the positive recall hint.

---

## Not Useful For Us

### Git-native markdown-on-disk storage (no index)
Deliberate for a git-versioned, human-auditable coding-agent fabric, but a strict downgrade from Somnigraph's SQLite + sqlite-vec + FTS5 for retrieval at scale. Not adoptable.

### Per-agent multi-agent isolation, working-memory scratch, task-status tracking
Icarus is a team-of-coding-agents coherence tool; these solve problems single-user Somnigraph does not have.

### RRF k=60 default
Somnigraph's k=14 is Bayesian-optimized on real data; Icarus's k=60 is the unexamined textbook default. Nothing to take.

---

## Connections
- **Write-path over retrieval** thesis: strongly convergent with `byterover.md`, `agentmemory.md`, and the `ai-memory-comparison.md` sweep — the systems that win LoCoMo do so on write-time grounding/provenance, not retrieval cleverness. Icarus is a pure expression of that stance (rich provenance schema, deliberately dumb retrieval), though it publishes no benchmark to prove it.
- **Supersession / non-destructive versioning**: convergent with memv's supersession pattern and MIRIX-style provenance chains — multiple systems independently land on "mark stale, never overwrite" + lineage walk.
- **Verified/contradicted trust axis**: a live-retrieval consumer of exactly the contradiction signal Somnigraph computes offline in `sleep_nrem.py` but currently discards at recall time.

---

## Summary Assessment

Icarus's core contribution is a *clean data model for provenance and non-destructive revision* aimed at coding agents on long-lived codebases: the orthogonal `verified`×`lifecycle` axes, SHA256-hashed evidence pointers, non-destructive rollback-to-last-verified-ancestor with cascade taint, and a pre-task briefing that compiles current/superseded/failed context. The engineering is careful (atomic writes, referential validation, cycle detection, good test coverage) and the design docs are honest about scope. It is a versioning/coherence tool, not a retrieval-quality system — retrieval is token-overlap by default with an opt-in untuned BM25+embedding RRF.

The single most useful takeaway for Somnigraph is not a mechanism to clone wholesale but a *stance made concrete*: put provenance and trust in the schema and let them influence live recall. Somnigraph already computes contradiction/supersession signals during sleep and then ignores them at recall time; Icarus shows the payoff of a trust-tier sort bucket (idea #1) and source-hash grounding (idea #2). Both are low-to-medium effort and additive.

What's overhyped: the original comparison-table entry claimed ~30 features; the carsteneu audit verified only 12 and debunked 17 (webUi, entities, decay, dedup, autoExtract, quality-refine, most platform supports). The carsteneu evidence file is unusually rigorous here — it *corrects the project's own inflated claims* and fixes a wrong URL. My cross-check agrees with it on every point. No benchmark exists, so none of Icarus's coherence claims are quantitatively validated, and it is not comparable to Somnigraph's 85.1% LoCoMo QA. Verdict: MAYBE — one genuinely worth-revisiting idea (trust-tier ranking of already-computed contradiction signal), plus a corroborating data point for the write-path-grounding thesis.
