# Declined After Examination

A consolidated record of systems, papers, benchmarks, and techniques Somnigraph has **consciously examined and chosen not to pursue** — or tried and removed. This is honest-accounting infrastructure: the negative space around the research corpus. A thing listed here is *not* a backlog item (see `roadmap.md` for deferred work); it is something we looked at and decided against, with the reason recorded so a future session doesn't redo the examination.

Three parts:

1. **Comparison-table rows** — every system from the [carsteneu/ai-memory-comparison](https://github.com/carsteneu/ai-memory-comparison) directory ([live table](https://carsteneu.github.io/ai-memory-comparison/)) not already in `research/sources/`, now **surveyed at the code level** (full analysis per system). Most yielded ideas (consolidated in [`ideas-considered.md`](ideas-considered.md)); the genuinely-declined subset is the "nothing to adopt" set in Part 1.
2. **Our own historical declines** — systems, papers, and techniques recorded as declined/dropped across our docs, source analyses, session notes, and git history.
3. **Reference notes** — name collisions and dead links surfaced during the sweep.

---

## Part 1 — Comparison-table rows (carsteneu directory), surveyed at code level

The [ai-memory-comparison](https://github.com/carsteneu/ai-memory-comparison) directory lists ~74 open-source coding-agent memory systems. We have full analyses of 15 of its rows from earlier phases (see `research/sources/`). The remaining **59** were first triaged light-touch (README + ≤2 docs) and then, on 2026-06-30, **surveyed at the code level** — one Opus agent per system: `git clone --depth 1`, read the retrieval / write-path / consolidation / decay implementations, cross-check the system's `evidence/<name>.md` against the code, and write a template-conforming analysis to `research/sources/<file>.md`.

**Why the deep read replaced the triage**: the light-touch pass returned **0 DIVE / 7 MAYBE / 50 SKIP**; the code-level survey returned **13 DIVE / 41 MAYBE / 5 SKIP** and **137 logged ideas**. Reading the code corrected the metadata triage in *both* directions — systems that looked strong by benchmark cell (memU 92.09, m_flow, OpenViking) stayed weak, but several near-SKIPs held real value (mcp-memory-service's benchmark harness, token-savior's bandit ranker, Midas's control-plane views). Light-touch triage demonstrably under-counts.

This part is therefore **no longer a list of declines** — it is the survey index. The *positive* output (every adoptable/considerable mechanism, mapped to a Somnigraph module) lives in **[`ideas-considered.md`](ideas-considered.md)**. What remains genuinely declined is the small "nothing to adopt" set at the end of this part. Every row below links to its full analysis; one-line comparisons and evidence cross-checks are in each source file and in [`index.md`](../research/sources/index.md).

### All 59, with verdict and source file

**Verdict semantics**: **DIVE** = holds an adopt-tier idea; **MAYBE** = one or more consider-tier angles; **SKIP** = surveyed, nothing to adopt. "Nug" = ideas logged to `ideas-considered.md`. This supersedes the light-touch MAYBE(7)/SKIP(50) tables that previously stood here.

| System | Verdict | Nug | Source file | One-line vs Somnigraph |
|--------|---------|-----|-------------|------------------------|
| ai-memory (akitaonrails) | DIVE | 3 | [ai-memory.md](../research/sources/ai-memory.md) | Mirror image of Somnigraph: plain fixed-k RRF retrieval (no reranker/feedback) but a rich write/maintenance path (git-wiki supersession, admission webhooks, eval-gated LLM auto-improve with a rejection buffer) — write-path is where its s... |
| ClawMem | DIVE | 4 | [clawmem.md](../research/sources/clawmem.md) | Broader write/consolidation path (observer-model extraction, entity resolution, subject-anchor merge guards) but zero measured retrieval quality — a synthesis of QMD/SAME/Thoth with an off-the-shelf pretrained reranker vs Somnigraph's le... |
| EverOS | DIVE | 3 | [everos.md](../research/sources/everos.md) | Local-first markdown+LanceDB runtime wrapping closed everalgo wheels; stronger on atomic-fact/MaxSim retrieval granularity and human-editable storage, but has no learned reranker, no feedback loop, and no graph/decay that are Somnigraph'... |
| Honcho | DIVE | 2 | [honcho.md](../research/sources/honcho.md) | Complementary to Somnigraph: Honcho invests heavily in the write path (reasoning-first extraction, cosine-95 dedup with token-superiority + times_derived reinforcement, surprisal-targeted dream consolidation) but leaves retrieval plain (... |
| m_flow | DIVE | 4 | [m-flow.md](../research/sources/m-flow.md) | Makes the graph the scorer via min-cost evidence-path propagation up a 4-layer cone graph (no LLM at query time) — a structural answer to Somnigraph's multi-hop vocabulary-gap ceiling — but is entirely hand-tuned with no learned reranker... |
| mcp-memory-service | DIVE | 3 | [mcp-memory-service.md](../research/sources/mcp-memory-service.md) | Broad-surface local MCP server (REST/OAuth/4 backends) that is behind Somnigraph on ranking (cosine + textbook RRF, no learned reranker, no feedback loop, rule-based consolidation) but ships a reproducible zero-LLM LongMemEval-S retrieva... |
| Memory Palace | DIVE | 4 | [memory-palace.md](../research/sources/memory-palace.md) | Safety-first, human-gated MCP memory whose write-time ADD/UPDATE/NOOP/DELETE dedup guard, simulate-before-mutate forgetting, and provenance-hash staleness are the adopt-worthy write-path ideas Somnigraph lacks — but its retrieval (64-dim... |
| memory-lancedb-pro | DIVE | 4 | [memory-lancedb-pro.md](../research/sources/memory-lancedb-pro.md) | Weaker retrieval (weighted-linear fusion mislabeled RRF, outsourced cross-encoder, no query-time graph, no learned/feedback reranker) but a much more built-out write path than Somnigraph — synchronous typed-relationship dedup, a groundin... |
| Midas | DIVE | 4 | [midas.md](../research/sources/midas.md) | A leaner, no-LLM, retrieval-only cousin of Somnigraph (no graph/reranker/feedback loop) whose real gifts are deterministic state/diff control-plane views, a dumb-reader eval-honesty floor, and bitemporal currency surfaced at recall time. |
| omega-memory | DIVE | 4 | [omega-memory.md](../research/sources/omega-memory.md) | Broader/pragmatic local-first pipeline with write-time contradiction/supersession, HyDE, and a stock cross-encoder reranker — where Somnigraph is narrower but more principled (learned LightGBM reranker, measured float feedback loop, LLM-... |
| Origin (Wenlan) | DIVE | 3 | [origin.md](../research/sources/origin.md) | Stronger on write-path/provenance discipline (quality gate, store-time dedup, faithfulness-gated distillation) but weaker on retrieval — no learned reranker or feedback loop, and its two rerankers plus CE⊕RRF blend all ship off by defaul... |
| token-savior | DIVE | 3 | [token-savior.md](../research/sources/token-savior.md) | A code-agent token-compaction tool whose secondary memory subsystem ships a live LinUCB contextual-bandit injection ranker (delayed implicit reward) that directly maps onto Somnigraph's open proactive-injection design, but has no memory-... |
| YesMem | DIVE | 4 | [yesmem.md](../research/sources/yesmem.md) | Write-path-heavy Go system with hand-tuned scoring (no learned reranker, nominal RRF) that hits 0.87 end-to-end LoCoMo QA on ingestion discipline — behind Somnigraph on retrieval but ahead on write-path quality gating and provenance. |
| Acontext | MAYBE | 3 | [acontext.md](../research/sources/acontext.md) | Mirror-image bet to Somnigraph: memory as human-editable Markdown "skill" files with a rich LLM write path but deliberately dumb grep/progressive-disclosure retrieval — no reranker, feedback loop, graph, decay, or benchmarks. |
| agentmemory (rohitg00) | MAYBE | 1 | [agentmemory-rohitg00.md](../research/sources/agentmemory-rohitg00.md) | Broad TS agent-memory product on iii-engine with untuned 3-way RRF, LLM-extracted graph, and a default-off generic cross-encoder — engineering breadth but strictly weaker retrieval science than Somnigraph's learned reranker + feedback lo... |
| AIPass | MAYBE | 2 | [aipass.md](../research/sources/aipass.md) | A multi-agent CLI workspace whose memory is a minimal file-JSON-hot + ChromaDB-cold vector store (no rerank/fusion/graph/feedback/benchmark); Somnigraph dominates on every retrieval-quality axis, but its symbolic subsystem contributes an... |
| ArcRift | MAYBE | 2 | [arcrift.md](../research/sources/arcrift.md) | A local-first consumer chat-memory product (extension + MCP) whose retrieval is naive max-score union + 1-hop SQL graph — architecturally far below Somnigraph's RRF/reranker/PPR/sleep stack — but its query-time HyDE targets the exact mul... |
| claude-mem | MAYBE | 2 | [claude-mem.md](../research/sources/claude-mem.md) | Strong automatic transcript-to-memory write path but a thin FTS5-or-Chroma-fallback retrieval layer (no RRF, no reranker, hard 90-day recency cutoff, no decay/graph/consolidation) — a generation behind Somnigraph on everything except aut... |
| context-infra | MAYBE | 1 | [context-infra.md](../research/sources/context-infra.md) | A prompt-driven markdown observation log with cron-fired LLM observer/reflector agents; strictly weaker retrieval/consolidation than Somnigraph, but its promote-durable-memories-into-an-always-loaded-tier pattern is a lever Somnigraph ha... |
| Continuity v2 | MAYBE | 2 | [continuity-v2.md](../research/sources/continuity-v2.md) | A read-only FTS5+embedding index over raw Claude Code JSONL transcripts with fixed-weight hybrid scoring and BFS thread recall - no curation, extraction, learned ranking, feedback, or consolidation; the inverse of Somnigraph's curated le... |
| engram (Gentleman-Programming) | MAYBE | 2 | [engram-gentleman.md](../research/sources/engram-gentleman.md) | FTS5-only Go agent-memory with cloud sync and save-time LLM-judged conflict surfacing; retrieval is strictly weaker than Somnigraph (no embeddings/rerank/graph/decay, no benchmarks), but its cheap-detect/expensive-judge conflict loop and... |
| fidelis | MAYBE | 2 | [fidelis.md](../research/sources/fidelis.md) | A thin zero-LLM verbatim-retrieval wrapper over mem0/Chroma with no learned ranking, feedback, graph, consolidation, or decay — weaker than Somnigraph on every differentiator axis, but its offline-built query-side vocab_map targets the e... |
| gbrain | MAYBE | 3 | [gbrain.md](../research/sources/gbrain.md) | A shipping multi-user "company brain" product whose real edge is deterministic write-time typed-KG construction (the gap Somnigraph names in its own lacks list), but with a weaker retrieval stack (external default-off cross-encoder, no l... |
| gitmem | MAYBE | 1 | [gitmem.md](../research/sources/gitmem.md) | A coding-agent mistake-avoidance guardrail (scars/wins with an enforcement protocol), not a ranking engine — simpler retrieval, no learned reranker/feedback/sleep, and zero end-to-end benchmark vs Somnigraph's 85.1 LoCoMo QA. |
| icarus | MAYBE | 3 | [icarus.md](../research/sources/icarus.md) | A git-native provenance/versioning coherence tool for coding agents with deliberately basic retrieval (token-overlap default, untuned RRF k=60) - opposite of Somnigraph's retrieval-quality focus, but its trust/supersession schema is a li... |
| Jumbo | MAYBE | 2 | [jumbo.md](../research/sources/jumbo.md) | A goal-driven, event-sourced project-context orchestrator whose retrieval is SQL LIKE + a graph join (no vector/BM25/rerank/decay/benchmarks) — vastly weaker than Somnigraph on retrieval, but different in shape: curate-and-join task-scop... |
| Kage | MAYBE | 3 | [kage.md](../research/sources/kage.md) | Code-grounded team memory with deterministic evidence-based freshness (withhold-on-drift) and a real write-path quality gate — orthogonal to Somnigraph's learned-retrieval, time-decay, single-user design; retrieval is weaker (lexical BM2... |
| LangMem | MAYBE | 3 | [langmem.md](../research/sources/langmem.md) | A thin LLM-prompted CRUD/extract layer over LangGraph's vector BaseStore (no fusion, reranker, feedback, decay, or working graph) plus a distinctive prompt-optimization toolkit — far behind Somnigraph on retrieval, ahead only on deferred... |
| MarsNMe | MAYBE | 1 | [marsnme.md](../research/sources/marsnme.md) | An agent-agnostic MCP continuity gateway over Supabase/pgvector (vector-only, no BM25/graph/reranker/feedback/benchmark) whose value is lifecycle plumbing, not retrieval — far behind Somnigraph on every research axis but with one transfe... |
| mem9 | MAYBE | 2 | [mem9.md](../research/sources/mem9.md) | A productized TiDB-backed Mem0-clone: synchronous LLM ADD/UPDATE/DELETE write path is its one real idea, but retrieval is plain RRF with no reranker/feedback/consolidation/decay, all of which Somnigraph has and does better (85.1% Opus QA... |
| memanto | MAYBE | 2 | [memanto.md](../research/sources/memanto.md) | A typed-memory management wrapper whose entire retrieval/ranking/answer stack is a single call into the proprietary Moorcheh engine — the inverse of Somnigraph, which owns and measures its reranker (reranker.py), RRF+PPR fusion (scoring.... |
| MemLayer | MAYBE | 1 | [memlayer.md](../research/sources/memlayer.md) | A salience-gated LLM-memory library (ChromaDB cosine + write-time NetworkX entity graph) with no fusion, no reranker, no feedback, no benchmarks — strictly behind Somnigraph on retrieval; its one real idea is a cheap write-time salience ... |
| memoir | MAYBE | 2 | [memoir.md](../research/sources/memoir.md) | Git-versioned semantic-path store with a strong typed write-path merge policy but no learned reranker, no graph/PPR, no feedback loop, and no QA benchmarks — retrieval is an LLM scanning every path, which Somnigraph's RRF+reranker alread... |
| Memora | MAYBE | 4 | [memora.md](../research/sources/memora.md) | Simpler-retrieval MCP server (weighted RRF, no reranker/feedback/sleep, brute-force cosine) but with a genuinely disciplined synchronous write path and live graph UI — corroborates our Phase 18 write-path finding. |
| Memori | MAYBE | 2 | [memori.md](../research/sources/memori.md) | Memori's open SDK is a thin client (dense+max-normalized-BM25 linear fusion, no reranker/RRF/graph-traversal/decay/feedback) around a closed cloud extractor; it sits a generation behind Somnigraph on every retrieval and lifecycle axis, a... |
| MemoryBear | MAYBE | 1 | [memorybear.md](../research/sources/memorybear.md) | Large graph-first (Neo4j) production memory product with real write-time extraction/dedup and a genuine ACT-R forgetting engine, but no learned reranker, no feedback loop, and lower self-reported LoCoMo QA (75.0/72.9 J vs our 85.1); its ... |
| memU | MAYBE | 2 | [memu.md](../research/sources/memu.md) | memU wins on the write path (multimodal extraction + self-refining skill notes from tool traces) but is thin on retrieval (cosine-only + per-tier LLM orchestration, no BM25/RRF/reranker, no real graph), where Somnigraph is far stronger. |
| Memvid | MAYBE | 3 | [memvid.md](../research/sources/memvid.md) | A polished portable single-file .mv2 memory FORMAT (Rust) with regex-SPO entities and RRF-fused lexical/vector RAG, but thinner on retrieval intelligence than Somnigraph (no learned reranker, no graph traversal, no consolidation/decay/fe... |
| MoltBrain | MAYBE | 2 | [moltbrain.md](../research/sources/moltbrain.md) | A hook-driven passive session-transcript recorder (SQLite+FTS5+optional ChromaDB, observer subprocess) with no fusion, no learned reranker, no graph, no consolidation beyond summaries, and no lifecycle — strictly weaker than Somnigraph o... |
| Nanobot | MAYBE | 2 | [nanobot.md](../research/sources/nanobot.md) | A standalone agent framework whose memory is LLM-curated markdown files + git versioning with whole-file injection and manual grep — no retrieval engine at all (no vector/BM25/graph/reranker), so architecturally distant from Somnigraph; ... |
| nocturne | MAYBE | 3 | [nocturne.md](../research/sources/nocturne.md) | A deliberately anti-automation, human-curated tree/version memory with FTS-only retrieval and visual diff/rollback — far weaker than Somnigraph on retrieval, but stronger on write-path human review and write-time recall-condition authoring. |
| Noosphere | MAYBE | 2 | [noosphere.md](../research/sources/noosphere.md) | A Postgres-FTS wiki for agents whose retrieval/consolidation is well behind Somnigraph (no vector, no learned reranker, no LLM sleep), but whose write-time quality gating hits Somnigraph's own acknowledged gap. |
| obsidian-mind | MAYBE | 1 | [obsidian-mind.md](../research/sources/obsidian-mind.md) | Not a memory engine but an Obsidian-vault + multi-agent scaffold that rents all retrieval (BM25/vector/RRF/LLM-rerank) from upstream @tobilu/qmd; Somnigraph owns a far stronger self-built retrieval stack, while this repo's only original ... |
| Octopoda-OS | MAYBE | 2 | [octopoda-os.md](../research/sources/octopoda-os.md) | A multi-tenant memory SaaS (Synrix engine) strong on platform ops (RLS tenancy, hash-chained audit, 10-classifier loop detection) but far weaker on retrieval than Somnigraph — prefix lookup + single-channel cosine, no fusion, no reranker... |
| opencode-mem | MAYBE | 2 | [opencode-mem.md](../research/sources/opencode-mem.md) | A thin-retrieval OpenCode plugin (fixed 0.6/0.4 content-vs-tags cosine blend, no FTS/RRF/reranker/graph/decay) whose only Somnigraph-relevant idea is a confidence-reinforced user-persona layer that maps to the PERMA preference-tracking gap. |
| OpenMemory | MAYBE | 2 | [openmemory.md](../research/sources/openmemory.md) | Heuristic 5-sector (regex-classified) multi-vector memory with a real-time waypoint graph; no LLM extraction, no learned reranker, no benchmarks — weaker than Somnigraph everywhere except it has write-time SimHash dedup that Somnigraph l... |
| second-brain | MAYBE | 2 | [second-brain.md](../research/sources/second-brain.md) | A one-file serverless Cloudflare memory layer with synchronous write-path dedup/merge/contradiction gating and nightly tag-digest compression, but a LIKE-based hybrid + hand-tuned rerank formula that Somnigraph's FTS5 BM25 + learned 26-f... |
| shodh-memory | MAYBE | 4 | [shodh-memory.md](../research/sources/shodh-memory.md) | A no-LLM, offline Rust cognitive-memory substrate (Hebbian graph + PPR spreading activation + power-law decay) that lands on the same PPR bet as Somnigraph but competes on algorithmic retrieval with a weaker scoring stage and no audited ... |
| stash | MAYBE | 3 | [stash.md](../research/sources/stash.md) | Richer write-path/consolidation taxonomy (facts, causal links, hypotheses, goals, failures) than Somnigraph, but retrieval is pure single-channel pgvector with no reranker, hybrid, feedback, or benchmarks — strictly weaker where Somnigra... |
| TeleMem | MAYBE | 2 | [telemem.md](../research/sources/telemem.md) | A narrow Mem0 fork for Chinese roleplay dialogue: single-channel dense retrieval (reranker off by default), no graph/feedback/decay/sleep — weaker than Somnigraph everywhere except its write-time cluster-dedup and per-character extraction. |
| TencentDB-AM | MAYBE | 2 | [tencentdb-am.md](../research/sources/tencentdb-am.md) | Richly-layered L0-L3 personalization + Mermaid context-offload with real write-time dedup, but plain RRF retrieval (no reranker/feedback/graph) where Somnigraph is strongest; its write-path gate is the one idea worth borrowing. |
| VIR | MAYBE | 3 | [vir.md](../research/sources/vir.md) | VIR is a write-path distillation tool (confidence-gated transcript-to-markdown) with trivial retrieval; the mirror image of Somnigraph's research-grade retrieval stack bolted onto an ungated remember() write path. |
| Wax | MAYBE | 3 | [wax.md](../research/sources/wax.md) | A fast, offline Swift/Metal single-file RAG engine with hand-tuned heuristic ranking — engineering-strong but memory-shallow next to Somnigraph's learned reranker, feedback loop, and LLM sleep consolidation. |
| YourMemory | MAYBE | 3 | [yourmemory.md](../research/sources/yourmemory.md) | A well-engineered product with a disciplined mostly-LLM-free write path (dedup/salience/NER entity-graph) but a generation behind Somnigraph on retrieval — no learned reranker, no feedback loop, no typed graph, no multi-phase sleep; its ... |
| CommonGround | SKIP | 1 | [commonground.md](../research/sources/commonground.md) | Category mismatch: CommonGround is a multi-agent coordination ledger (claim fencing, turn lifecycle, causal lineage) with zero retrieval/extraction/decay, i.e. the substrate a memory system like Somnigraph would sit on top of, not a comp... |
| Fullerenes | SKIP | 0 | [fullerenes.md](../research/sources/fullerenes.md) | A zero-LLM Tree-sitter code-graph indexer (symbols/callers/blast-radius over SQLite with LIKE search), out of scope versus Somnigraph's conversational persistent memory — no vectors, reranker, feedback, decay, or consolidation. |
| MemoMind | SKIP | 1 | [memomind.md](../research/sources/memomind.md) | Windows/dashboard packaging over the upstream Hindsight engine (already in our corpus as hindsight.md/hindsight-paper.md); adds only localization patches and one super-hub entity filter, no benchmarks, nothing adopt-worthy Somnigraph doe... |
| memorix | SKIP | 0 | [memorix.md](../research/sources/memorix.md) | A pre-alpha generic vector-store SDK whose embedders and backends are all stubs; a strict, non-functional subset of Somnigraph with nothing to adopt. |
| OpenViking | SKIP | 0 | [openviking.md](../research/sources/openviking.md) | A filesystem-paradigm "context database" whose token-saving L0/L1/L2 tiering is context-shaping not retrieval-quality; its two differentiating ranking knobs (hotness decay, directory score-propagation) ship default-disabled and its write... |

### Genuinely declined — surveyed, nothing to adopt (5)

These five were read at the code level and yielded no adoptable or considerable mechanism beyond what Somnigraph already has. Reason preserved so a future session does not redo the read.

| System | Why declined |
|--------|-------------|
| [CommonGround](../research/sources/commonground.md) | Category mismatch: CommonGround is a multi-agent coordination ledger (claim fencing, turn lifecycle, causal lineage) with zero retrieval/extraction/decay, i.e. the substrate a memory system like Somnigraph would sit on top of, not a competitor. |
| [Fullerenes](../research/sources/fullerenes.md) | A zero-LLM Tree-sitter code-graph indexer (symbols/callers/blast-radius over SQLite with LIKE search), out of scope versus Somnigraph's conversational persistent memory — no vectors, reranker, feedback, decay, or consolidation. |
| [MemoMind](../research/sources/memomind.md) | Windows/dashboard packaging over the upstream Hindsight engine (already in our corpus as hindsight.md/hindsight-paper.md); adds only localization patches and one super-hub entity filter, no benchmarks, nothing adopt-worthy Somnigraph doesn't already do better. |
| [memorix](../research/sources/memorix.md) | A pre-alpha generic vector-store SDK whose embedders and backends are all stubs; a strict, non-functional subset of Somnigraph with nothing to adopt. |
| [OpenViking](../research/sources/openviking.md) | A filesystem-paradigm "context database" whose token-saving L0/L1/L2 tiering is context-shaping not retrieval-quality; its two differentiating ranking knobs (hotness decay, directory score-propagation) ship default-disabled and its write-time memory graph never touches retrieval scoring, so shipped retrieval is plainer than Somnigraph's learned-reranker + RRF + PPR + feedback stack. |

**Note on the previously "not assessed" pair**: MemoMind and mem9 hit the structured-output retry cap during the light-touch sweep and returned no verdict. Both were fully read in this survey — mem9 is MAYBE (2 crumbs: shadow-mode dedup metric, integer-ID indirection), MemoMind is SKIP (packaging over the upstream Hindsight engine, already in our corpus).

---

## Part 2 — Our own historical declines

Recorded across our docs, source analyses, session notes, and git history. Distinct from the comparison-table triage above.

*(Note: PERMA is **not** here — it is an active Priority 5 in `STEWARDSHIP.md`, deferred not declined, despite the "Not Useful" subsection in `perma.md` that targets only MCQ-format specifics.)*

### Systems (examined, judged not useful as a whole, or cited as a rejected approach)

| Item | Where recorded | Reason |
|------|----------------|--------|
| Dynamic Cheatsheet | `research/sources/dynamic-cheatsheet.md` | "Nothing to adopt." Single-document-rewritten-each-step is lossy (~2500-word cap); DB-backed typed entries categorically superior. |
| Rashomon Memory | `research/sources/rashomon-memory.md` | Multi-agent argumentation; single-user memory has no natural perspective decomposition; O(N²) LLM calls/query; proof-of-concept, no quantitative eval. |
| Neuroca | `research/sources/neuroca.md` | Alpha; ~110K lines of AI-generated scaffolding, integration tests all skipped; "## Not Worth It." |
| AriGraph (exact-match entity resolution) | `docs/similar-systems.md § RedPlanet CORE`, `vestige-fsrs.md` | Exact-match entity resolution is our Phase 14 negative result; cited only as a rejected approach. |
| Recall / recall-memory-substrate (as a stack) | `research/sources/recall-substrate.md` | Node-24-only, no Python/SQLite interop; n-ary hyperedges + per-actor Brier calibration assume multi-writer graphs (zero payoff single-user); SENTINEL scores pull-systems 0 by construction. |
| ByteRover (as a stack) | `research/sources/byterover.md` | Context-Tree HTML-file-per-entry and git-like branch/merge are team-collaboration features, unjustified at ~300-memory single-user scale. |
| Memobase (profile-only model) | `research/sources/memobase.md`, `docs/similar-systems.md` | "Dump all profiles" has near-zero retrieval intelligence; profile-only bet too narrow; PERMA multi-domain collapse (0.248) shows the cost. |
| MIRIX (multi-agent write path) | `research/sources/mirix.md` | 6 sub-agent LLM calls per input prohibitively expensive for write-at-decision-time; run with `chaining=False` even at eval; per-type tables + Knowledge-Vault are multi-user concerns. |
| TrueMemory (tier machinery) | `research/sources/truememory.md` | Multi-tier local embeddings, NaN-migration for a macOS SDPA bug, Dunbar social-distance tiers, HDBSCAN — "none of this translates" for a single-user system with API access. |
| SuperLocalMemory (channels) | `research/sources/superlocalmemory.md` | Code-KG is coding-agent-specific; Hopfield channel lowest-weighted/unmeasured; auto-cognitive hooks remove agent judgment (against our explicit-capture design). |

### Papers / Benchmarks (surveyed, marked low-relevance / out-of-scope / non-comparable / not worth running)

| Item | Where recorded | Reason |
|------|----------------|--------|
| Graph-Aware Late Chunking | `graph-aware-late-chunking.md` | "Limited relevance"; entity-based retrieval hurts MRR (biomedical UMLS domain). |
| Graphs RAG at Scale | `graphs-rag-at-scale.md` | "Domain mismatch"; LPG + text-to-Cypher for structured financial data. One salvaged idea (edge-type-aware PPR weighting). |
| MemCollab | `memcollab.md` | "Limited relevance (single-agent system)"; contrastive cross-agent memory sharing. |
| ProGRank | `progrank.md` | "Low relevance (single-user, API embeddings)"; adversarial corpus-poisoning defense. |
| Knowledge Access Beats Model Size | `knowledge-access.md` | "Routing irrelevant to single-instance." Kept two data points only (timestamp-as-text −3.8 F1; LLM summaries can poison RAG). |
| First-Mover Bias in GBMs | `first-mover-bias.md` | Theoretical; not directly actionable (we use metric-driven, not SHAP-driven, feature selection). |
| Contradiction Reconciliation | `contradiction-reconciliation.md` | Dataset mismatch: short NLI pairs vs. paragraph/weeks-spanning personal contradictions; REG is a research task, not an adoption. |
| Geometry of Forgetting | `geometry-of-forgetting.md` | Theoretical; bAbI POC (0.475) not competitive; validated the move off time-based decay but "nothing to adopt." |
| MSA (Memory Sparse Attention) | `msa.md` | Model-layer architecture (trainable attention/KV compression), not an application-level memory system. |
| HippoRAG v2 recognition-memory step | `hipporag.md:258` | LLM triple-filtering showed −0.7% ablation impact; not worth prompt-optimization complexity for marginal gains. |
| WRIT | `writ.md:258` | Thesis worth stealing, but running the benchmark itself not worth it at 50 scenarios (4 commits, very early). |
| A-MBER | `a-mber.md:212` | No public code/data; not worth building a pipeline unless the data ships. |
| AMB (Agent Memory Benchmark) | `amb.md` | Conflict-of-interest: Hindsight-team harness where their own tuned adapter tops the leaderboard. |
| 2WikiMultiHopQA | `2wikimultihop.md` | Benchmark has significant shortcuts (DiRe 63.4); only the vocabulary-gap taxonomy was harvested. |
| MemoryBench (Chinese split) | `memorybench.md:210` | Language dimension orthogonal to memory-architecture evaluation; skip unless multilingual deployment. |
| MemoryAgentBench (TTL/LRU) | `memoryagentbench.md:139` | TTL and LRU test different capabilities than what our system provides. |
| context-mem 98% LoCoMo claim | `context-mem.md` | Non-comparable: session-level granularity + benchmark-specific synonyms, not turn-level. |
| Cross-vendor LoCoMo/LME numbers | `index.md`, `similar-systems.md` | Repeatedly marked non-comparable across vendors (differing judges/configs). |

### Techniques / Features / Experiments (ours — implemented-and-removed, or considered-and-rejected)

| Item | Where recorded | Reason |
|------|----------------|--------|
| Parametric scoring formula (14-param) | `architecture.md` § The parametric scoring formula (superseded), commit `b5491cb` | Superseded by LightGBM reranker; couldn't express interactions/metadata; kept as fallback only. |
| Quality floor (`quality_floor_ratio`) | `architecture.md` § Quality floor (removed), `roadmap.md` § What we'd tell someone starting from scratch | wm1 found optimal ratio = 0.0; fixed ratio can't adapt to score distributions. |
| Cliff detector (`apply_quality_floor`) | commit `bce9793`, `roadmap.md` § Can cutoff history calibrate the cliff detector? | Over-delivers 96.1% of the time (avg +7.4 memories); score features can't predict cutoff (R²=−0.215); replaced by agent `limit`. |
| Shadow penalty in scoring | `architecture.md` § Shadow penalty in scoring (removed) | 9/10 reviewers recommended removal; conflates temporal with query relevance. Retained for dormancy only. |
| Confidence weight in scoring | `architecture.md` § Confidence weight in scoring (removed) | <0.1% contribution; correlates with feedback → double-counts. Retained as metadata. |
| Diversity feature (MMR / `max_sim_to_higher`) | `experiments.md` § Diversity feature (max_sim_to_higher): negative result | Hurt NDCG −0.3% despite #6 importance; pointwise model can't coordinate selections; 70× latency (O(n²)). |
| Intent routing (query-type classifier) | `architecture.md` § Intent routing (removed) | 0% fire rate; queries don't decompose cleanly into intents. |
| LambdaRank objective | `experiments.md` § LambdaRank: negative result, then parity | Negative then parity; "signal is in the features, not the loss"; pointwise wins on simplicity. |
| Group C cross-result features | commit `2bf4cc3` | Hurt generalization despite high gain — classic overfitting. |
| Dead reranker features (proximity, age_days, theme_count) | commit `2bf4cc3` | Confirmed 0.0 gain; dropped. |
| Capitalization-based entity extraction | `benchmarks.md` § What didn't work: capitalization-based entity extraction | "Capitalization is a feature of edited prose, not casual dialog"; replaced by ~200-entity allowlist. |
| Expansion methods: rocchio, multi_query, entity_focus | session `2026-03-22-two-phase-expansion-fixes.md` | Dead: 0% / 2% / 4% contribution in LoCoMo ablation. |
| `-1.0` sentinel for missing feedback features | `architecture.md` § `-1.0` sentinel for missing feedback features | Read as "worse than any real value" → demoted every cold-start memory; fixed to NaN. |
| Masquerading sentinels (rank/dist/norm features) | `architecture.md` § Sentinel-encoded missing values | Defaults read as "better than top result"; 7 NaN-encoded, pathologies 239→37 (−85%). |
| `fts_bm25_norm` always-zero bug | `architecture.md` § `fts_bm25_norm` normalization always-zero bug | `if max_fts>0` never fired (FTS5 scores negative); dead weight (std=0 across 178K samples). |
| Single-shape synthetic GT `(summary,summary,1.0)` | `architecture.md` § Single-shape synthetic GT | Goodhart-confirmed: dropped audit pathologies to 0 but content-residual OOD worsened; taught shape-memorization. |
| Audit-based pathology selector (as primary path) | `STEWARDSHIP.md` §2, `architecture.md` § The audit's ceiling | Synthetic audit pathologies FTS-handicapped or Goodhart-correlated by construction; demoted to fallback. |
| `audit_reranker_pathology.py` content-residual OOD test | `architecture.md` § The audit's ceiling | Structurally flawed — handicaps target's FTS while leaving imposters' intact; residual is an audit artifact. |
| Seed-curation step (sleep) | commit `601ea35` | Opus call loaded ~50k tokens, timed out repeatedly; cost not justified — seed curated manually during /reflect. |
| Theme-suggestion queue / heuristic threshold second-guessing | commits `a0b79fb`, `b61bf7d` | "The LLM review is the quality gate. No second-guessing with heuristic thresholds." |
| Parallel NREM batch processing | `architecture.md` § NREM: Relationship detection | Abandoned due to Windows memory constraints with subprocess-based LLM calls. |
| Distance-based novelty / prediction-error gating | `vestige-fsrs.md:92` | Phase 14 Exp 5: Spearman −0.1952 — closer memories are MORE useful; the gating premise is inverted at our scale. |
| FSRS power-function decay (vs exponential) | `vestige-fsrs.md:122` | Phase 14 Exp 3 null result — no difference at our timescale. |
| Retrospective Opus re-rating of live GT (Path A) | session `2026-05-09-phase2-...:33` | Rejected: production rater pool is already ~uniformly Opus with full context; a bounded-context judge can't beat it. |
| Calibrate-cliff-from-score-features | `roadmap.md` § Can cutoff history calibrate the cliff detector? | Negative: linear R²=−0.215; per-position logistic dominated by rank position; scores add nothing. |
| `target_ids` in RATE_TEMPLATE | session `2026-05-09-phase3abc-plumbing.md:17` | Leaked the answer to the LLM rater, biasing neighbor labels; removed from prompt. |

---

## Part 3 — Reference notes (collisions & dead links)

Surfaced while mapping the 74 comparison rows against our corpus.

**Name collisions — the table row is a *different project* than our analyzed file:**

- **`engram`**: our `research/sources/engram.md` = `Harshitk-cp/engram` (Go + PostgreSQL, HTTP). The table row = `Gentleman-Programming/engram` (Go binary, TUI, FTS5-only). Different projects, same name; the table's engram now has its own analysis at [`engram-gentleman.md`](../research/sources/engram-gentleman.md) (MAYBE). `engram.md` was **not** overwritten.
- **`agentmemory`**: our `research/sources/agentmemory.md` = `JordanMcCann/agentmemory` (6-signal weighted sum, LongMemEval 96.2%). The table row = `rohitg00/agentmemory` (3-way RRF, 53 MCP tools, LongMemEval 95.2% recall-only) — confirmed a **distinct project**, now analyzed at [`agentmemory-rohitg00.md`](../research/sources/agentmemory-rohitg00.md) (MAYBE). `agentmemory.md` was **not** overwritten.

**Distinct-but-similar-name (confirmed different systems, own files written):** `Memvid` (video-container, ≠ our `memv.md`), `Memory Palace` (≠ `MemPalace`/`mempalace.md`), `gitmem` (≠ `diffmem.md`), `claude-mem` (≠ our claude-sleep/-total-memory/-subconscious/-cognitive), `EverOS` (distinct product, same org as analyzed EverMemOS).

**Dead links — all recovered during the code-level survey** (the earlier sweep's 404s were renames / non-GitHub distribution):
- `mcp-memory-service` → live at [`github.com/doobidoo/mcp-memory-service`](https://github.com/doobidoo/mcp-memory-service) (DIVE — holds the LongMemEval retrieval harness).
- `MemoryBear` → renamed to [`SuanmoSuanyangTechnology/MemoryBear`](https://github.com/SuanmoSuanyangTechnology/MemoryBear) (MAYBE; ACT-R decay *is* implemented, correcting the earlier "unrealized" note).
- `m_flow` → GitHub 404s, but source recovered from the PyPI sdist [`mflow-ai==0.3.6`](https://pypi.org/project/mflow-ai/) (DIVE — min-cost evidence-path scoring).
- `memorix` → [`github.com/memorix-ai/memorix-sdk`](https://github.com/memorix-ai/memorix-sdk) (SKIP — pre-alpha, embedders/backends all stubs).

---

*Last swept: 2026-06-30 (code-level survey of all 59 remaining rows; supersedes the 2026-06-30 light-touch triage). Each row has a full analysis in `research/sources/`; adoptable/considerable ideas are consolidated in [`ideas-considered.md`](ideas-considered.md). Comparison directory content as of the carsteneu `main` branch on that date. SKIP verdicts are revivable on new evidence.*
