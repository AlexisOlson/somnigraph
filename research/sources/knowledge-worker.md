# Knowledge-Worker — Human-curated knowledge graph with graph-analytic discovery layer

*Generated 2026-06-28 by Sonnet agent reading local clone*

**Repo**: https://github.com/rahulmranga/knowledge-worker (MIT)

---

## Architecture

### Storage & Schema

Plain JSON (`mygraph.json`), stdlib-only core. No SQLite, no vector index, no embeddings. Two primitives:

- **Node** (`mygraph.py:63-70`): `id`, `type`, `label`, `body`, `confidence` (high/medium/low), `created_at`.
- **Edge** (`mygraph.py:76-85`): `src`, `dst`, `type`, `source_id`, `excerpt`, `confidence`, `created_at`, `last_seen`.

Node types: `person`, `topic`, `idea`, `project`, `goal`, `question`, `decision`, `reference`, `source`. Edge types: `HAS_IDEA`, `RELATES_TO`, `SUPPORTED_BY`, `CHALLENGES`, `SERVES`, `INVOLVES`, `ABOUT`, `MENTIONED_IN`, `MADE_AT`.

`last_seen` on edges is refreshed by the merger on re-ingest and feeds the staleness radar in `discover.py`.

### Memory Types

No multi-tier memory model. All content is a flat typed node graph. The only gradient is `confidence` (high/medium/low), enforced by the validator with an excerpt-presence gate: `high` requires a literal quote from the source text; the validator demotes to `medium` when no excerpt is provided. Source nodes serve a provenance-anchor role analogous to episodic memory but are not partitioned from conceptual content at storage level.

The confidence system has enforcement teeth: the validator checks excerpt presence at ingest time, not as a soft warning. An extractor that proposes a `high`-confidence node with no excerpt gets demoted to `medium` before the human review step — the reviewer sees the demotion flag. This is stronger than Somnigraph's confidence field, which is set at write time by the LLM and is not subsequently validated against any evidentiary standard.

A separate `state_log.jsonl` sidecar captures mood/state entries via `mykg state "..."` without touching the main graph (per SPEC §5). Explicitly by design.

### Write Path

Five-stage ingest pipeline with a mandatory human review gate:

1. **Extract** — LLM (Claude/OpenAI/Ollama, pluggable backend) reads a markdown file and proposes candidate nodes and edges via structured tool call, requiring literal source excerpts.
2. **Validate** — Schema checks, ID checks, excerpt-presence enforcement. Demotes `high → medium` when excerpt is absent.
3. **Review** — Human CLI gate: accept, reject, or edit each candidate. No memory enters the graph without human sign-off.
4. **Merge** — Idempotent write; body-conflict resolution via diff (keep/replace/append prompt); auto-injects `MENTIONED_IN` edges when the extractor omits them.
5. **Eval log** — Appends a JSONL audit record per ingest run.

The human review gate is the architectural center. The system's design principle: "AI proposes, provenance verifies, the owner promotes." This is not a gap — it is the product.

An additional opt-in mode: `mykg deep-dive <source.md> --out-dir <workspace>` creates a pre-ingest reasoning workspace where LLM synthesis and adversarial challenge generation run before the standard pipeline. Candidate nodes are held in a separate file until the owner runs `deep-dive add-to-graph <workspace>`. Designed for sources where the signal-to-noise ratio is low enough that standard single-pass extraction would produce too many uncertain candidates for the review queue.

### Retrieval

Substring match on `id.lower()`, `label.lower()`, `body.lower()`, results sorted by `(type, label)` (`mygraph.py:166-174`). No vector index, no FTS5, no BM25, no RRF, no reranker, no feedback signal.

`mykg context` ranks ideas by incoming edge count (degree-centrality proxy) and caps output for LLM pastability. This is fixed-template rendering, not recall-quality retrieval.

### Consolidation

No automated consolidation. The `discover.py` module (the analytically interesting half of this repo) runs eight read-only analyses and emits all derived-edge proposals to a candidates file for human review — it never mutates the graph:

1. **Staleness radar** (`discover.py:149-200`): scores nodes by `normalized_PageRank × days_since_last_touch`; reference time is the newest timestamp in the whole graph (deterministic for a committed file).
2. **Co-mention inference** (`discover.py:205-237`): pairs appearing together in ≥N distinct source documents with no direct edge. Scored by source count.
3. **Serves candidates** (`discover.py:272-299`): ideas/decisions with Adamic-Adar neighborhood overlap with a goal but no directed contribution path — "work the graph can't yet explain."
4. **Related candidates** (`discover.py:302-323`): classic Adamic-Adar link prediction for non-adjacent pairs with ≥2 shared neighbors.
5. **Question debt** (`discover.py:327-373`): open questions scored by `centrality × (1 + age_days)`; detects when a decision's `ABOUT` edge answers a question.
6. **Corroboration** (`discover.py:378-402`): single-source claims ranked by PageRank — "one bad transcript away from being wrong."
7. **De-spined bridges** (`discover.py:407-460`): removes hub nodes, re-runs betweenness and Girvan-Newman community detection, surfaces real cross-domain connectors.
8. **Tensions** (`discover.py:465-510`): nodes with both `SUPPORTED_BY` and `CHALLENGES` edges, plus goal-contributor conflict propagation.

A `deep-dive` workspace mode (`deep_dive.py`) pre-processes complex sources with LLM synthesis and challenge generation before ingest, keeping uncertain inferences out of the main graph until the owner promotes them.

The `memory_audit.py` module produces PageRank, betweenness, bridge, and weak-claim panels as a governance HTML, also read-only.

### Lifecycle Management

No decay, no dormancy, no importance scoring, no deletion pipeline. Memories persist indefinitely. The `discover.py` staleness radar surfaces cold nodes as proposals, but cannot act without human intervention. `last_seen` on edges refreshes on re-ingest only. The explicit design choice: knowledge-worker treats forgetting as a human decision, not a system policy.

**Interop**: `mykg export --ttl` emits OWL/Turtle for external graph tool consumption. Not relevant for Somnigraph's SQLite+sqlite-vec stack, but notable as the only system in this corpus that provides a standards-compliant export path alongside its native storage.

**Demo data**: The `seed()` function (`mygraph.py:234-507`) populates a deterministic three-era graph (ERA1: greenhouse sensor build 2026-03-08, ERA2: field-notes newsletter 2026-05-02, ERA3: knowledge-worker toolkit 2026-05-28) designed to exhibit multiple community structures, stale ERA1 nodes, and bridge ideas like `idea:sensor-data-as-memory` that span embedded systems and knowledge graph topics. The deterministic timestamps make `discover.py` output reproducible, which serves as a functional test harness for the analytics.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Provenance-first reduces hallucination risk | Validator enforces literal excerpt; merger auto-injects MENTIONED_IN; coverage metrics tracked | **Plausible, unquantified** — no benchmark; design argument only |
| Human review gate preserves accuracy | Architectural claim; no ablation vs. auto-ingest pipeline | **Unvalidated** — reasonable but untested |
| De-spined betweenness reveals true bridges | Algorithm is analytically sound; demonstrated on deterministic demo graph | **Plausible** — correct in theory, small-scale demo only |
| Staleness × centrality prioritizes correctly | `discover.py:180-195`, normalized PageRank × days_cold; reference timestamp is newest in graph | **Plausible** — weighting direction is correct, no ground truth evaluation |
| Question debt score surfaces high-value open questions | `discover.py:327-373`, `centrality × (1 + age_days)` with decision-as-answer detection | **Plausible** — reasonable heuristic, no evaluation |
| Adamic-Adar link prediction surfaces missing edges | `discover.py:302-323`, score ≥ 1.0 with ≥ 2 shared neighbors threshold | **Plausible** — standard graph-theoretic method; threshold is tunable |
| Structural tension propagates through goal contributors | `discover.py:493-510`: if B challenges goal G and A serves G, B and A are in tension | **Plausible** — the logic is sound; false positives likely in sparse graphs |

---

## Relevance to Somnigraph

### What Knowledge-Worker does that Somnigraph doesn't

1. **Mandatory write-path quality gate**. knowledge-worker's 5-stage pipeline blocks autonomous LLM-extracted content from entering the graph without human sign-off. Somnigraph writes autonomously at session time. The tradeoff is explicit: Somnigraph favors capture latency; knowledge-worker favors provenance accuracy. The write-path gate is missing from Somnigraph's architecture (noted in `docs/architecture.md` as a known gap).

2. **Provenance enforcement with coverage metrics**. Every non-source node must carry a `MENTIONED_IN` edge to a source document with a literal excerpt. The validator enforces this; the merger auto-injects it. `node_coverage` and `excerpt_coverage` are tracked metrics. Somnigraph has no write-path provenance requirement.

3. **Question debt tracking** (`discover.py:327-373`). Open questions scored by `centrality × (1 + age_days)`. Decision nodes with `ABOUT` edges to question nodes count as answers and are detected automatically. REM sleep (`sleep_rem.py`) generates gap-analysis questions but does not track them as first-class graph citizens with aging debt scores.

4. **De-spined bridge detection** (`discover.py:407-460`). Removes the top-2 hub nodes (those exceeding 2× median nonzero betweenness), then re-runs betweenness and Girvan-Newman community detection on the remainder. Surfaces conceptual connectors masked by high-betweenness hubs. Not in Somnigraph.

5. **Corroboration scoring** (`discover.py:378-402`). Single-source claims ranked by PageRank as a review queue — the ones most worth verifying are both central and under-supported. Somnigraph has no equivalent; it stores utility scores but has no multi-source corroboration signal. For Somnigraph's use case (single-user session memory) this gap is less critical, but for procedural memories that arrive from one session and persist indefinitely, the risk is real.

### What Somnigraph does better

1. **Retrieval quality**. 31-feature LightGBM reranker, hybrid BM25 + vector RRF, PPR graph expansion, feedback loop with measured GT correlation. knowledge-worker's substring search (`mygraph.py:166-174`) is not retrieval in any comparative sense.

2. **Automated consolidation**. Three-phase sleep pipeline: NREM pairwise classification and edge creation, REM taxonomy and dormancy, probe_recall for underserved memories. knowledge-worker's `discover.py` is a read-only proposal generator; Somnigraph writes edges, merges duplicates, updates summaries, transitions memory states without waiting for human sign-off.

3. **Lifecycle management**. Per-category exponential decay with configurable half-lives, dormancy detection via `sleep_rem.py:1569`, importance-weighted probe targeting via `probe_recall.py` and `select_real_pathology_targets.py`. knowledge-worker treats forgetting as a human decision; Somnigraph models decay as a system property with configurable per-memory rates.

4. **Scale and embeddings**. Somnigraph operates on 300+ memories with 1536d vectors (or 384d fastembed). knowledge-worker targets ~dozens of manually curated nodes; its graph analytics are O(n²) and designed for small graphs where Brandes betweenness is fast enough to run at CLI invocation time.

5. **Feedback loop**. Explicit per-query utility ratings (0-1), EWMA aggregation, UCB exploration bonus, Hebbian PMI co-retrieval boost. knowledge-worker has no retrieval feedback signal of any kind.

6. **Entity nodes as retrieval hubs**. Somnigraph's `category="entity"` nodes carry curated summaries and are linked to related memories during NREM step 4c, enabling PPR walks from entity → detail cluster. knowledge-worker's `person` and `project` nodes play a similar hub role in the graph topology, but they participate in retrieval only via substring match on their `label` or `body` fields — no PPR, no edge-traversal at query time.

---

## Worth Stealing (ranked)

### 1. Staleness × centrality for probe targeting (Low effort)

**What**: Score dormant memories by `normalized_PPR × days_since_last_access` (`discover.py:180-195`). The reference point is the newest timestamp in the graph (deterministic for a committed state). High-centrality memories that have gone cold are higher-priority probe targets than peripheral memories cold for the same duration. The formula: `staleness_score = (ppr_i / max_ppr) × days_cold`.

**Why**: `probe_recall.py`'s selector prioritizes by low recent access and low utility scores, but does not weight by graph importance. A memory central to the PPR graph that has not been accessed in 90 days is a more costly miss than a low-centrality memory with the same access gap. The existing `select_real_pathology_targets.py` adversarial selector already computes gap scores (difference between expected utility and actual retrieval rank); staleness × centrality is a complementary orthogonal signal that covers the "important but forgotten" failure mode, whereas the real-recall pathology selector covers the "important but mis-ranked" failure mode.

**How**: In `scripts/probe_recall.py` after the eligible-list pass in `select_targets()`, apply `ppr_score_normalized × days_cold` as a secondary sort key for memories below the activity threshold. PPR scores are available from `scoring.py`'s `_expand_ppr()` return cache; `days_cold` is `(now - last_accessed) / 86400` from the memories table. The reference timestamp (newest `last_accessed` in DB) makes the score reproducible. Surface as a `--mode staleness_weighted` flag in `select_real_pathology_targets.py`, bundled into `probe_recall.py`'s `--mix` orchestration alongside the adversarial selector.

### 2. De-spined betweenness for NREM link proposals (Medium effort)

**What**: Remove the top-N hub nodes (those exceeding 2× median nonzero betweenness) from the graph adjacency, re-run betweenness and community detection on the remainder (`discover.py:407-460`). Bridges that survive hub removal are genuine cross-domain connectors, not just hub-adjacent nodes. The threshold is: any node ranked in the top-2 by betweenness AND exceeding 2× the median nonzero betweenness value gets pruned.

**Why**: Somnigraph's Brandes betweenness computation in `reranker.py:227-265` (called from `_load_memory_meta()`) already runs over all active memories; betweenness is feature 22 at `reranker.py:560`. Entity nodes (`category="entity"`) and high-PPR procedural memories dominate this ranking, masking which conceptual memories actually bridge topic clusters. NREM step 4c (`sleep_nrem.py:1043`) currently seeds link candidates via entity string matching — it misses non-entity bridges because those are buried under entity-hub betweenness scores. A de-spined pass would produce a ranked list of bridge-memory pairs as high-priority candidates for the pairwise LLM classifier.

**How**: Add `_despined_bridges(adj, ids)` to `scoring.py`, mirroring `despined_bridges()` in `discover.py:407-460`. The function takes the existing undirected adjacency dict (already built for PPR expansion) and returns `[(bridge_src_id, bridge_dst_id, score), ...]` sorted by combined post-spine betweenness. Call it once per NREM run after the PPR graph is built in `sleep_nrem.py:936`. Pass the top-N pairs to step 4c's pairwise classifier as additional candidates beyond entity string-match seeds. Since PPR and betweenness are already computed before step 4c, the incremental cost is one O(V²) Brandes pass over the spine-pruned subgraph.

### 3. Structural tension detection for contradiction surfacing (Low effort)

**What**: Flag memory pairs where A has both `SUPPORTED_BY` and `CHALLENGES` edges as a structural contradiction signal without an LLM call (`discover.py:465-510`). Two patterns: (1) if memory A is `SUPPORTED_BY` C and memory B `CHALLENGES` A, then (A, B) is a contested pair; (2) if B `CHALLENGES` procedural memory G and A `SERVES` G (i.e., A is an elaboration of G), then (A, B) inherit the tension — the challenge propagates to dependents. Score by sum of supporting + challenging edge counts.

**Why**: NREM step 3 (`sleep_nrem.py:973`) classifies pairwise relationships semantically via LLM and writes `hard_contradiction` / `soft_contradiction` edge types into the edges table. Over many sleep cycles, these edges accumulate but are not subsequently audited for contradictory tension — a pair (A, B) where A has `supports` edges and B has `hard_contradiction` targeting A could be in the DB for months with no synthesis pass surfacing the conflict. Structural tension detection is a cheap standing audit: one SQL join to find contested pairs, ranked by combined PPR centrality, produced without any model call. It's a complement to NREM's LLM classification, not a replacement.

**How**: Add `_structural_tensions(db)` to `scoring.py`: join the edges table to find memories with both `supports` and `contradicts`-family incoming edges, then rank by the sum of their PPR centrality scores. The join is:
```sql
SELECT e1.src, e2.src
FROM edges e1 JOIN edges e2 ON e1.dst = e2.dst
WHERE e1.edge_type = 'supports' AND e2.edge_type IN ('hard_contradiction', 'soft_contradiction')
```
Invoke during REM tier-1 housekeeping (`sleep_rem.py:1479`) and surface the top-5 contested pairs in the health diagnostics report. No new edges, no schema changes — purely a read-side scan over the existing edges table.

### 4. Shared-state workspace for sleep analytics (Low effort)

**What**: knowledge-worker's `_Workspace` class (`discover.py:126-145`) computes PageRank, betweenness, adjacency, and co-mention indexes once at the start of a `discover` run and passes the shared struct to all eight analysis functions. All analyses operate on this pre-computed state rather than independently re-walking the graph.

**Why**: Somnigraph's sleep pipeline computes betweenness in `reranker.py:_load_memory_meta()`, PPR adjacency in `scoring.py:_expand_ppr()`, and graph structure in several places across `sleep_nrem.py`. If items 1-3 above are implemented, the same adjacency graph would be built three times per sleep run. Consolidating into a single per-run workspace object — built once after Step 1 (gather), passed to Steps 4, 4b, 4c — eliminates redundant computation and makes the shared graph state explicit.

**How**: Create a `SleepWorkspace` dataclass in `sleep_nrem.py` or `scoring.py` that holds adjacency, PPR scores, betweenness, and community partition. Build it once after Step 1's memory fetch, pass it to `_despined_bridges()`, `_structural_tensions()`, and `_staleness_scores()`. Low implementation risk; the individual functions already exist, they just need a shared container.

---

## Not Useful For Us

**Substring-only retrieval** (`mygraph.py:166-174`). No semantic dimension. Not instructive — Somnigraph already has BM25 + vector + RRF + 31-feature reranker.

**Five-stage human review gate**. The design principle is correct for a provenance-first knowledge base. It is wrong for an autonomous session-time memory system. The excerpt-validation invariant would add friction without benefit for ephemeral session memories where the "source" is the conversation itself.

**Plain JSON storage**. Deliberate for diffability at small scale. Somnigraph's SQLite + sqlite-vec is load-bearing for FTS5, vector search, and PPR graph construction.

**Manual operation model**. knowledge-worker requires human promotion for every derived insight. The algorithms in `discover.py` are automatable (and worth automating in Somnigraph's sleep pipeline); the human-in-the-loop design pattern is not the thing to copy.

**OWL/Turtle export**. Useful for interop with external ontology tools. Not relevant for Somnigraph's SQLite-first architecture; the graph lives in the edges table, not in a named-graph format.

**Deep-dive workspace**. The pre-ingest synthesis workspace (`deep_dive.py`) is interesting as a pattern for high-uncertainty source processing, but Somnigraph's memories arrive from live sessions, not from markdown documents that sit on disk for deliberate review. The deep-dive mode's LLM challenge-and-synthesis step has no natural trigger point in Somnigraph's write path.

---

## Connections

- **hipporag.md**: Both systems use PageRank-family traversal for graph navigation, but HippoRAG builds its graph automatically from passage-level triple extraction while knowledge-worker requires human promotion of every edge. The de-spined betweenness idea is applicable to HippoRAG's hub problem — named entity nodes dominate betweenness in dense triple-extraction graphs identically to the way entity nodes dominate in Somnigraph's PPR graph. See hipporag.md §Retrieval for the entity-hub observation.

- **cortexgraph.md / kumiho.md**: Both perform automated graph construction, contrasting with knowledge-worker's manual promotion model. knowledge-worker's provenance discipline (literal excerpt requirement per edge, coverage metrics) is stronger than either system's citation tracking. However, neither cortexgraph nor kumiho have the staleness or tension analytics — the graph-maintenance intelligence lives in `discover.py`, not in their retrieval stacks.

- **contradiction-reconciliation.md**: knowledge-worker's `tensions()` detects contradictions structurally (edge patterns only, no LLM), while the systems in contradiction-reconciliation.md resolve them semantically. Worth Stealing item 3 bridges the two: structural scan for candidates first, LLM for resolution only on flagged pairs. This matches the pattern Somnigraph already uses in NREM (pairwise LLM classification) but without a structural pre-filter step.

- **graphrag.md**: GraphRAG's community detection (Leiden algorithm) parallels knowledge-worker's Girvan-Newman partition in `memory_audit.py`. knowledge-worker's de-spined variant directly addresses the hub-masking problem that GraphRAG sidesteps via hierarchical community summarization. The two are complementary strategies: GraphRAG's summaries absorb hub influence into a higher level; knowledge-worker removes hub influence to expose the base-level bridge structure. For Somnigraph's sleep pipeline, the removal strategy is more actionable since it produces edge proposals, not summary nodes.

- **memmachine.md**: Both systems lack corroboration metrics, but from opposite directions. MemMachine stores raw episodes and doesn't need corroboration (the source IS the memory); knowledge-worker tracks corroboration explicitly as a review signal for high-centrality single-source claims. Somnigraph falls between them — LLM-extracted summaries reduce the traceability of individual claims without the provenance enforcement knowledge-worker applies.

---

## Summary Assessment

Knowledge-worker is not an architectural peer to Somnigraph. It is a human-curated knowledge base with provenance discipline, built for a fundamentally different use pattern: deliberate long-term curation of source-backed claims, not autonomous session-time memory capture. Its retrieval is primitive by design and its consolidation is entirely manual. Comparing the two systems directly produces an unflattering picture of knowledge-worker, but the comparison is partly unfair — the systems answer different questions. Code quality is high: all analytics are deterministic, stdlib-only, tested (see `tests/test_discover.py`), and readable. The `_Workspace` shared-state pattern and the eight-function `discover.py` structure are good engineering references independent of the domain.

The value for Somnigraph is entirely in `discover.py`. Three algorithms are directly applicable to the sleep pipeline without architectural changes: staleness×centrality scoring gives probe_recall a principled importance weighting over the current flat age/utility threshold; de-spined betweenness surfaces conceptual bridge memories that entity-hub nodes currently mask in step 4c's link candidate pool; and structural tension detection provides a cheap pre-filter for contradiction surfacing in REM's health diagnostics. A fourth derivative — the shared `_Workspace` pattern — is an implementation hygiene win if items 1-3 land together. Total estimated implementation effort: one focused session. All three analytics are read-side scans over the existing graph, with no schema changes required and no new write paths.

The human review gate and provenance-first design are intellectually serious and produce a genuinely more auditable artifact. They are not adoptable for Somnigraph's use case, and the excerpt-validation invariant would be actively harmful for ephemeral session memories. Read knowledge-worker for the graph analytics in `discover.py`; treat the operational model as a design point for a different problem.

One implementation note: knowledge-worker's analytics all operate on a `_Workspace` object (`discover.py:126-145`) that computes PageRank, betweenness, and the graph projection once and passes the shared state to all eight analysis functions. Somnigraph's equivalent computation is spread across `reranker.py`'s `_load_memory_meta()` (betweenness, PPR-based centrality) and `scoring.py`'s `_expand_ppr()`. If the de-spined betweenness and structural tension additions land, consolidating the shared graph state into a single per-sleep-run computation object — mirroring knowledge-worker's `_Workspace` — would avoid redundant adjacency construction across sleep steps.
