# MSAM (Multi-Stream Adaptive Memory)

**Phase 14 source analysis** | 2026-03-06 | `jadenschwab/msam`

---

## 1. Architecture Overview

| Dimension | Detail |
|-----------|--------|
| **Repo** | [github.com/jadenschwab/msam](https://github.com/jadenschwab/msam) |
| **Stars** | 13 |
| **License** | MIT |
| **Author** | Jaden Schwab |
| **Language** | Python (~16,600 LOC across 24 modules + 7,970 LOC tests/benchmarks) |
| **Storage** | SQLite + FTS5 + FAISS (optional) |
| **Embeddings** | Pluggable: NVIDIA NIM (default), OpenAI, ONNX Runtime, sentence-transformers |
| **Embedding dim** | 1024 (configurable) |
| **Interface** | FastAPI REST API (20 endpoints) + CLI (56 commands) |
| **Config** | TOML-based, 27 tunable sections |
| **Tests** | 437 tests across 25 files |
| **Deployment** | Hetzner CAX11 (2 vCPU ARM, 4GB RAM, EUR 4/month) |
| **Scale** | 675+ active atoms in production |

MSAM is a standalone memory server (not MCP) exposing a REST API. The fundamental unit is an "atom" -- a self-contained memory with rich metadata stored in SQLite. Four cognitive streams (working, episodic, semantic, procedural) partition atoms by retrieval strategy. ACT-R-inspired activation scoring governs retrieval ranking. The system includes knowledge graph triples, sleep-inspired consolidation, intentional forgetting, contradiction detection, predictive prefetch, emotional drift tracking, sycophancy detection, and a Grafana observability layer.

The architecture is notably broad -- 24 modules covering a wider feature surface than any other single-author memory system surveyed. It reads as an exhaustive exploration of what a cognitively-informed memory architecture *could* include, built by someone working through the design space methodically.

---

## 2. Memory Type Implementation

### Schema

The atom schema is one of the richest in the survey:

```sql
atoms (
    id TEXT PRIMARY KEY,
    schema_version INTEGER DEFAULT 1,
    profile TEXT,    -- 'lightweight' (~50tok), 'standard' (~150tok), 'full' (~300tok)
    stream TEXT,     -- 'working', 'episodic', 'semantic', 'procedural'
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count INTEGER DEFAULT 0,
    stability REAL DEFAULT 1.0,        -- spaced repetition resistance
    retrievability REAL DEFAULT 1.0,   -- current recall probability
    arousal REAL DEFAULT 0.5,          -- 0.0-1.0, immutable at encoding
    valence REAL DEFAULT 0.0,          -- -1.0 to 1.0, immutable at encoding
    topics TEXT DEFAULT '[]',          -- JSON array
    encoding_confidence REAL DEFAULT 0.7,
    provisional INTEGER DEFAULT 0,
    source_type TEXT DEFAULT 'conversation',  -- conversation|inference|correction|external
    state TEXT,      -- 'active', 'fading', 'dormant', 'tombstone'
    embedding BLOB,
    metadata TEXT DEFAULT '{}',
    agent_id TEXT DEFAULT 'default',
    embedding_provider TEXT,
    is_pinned INTEGER DEFAULT 0,
    session_id TEXT,
    working_expires_at REAL
)
```

Supporting tables: `atom_topics` (junction), `access_log` (retrieval history), `corrections`, `triples` (knowledge graph), `co_retrieval` (Hebbian-like co-access), `atom_relations` (elaborates/supports/contextualizes/consolidated_into), `negative_knowledge`, `provenance`.

### Memory Lifecycle

Four states with no deletion:

```
active --> fading --> dormant --> tombstone
    \        \
     '-> active  '-> active (reactivation)
```

- **active -> fading**: retrievability < 0.3; profile compacted (full -> standard -> lightweight)
- **fading -> dormant**: retrievability < 0.1; excluded from default retrieval but searchable
- **dormant -> tombstone**: manual only; content preserved, never retrieved
- **any -> active**: explicit access resets stability upward

**Retrievability formula**: `R = exp(-age_hours / (stability * 168))` where 168 = hours/week. Stability increases with access (capped at 10.0, boost factor 1.1 per access).

### Token Budget Enforcement

Self-imposed ceiling of 40,000 tokens (20% of 200K context window). Two automatic behaviors:
- At 85% budget: auto-compact new atoms to lightweight profile
- At 95% budget: refuse to store entirely (forces decay cycle)

### Working Memory

Session-scoped atoms with TTL (default 120 minutes). When TTL expires:
- access_count > 3: promote to episodic (proved useful)
- otherwise: tombstone

### Atom Profiles

Three storage density tiers:
- **Lightweight**: ~50 tokens (simple facts, preferences)
- **Standard**: ~150 tokens (decisions, events)
- **Full**: ~300 tokens (complex analyses, emotional recordings)

Profile compaction happens during decay transitions: full -> standard -> lightweight, with content truncation to target character limits (90 chars lightweight, 240 chars standard).

---

## 3. Retrieval Mechanism

MSAM implements a layered retrieval pipeline with three main pathways:

### Pathway 1: Core Hybrid Retrieval (`hybrid_retrieve()`)

Combines three signals:
1. **Vector similarity** (FAISS fast path when available, else brute-force batch cosine)
2. **FTS5 BM25 keyword search** (with Python TF-IDF fallback)
3. **ACT-R activation scoring** (access count, recency, stability, annotations)

Combination: semantic results at full activation score + keyword results at 0.3 weight. Multi-signal bonus when both channels find the same atom.

### Pathway 2: Retrieval v2 (beam search pipeline)

Nine independently toggleable enhancements for scale:
1. Triple-augmented retrieval (KG triples bridge vocabulary gaps)
2. Query expansion via triple relationships
3. Temporal query detection and recency filtering
4. Atom quality scoring (length, vocabulary, entity density, structure)
5. Negative example tracking (implicit feedback)
6. Cross-encoder re-ranking via LLM-as-judge (NVIDIA NIM)
7. Embedding model hot-swap support
8. Pattern-based query rewriting
9. Beam search: three parallel retrieval paths (original, rewritten, expanded), deduplicated, multi-beam bonus

### Pathway 3: Triple-Augmented Hybrid (`hybrid_retrieve_with_triples()`)

Query classification distinguishes factual queries ("what/who/when") from contextual ("why/how does"), adjusting the triple-to-atom ratio. Operates within a token budget, blending KG triples and atom content.

### Spreading Activation

After core retrieval, spreading activation expands results via:
1. **Co-retrieval neighbors**: atoms frequently co-retrieved (from `co_retrieval` table), boost proportional to source activation * decay factor (0.3) * normalized co-count
2. **Triple-linked atoms**: atoms connected via `atom_relations` (elaborates/supports/contextualizes), weaker boost (0.5x of co-retrieval)

### Confidence Tiering

Every retrieval result gets classified into tiers based on **two signals** -- max cosine similarity (primary) and combined score (secondary):

| Tier | Similarity threshold | Score threshold |
|------|---------------------|-----------------|
| high | >= 0.45 | >= 40.0 (with semantic signal) |
| medium | >= 0.30 | >= 10.0 (with semantic signal) |
| low | >= 0.15 | -- |
| none | below all | -- |

Temporal queries get stricter handling: if query contains markers like "right now" / "today" / "this session" and no recent (24h) high-similarity atoms exist, results are capped at "low" tier.

---

## 4. Standout Feature: ACT-R Activation with Confidence-Gated Retrieval

The single most technically interesting contribution in MSAM is its implementation of ACT-R activation theory adapted to AI memory, combined with a confidence tiering system that enables the agent to explicitly admit uncertainty.

### Cognitive Science Lineage

ACT-R (Adaptive Control of Thought -- Rational) is John Anderson's cognitive architecture, formalized across decades of work (Anderson 2007). Its base-level activation equation models how memory accessibility changes with practice and time:

```
B_i = ln(sum(t_j^(-d))) + S + epsilon
```

Where:
- `B_i` is base-level activation of chunk i
- `t_j` is time since the j-th access
- `d` is decay parameter (typically ~0.5)
- `S` is spreading activation from associated chunks
- `epsilon` is noise

The key insight: memories that are accessed frequently AND recently have highest activation, but the relationship is logarithmic (diminishing returns on repetition) and power-law decaying (recent access matters far more than old).

### MSAM's Implementation

MSAM's `compute_activation()` function translates this to code with several practical adaptations:

```python
def compute_activation(atom, query_similarity=0.0, mode="task"):
    # Base activation (ACT-R) -- CAPPED to prevent frequency dominance
    access_count = atom.get("access_count", 0)
    age_hours = max((now - created).total_seconds() / 3600, 0.01)

    base = min(log(access_count + 1), 3.0) - 0.5 * log(age_hours + 1)

    # Similarity component -- sigmoid replaces linear
    if query_similarity < 0.2:
        similarity = 0.0  # noise floor
    else:
        similarity = sigmoid_boost(query_similarity) * 6.0

    # Annotation boost (mode-dependent)
    if mode == "companion":
        annotation_boost = arousal * 0.8 + abs(valence) * 0.4
    else:  # task mode
        annotation_boost = confidence * 0.3 - arousal * 0.1

    # Stability factor
    retrievability = exp(-age_hours / (stability * 168))
    stability_factor = retrievability * 0.3

    # Outcome attribution (Felt Consequence)
    if outcome_count >= 3:
        normalized = clamp(outcome_score, -5, 5) / max(outcome_count, 1)
        outcome_bonus = 0.15 * normalized

    return base + similarity + annotation_boost + stability_factor + outcome_bonus
```

**Key adaptations from ACT-R theory:**

1. **Base activation cap at 3.0**: Pure ACT-R would let `ln(access_count + 1)` grow unboundedly. MSAM caps it to prevent "hot atom" dominance -- an atom accessed 1000 times shouldn't permanently dominate retrieval. This is a pragmatic departure from the theory: in human cognition, very frequent memories do dominate (your name, your address), but in an AI system this creates a runaway feedback loop since retrieval itself increments access count.

2. **Simplified decay to single-timestamp**: Rather than tracking every access timestamp (sum of `t_j^(-d)` across all accesses), MSAM approximates with `log(access_count + 1) - 0.5 * log(age_hours + 1)`. This is computationally cheaper but loses the recency-of-recent-access signal -- a memory accessed 10 times last week and 0 times this week gets the same score as one accessed 5 times each week. The original ACT-R equation handles this correctly.

3. **Sigmoid similarity curve**: Instead of linear similarity weighting, MSAM applies a sigmoid with configurable midpoint (default 0.35) and steepness (15.0). This creates a hard noise floor -- similarities below 0.2 contribute nothing, and the sigmoid amplifies the 0.35-0.50 range where genuine semantic matches cluster. The formula:

   ```
   sigmoid_boost(x) = 1 / (1 + exp(-15 * (x - 0.35)))
   ```

   At x=0.2, output ~0.01. At x=0.35, output ~0.50. At x=0.50, output ~0.99. This is multiplied by spread_weight 6.0, so maximum similarity contribution is ~6.0 activation units.

4. **Mode-dependent annotation**: The emotion-at-encoding annotations (arousal, valence) participate differently based on retrieval mode. "Companion" mode boosts emotionally salient memories; "task" mode boosts confident memories and slightly penalizes high-arousal ones. This is the "inverted stack" philosophy -- emotion enriches associative retrieval but is deliberately suppressed during precision-seeking fact retrieval.

5. **Stability-gated retrievability**: A secondary decay signal using spaced-repetition-inspired stability: `R = exp(-age_hours / (stability * 168))`. Each access boosts stability by 1.1x (capped at 10.0). This creates a separate channel from base activation -- stability models how "consolidated" a memory is, while base activation models raw familiarity.

6. **Outcome attribution (Felt Consequence)**: After 3+ outcome events, an atom's historical contribution quality influences its activation. This is a feedback loop: atoms that helped produce good agent responses get boosted; atoms linked to poor outcomes get dampened. The signal decays exponentially (factor 0.95), so recent outcomes dominate.

### Confidence Gating

The activation scoring feeds into a **confidence-gated retrieval** system. After scoring and ranking, `hybrid_retrieve()` classifies the entire result set into one of four tiers:

```python
if max_sim >= 0.45 or (has_semantic_signal and top_score >= 40.0):
    confidence_tier = "high"
elif max_sim >= 0.30 or (has_semantic_signal and top_score >= 10.0):
    confidence_tier = "medium"
elif max_sim >= 0.15:
    confidence_tier = "low"
else:
    confidence_tier = "none"
```

This is attached to the first result as `_retrieval_confidence_tier`, and each individual result also gets its own `_confidence_tier`. The tiers are designed to let the consuming agent make explicit decisions:

- **high**: trust these results, use them directly
- **medium**: usable but verify
- **low**: supplement with external search
- **none**: no relevant knowledge -- admit the gap

The system also logs "retrieval misses" when top score falls below a quality threshold (default 2.0), feeding into the metrics/observability layer for tuning.

### Metamemory

The `metamemory_query()` function extends confidence gating to a **pre-retrieval** check: "What do I know about X, and how confident am I?" It returns coverage (high/medium/low/none), confidence (weighted by encoding_confidence * recency * evidence), atom/triple counts, stream distribution, and a recommendation (retrieve/search/ask). This enables the agent to decide whether to query its own memory, search externally, or ask the user -- before committing tokens to a full retrieval.

### Negative Knowledge

The `negative_knowledge` table records queries that returned nothing useful (empty, low_confidence, or contradictory), with configurable TTL (default 1 week). `check_negative()` is called *before* external search to avoid repeating known-futile queries. This is a form of "knowing what you don't know" that extends the confidence gating to cached uncertainty.

### What Makes This Stand Out

Most memory systems in the survey retrieve-and-rank without surfacing how *confident* the system is in its results. MSAM's four-tier classification is a direct mapping of ACT-R's activation levels to actionable uncertainty quantification. The metamemory + negative knowledge combination creates a complete pre-retrieval decision framework: check if you know you don't know (negative), check if you know enough (metamemory), then either retrieve, search, or ask.

The theoretical grounding is genuine -- the SPEC.md cites Anderson 2007, Ebbinghaus, Tulving, and emotional encoding papers (Richter-Levin 2003, Damasio 1994, McGaugh 2004, Sharot 2007). The implementation follows the theory where practical, and departs with documented rationale (e.g., the base activation cap, the inverted emotional stack).

---

## 5. Other Notable Features

### 5.1 Knowledge Graph Triples with Temporal World Model

The `triples.py` module implements a full knowledge graph:
- **LLM-powered extraction**: NVIDIA NIM Mistral endpoint parses atoms into normalized (subject, predicate, object) triples
- **Temporal metadata**: `valid_from` / `valid_until` timestamps on each triple. When a new fact conflicts with an existing one (same subject + predicate), the old triple auto-closes (`valid_until = now`)
- **Three query modes**: current state (valid_until IS NULL), point-in-time (specific timestamp), full history
- **Graph traversal**: `graph_traverse()` for multi-hop exploration, `graph_path()` for shortest path between entities
- **Hybrid retrieval**: `retrieve_triples()` blends semantic + keyword search over triples, and `hybrid_retrieve_with_triples()` combines triples + atoms within a token budget

This is one of the more complete KG implementations in the survey. The temporal validity tracking is particularly well-thought-out -- most systems treat facts as permanently true or require manual invalidation.

### 5.2 Intentional Forgetting Engine

Four signal detectors identify counterproductive memories:
1. **Over-retrieved**: high access count but low contribution rate (retrieved often, rarely helpful)
2. **Superseded**: replaced by newer atoms via relation mappings
3. **Contradicted**: conflicts with higher-confidence atoms
4. **Below confidence floor**: decayed confidence without recent usage

Multiple signals aggregate for stronger conviction. In auto mode: 2+ signals or contradiction -> tombstone; 1 signal -> dormant. This goes beyond passive decay -- it actively identifies and retires memories that are *hurting* retrieval quality.

### 5.3 Shannon Compression (Sub-Atom Extraction)

The `subatom.py` module implements sentence-level extraction from retrieved atoms:
1. Split atoms into sentences
2. Score sentences by: embedding similarity, specificity penalty, information density, keyword overlap, uniqueness ratio, atom rank boost
3. Deduplicate near-identical sentences (cosine threshold 0.85)
4. Optional LLM synthesis to compress further

Target: ~50 tokens per query from the extracted sentences. Claimed compression: 99.3% on startup context (7,327 tokens -> 51 tokens).

### 5.4 Entity-Role Awareness

The `entity_roles.py` module solves a real problem: embeddings can't distinguish WHO an atom is ABOUT from WHO is MENTIONED in it. "Things Agent Knows: professional performer" is ABOUT the user, not the agent. The module:
- Tags each atom with `about_entity` (user/agent/system/relationship) using regex pattern matching on first line + content signals
- Classifies query intent (who the query is asking about)
- Applies scoring adjustments: match = boost (1.0 + 0.8*confidence), mismatch = penalty (1.0 - 0.5*confidence)

### 5.5 Predictive Prefetch

Three prediction strategies combined with weighted scoring:
1. **Temporal patterns**: time-of-day access frequency (morning/afternoon/evening/night buckets)
2. **Co-retrieval patterns**: atoms frequently accessed together
3. **Topic momentum**: topic overlap scoring against recent topics

The prediction engine learns from session access patterns and pre-fetches likely-needed atoms.

---

## 6. Gap Ratings

### Layered Memory: 90%

Four cognitive streams (working/episodic/semantic/procedural) with distinct retrieval modes. Three profile tiers for storage density. Working memory with TTL and promotion. Only gap: streams don't have truly different retrieval algorithms (all go through the same activation scorer with mode-dependent weights).

### Multi-Angle Retrieval: 95%

One of the most complete retrieval stacks surveyed: vector cosine + FTS5 BM25 + ACT-R activation + KG triple augmentation + beam search + query expansion + entity-role scoring + spreading activation via co-retrieval + cross-encoder re-ranking. Multiple pathways with different blending strategies.

### Contradiction Detection: 80%

Dedicated `contradictions.py` with four heuristic detectors (negation, temporal supersession, value conflict, antonym pairs). `check_before_store()` validates new content against existing atoms pre-insertion. Temporal world model auto-closes superseded facts. Missing: embedding-based contradiction detection (relies on heuristics), no automated resolution beyond flagging.

### Relationship Edges: 70%

`atom_relations` table with types (elaborates/supports/contextualizes/consolidated_into), `co_retrieval` table for implicit associations. Spreading activation uses both. Missing: no explicit contradiction/revision/derivation flags on edges, no rich linking context, edges created only by consolidation engine (not session-time linking).

### Sleep Process: 55%

`consolidation.py` implements sleep-inspired clustering: cluster similar atoms, LLM-synthesize abstractions, reduce source stability. But it's a single pass (no NREM/REM distinction), no gap analysis or question generation, no edge creation beyond consolidation relations, and the LLM synthesis is a single Mistral call with graceful fallback to longest-content selection.

### Reference Index: 40%

Topics are JSON arrays on atoms, with an `atom_topics` junction table. Triples provide structured facts. But no formal taxonomy, no reference notes or curated indices, no hierarchical organization beyond the four streams.

### Temporal Trajectories: 75%

Strong temporal infrastructure: `valid_from`/`valid_until` on triples, temporal world model with point-in-time queries, emotional drift detection comparing early vs recent annotation windows, temporal query detection in retrieval. Missing: no cross-session continuity scoring beyond Jaccard overlap, no trajectory visualization, no temporal prediction beyond time-of-day patterns.

### Confidence/UQ: 85%

Best in survey for explicit uncertainty quantification. Four-tier confidence gating on every retrieval. Metamemory for pre-retrieval coverage assessment. Negative knowledge table caches known gaps. `encoding_confidence` on atoms, evidence-based confidence gradient on triples. Provisional flag for uncalibrated atoms. Missing: confidence doesn't compound through feedback the way our system's does -- it's more classification-based than gradient-based.

---

## 7. Comparison with claude-memory

### Stronger (MSAM vs claude-memory)

**Confidence-gated retrieval with metamemory**: MSAM's four-tier confidence classification on every retrieval result, combined with the `metamemory_query()` pre-retrieval check and `negative_knowledge` cached failures, creates a complete uncertainty-aware retrieval framework. Our system has a confidence gradient (0.1-0.95) that compounds through feedback, but we don't surface explicit "I don't know" signals or pre-retrieval coverage assessments. MSAM's agent can decide retrieve/search/ask *before* committing to a full query.

**ACT-R activation scoring with theoretical grounding**: The activation equation integrates access frequency, recency, semantic similarity, emotional annotations, stability, and outcome attribution into a single principled score. Our system uses post-RRF additive scoring with 9 coefficients (Bayesian-optimized), which is empirically tuned but less theoretically grounded. MSAM's equation has clearer cognitive science lineage.

**Knowledge graph triples with temporal validity**: Full triple extraction, temporal `valid_from`/`valid_until`, auto-closing superseded facts, graph traversal, point-in-time queries. Our system has memory_edges with linking_context but no structured triple extraction or temporal fact management.

**Broader feature coverage**: 24 modules covering entity-role awareness, predictive prefetch, sycophancy detection, cross-provider embedding calibration, Shannon compression, intentional forgetting with 4 signal types, multi-agent isolation. Many of these are features we haven't built (and some we shouldn't -- see section 9).

**Observability**: 13 metric tables, 25 Grafana panels, canary monitoring for identity drift, cross-session continuity scoring. Our system has quality-floor score logging and recall_meta events but nothing approaching this level of instrumentation.

### Weaker (MSAM vs claude-memory)

**Retrieval scoring calibration**: Our system has Bayesian-optimized scoring (Optuna TPE, 500 trials, 9 jointly optimized coefficients). MSAM uses hand-tuned weights throughout -- the sigmoid midpoint, spread weight, annotation boost coefficients, confidence tier thresholds are all manually chosen defaults. No evidence of systematic empirical tuning.

**Sleep/consolidation pipeline**: Our NREM (cluster, merge, create edges) + REM (gap analysis, question generation) pipeline is substantially more sophisticated than MSAM's single-pass consolidation (cluster similar atoms, LLM summarize, reduce stability). We process batches with careful edge creation, novelty-scored adjacency expansion, and taxonomy-driven clustering.

**Feedback loop maturity**: Our recall_feedback tool (continuous 0-1 utility + durability) feeds back into scoring via FEEDBACK_COEFF (dominant post-RRF component at 0.35), Hebbian PMI co-retrieval boost, and confidence gradient compounding. MSAM has contribution tracking (binary: contributed or not) and outcome-attributed scoring, but the feedback signal is less granular and doesn't compound through graph edges.

**Edge sophistication**: Our memory_edges have flags (contradiction/revision/derivation), linking_context with embeddings, novelty-scored adjacency expansion, and session-time linking via `link()` tool. MSAM's edges are simpler: `atom_relations` with 4 types (elaborates/supports/contextualizes/consolidated_into), created only by the consolidation engine.

**MCP integration**: Our system runs as an MCP server (stdio), directly callable from Claude Code with typed tools. MSAM is a standalone REST API -- the agent integration requires HTTP calls rather than native tool invocation. This creates latency overhead and loses the typed-parameter benefits of MCP.

**Decay model richness**: Our per-category exponential decay with configurable half-lives, floors, shadow-load quadratic penalty, and retention immunity via pinned flag is more nuanced than MSAM's uniform retrievability decay with stability factor. MSAM does have profile compaction during decay, which we don't.

### Shared Strengths

- **SQLite core**: Both chose SQLite for single-user scale, with FTS5 for keyword search. Both acknowledge PostgreSQL as the scaling path.
- **Hybrid retrieval**: Both combine vector similarity + keyword matching, rejecting pure-semantic approaches.
- **No-deletion philosophy**: Both preserve all data (our "dormant" states, MSAM's tombstone as deepest state).
- **Deduplication**: Both content-hash at write time.
- **Spreading activation / adjacency expansion**: Both boost retrieval results via graph neighbors (co-retrieval, linked atoms).
- **Lifecycle hooks**: Both have event-driven callbacks for state transitions.
- **Topic/theme management**: Both use topic tags for filtering and retrieval.
- **Pinned memories**: Both support retention immunity for important memories.

---

## 8. Insights Worth Stealing

### 8.1 Metamemory Query

**What**: A `metamemory_query(topic)` function that returns coverage assessment *without* actually retrieving atoms. Returns confidence, atom/triple counts, stream distribution, and recommendation (retrieve/search/ask).

**Why it matters**: We currently always retrieve before deciding if we have enough knowledge. A pre-retrieval coverage check could save tokens and enable better agent decision-making ("I know nothing about X, let me ask the user rather than hallucinating").

**Implementation sketch**: Count atoms/edges matching topic via FTS5 + theme match. Compute coverage from count + avg confidence. Return coverage tier + recommendation. Could be a lightweight variant of `recall()` that returns metadata only.

**Effort**: Low (1-2 hours)
**Impact**: Medium -- changes how the agent decides whether to recall vs ask, especially useful for topics with zero coverage

### 8.2 Negative Knowledge Cache

**What**: Record queries that returned empty/low-confidence/contradictory results, with TTL. Check cache before expensive external searches.

**Why it matters**: Prevents repeated failed retrieval attempts on topics we demonstrably know nothing about. Especially useful for subagent recalls that might redundantly query the same gaps.

**Implementation sketch**: Add `negative_knowledge` table (query, result_type, searched_at, expires_at). Check in `recall()` preamble. Log misses automatically when quality_floor filters everything.

**Effort**: Low (1-2 hours)
**Impact**: Low-Medium -- marginal for ~300 memories but grows with scale and multi-agent use

### 8.3 Confidence Tier on Retrieval Results

**What**: Classify each `recall()` result set as high/medium/low/none confidence based on top similarity + top score, so the consuming agent knows how much to trust the results.

**Why it matters**: We already log quality_floor_ratio metrics, but don't surface this to the agent as an actionable signal. A confidence tier on the recall response would let the agent decide to supplement with external search or admit uncertainty.

**Implementation sketch**: After scoring, compute max_sim and top_score. Classify into tiers using thresholds (calibrate from quality_floor_ratio data). Attach to recall response.

**Effort**: Low (30 minutes -- we already compute the inputs)
**Impact**: Medium -- enables uncertainty-aware agent behavior

### 8.4 Emotional Drift Detection

**What**: Compare annotations (arousal, valence, or in our case, themes/priority) on a topic across time windows to detect how associations are changing.

**Why it matters**: Could reveal interesting patterns -- topics whose priority or theme distribution shifts over time. More useful as an analytical tool than a retrieval feature.

**Effort**: Medium (needs time-windowed aggregation)
**Impact**: Low -- more interesting than practical for our current scale

### 8.5 Entity-Role Scoring

**What**: Tag memories by *who they're about* (user vs agent vs system) and boost/penalize based on match with query intent.

**Why it matters**: Our system doesn't distinguish "memory about Alexis" from "memory about project X" at a structural level. For the ~300 memory scale this isn't critical, but the concept of query intent classification (what is this question *really* asking about?) could improve recall precision.

**Effort**: Medium (regex pattern classification, scoring adjustment)
**Impact**: Low-Medium -- more impactful at higher memory counts

---

## 9. What's Not Worth It

### REST API (vs MCP)

MSAM exposes a FastAPI server with 20 endpoints, authentication, CORS, async execution. This makes sense for a language-agnostic integration target (any HTTP client can use it), but for our use case -- a single Claude Code agent consuming memory via MCP -- the overhead of an HTTP server is unnecessary. MCP's stdio transport is lower-latency, has typed tools, and integrates natively with the agent framework. The REST API is solving a different deployment problem.

### Grafana Observability Layer

13 metric tables, 25 dashboard panels, canary monitoring, systemd timers. This is impressive infrastructure engineering but overkill for our system. We get sufficient signal from quality_floor_ratio logging, recall_feedback scores, and the sleep pipeline's post-run stats. The canary monitoring (fixed identity query every 5 minutes) is clever but assumes a continuously running server, which doesn't match our session-based MCP model.

### Multi-Agent Memory Isolation

MSAM supports per-agent memory with shared layer. We're a single-user, single-agent system. The only multi-agent pattern we use is subagent recall (curated_recall via Sonnet), which already has the `internal=true` flag to prevent feedback pollution. Full agent isolation is engineering for a use case we don't have.

### Sycophancy Detection

Agreement rate tracking via sliding window. Interesting metacognitive feature but not a memory system concern -- this belongs in the agent's reasoning layer, not the storage/retrieval stack. Also difficult to implement accurately without conversation-level context that the memory server doesn't naturally have.

### Sub-Atom Sentence Extraction

Shannon compression to ~50 tokens per query. Our startup_load already delivers curated memories at controlled token budgets via the summary field. The sentence-level extraction approach assumes memories are long enough to benefit from sub-extraction, which doesn't match our curated-memory model (~300 memories with explicit summaries). More relevant for systems with large, unstructured memory dumps.

### Cross-Provider Embedding Calibration

MSAM supports hot-swapping between NVIDIA NIM, OpenAI, ONNX, and sentence-transformers, with calibration metrics (Kendall tau, overlap-at-k) to assess migration risk. We use a single provider (OpenAI text-embedding-3-small) and are unlikely to change. The migration tooling is defensive engineering for a risk we're not exposed to.

---

## 10. Key Takeaway

MSAM is the most theoretically grounded and feature-complete single-author memory system in the survey. Its ACT-R activation scoring, four cognitive streams, and confidence-gated retrieval with metamemory represent genuine cognitive science applied to AI memory rather than the ad-hoc approaches typical of the space. The breadth is remarkable -- 24 modules covering everything from Shannon compression to sycophancy detection -- though this breadth comes at the cost of depth in areas like consolidation and feedback loops where our system is significantly more sophisticated. The most transferable insight is the confidence tier framework: MSAM treats uncertainty as a first-class retrieval signal rather than an afterthought, enabling the agent to explicitly decide whether to trust its memory, supplement with external search, or admit ignorance. For our system, the metamemory query and confidence tier annotations would be low-effort, medium-impact additions that change how the agent relates to its own knowledge gaps.
