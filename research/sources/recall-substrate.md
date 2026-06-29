# Recall — push memory substrate with graph-computed confidence

*Generated 2026-06-28 by Sonnet agent reading local clone*

**Repo**: https://github.com/H-XX-D/recall-memory-substrate  
**Stats**: 13 stars, 2 forks, Apache-2.0, TypeScript, v0.1.0, last push 2026-06-27, 229 passing tests, 94 e2e checks  
**Clone path**: scratchpad `repos/recall-memory-substrate` (depth 1, HEAD `6294d2e`)

---

## Architecture

### Storage & Schema

Single SQLite file (`.recall/recall.sqlite3`) via Node.js built-in — no sqlite-vec, no server, no account. Semantic search is a separate optional layer (hash-based stub by default; real embeddings opt-in). Every cell gets a UUID, a URI address, a rollback entry, and a provenance stamp.

**Node schema** (`src/core/types.ts`):

```typescript
{ id, cellAddress, kind, title, body, summary, scope,
  tags, data: { intent, evidence, confidence, policy },
  provenance: { created_at, origin, produced_by, verification },
  status: "active" | "superseded" | "archived" }
```

37 `kind` values: observation, belief, decision, risk, task, hypothesis, preference, etc. Tags are structured: `category[]`, `type[]`, `subject[]`, `topics[]`, `entities[]`, `rings[]`, `lifecycle[]`.

Cell address: `recall://cell/<project>/<category>/<type>/<subject>/<idea>/<timestamp>/<id>` — fully dereferenceable across graph, compiler, and rollback journal.

### Memory Types

No episodic/semantic/procedural taxonomy. Typing is via `intent.kind` at write time. `kind=decision` is the closest Somnigraph analog to `category=procedural`; `kind=belief` to `category=semantic`; `kind=risk` to `category=meta`. Categories are a free-form tag, not a structural constraint.

### Write Path — Push Loop and Supersession

The write contract (`recall.write.v1`) is a full JSON proposal: actor, intent, content, scope, tags, evidence (`source_refs`, `depends_on`, `supports`, `contradicts`, `concerns`), confidence, provenance, policy. Admission pipeline (`src/core/admission.ts`):

1. Schema validation via `validateWriteProposal()`
2. Firewall check (`src/core/firewall.ts`): rejects credential shapes (JWTs, cloud keys, private-key blocks)
3. **Confidence attenuation** (`admission.ts:181-203`): if `evidence_count == 0 && verification == "unverified" && confidence.value > 0.7` → floor to 0.7, warn
4. Near-duplicate warning: Jaccard title ≥ 0.6 OR content cosine ≥ 0.9 → warn (never reject)
5. **Trust-edge guard** (`admission.ts:296`): drops any `supports/contradicts/concerns` reference targeting a non-existent node, warning rather than silently dangling
6. Insert node + relations + rollback entries atomically

**The push loop**: A `UserPromptSubmit` hook injects a mini-index of relevant cells into context before each prompt. The agent reads this, does the work, and writes a correction via MCP. If the new cell's proposal includes `evidence.contradicts: [old-cell-id]`, a `contradicts` relation is materialized at WRITE TIME. The old cell's effective confidence then collapses automatically at the next READ — no LLM, no query.

Clarification the README soft-pedals: there is **no automatic semantic contradiction detection in the admission layer**. The agent must explicitly declare `contradicts: [id]`. SENTINEL L2 requires an LLM/Checker to detect entailment-level contradictions; the admission firewall doesn't do it. "Push" means the substrate processes the declared contradiction automatically, not that it discovers contradictions on its own.

### Retrieval

Two channels:

1. **FTS5 BM25** (`src/core/retrieval.ts`): Porter stemming, OR'd quoted phrases, ~60-word stop list, configurable per-kind multipliers (e.g., `artifact` kind down-weighted 0.15× to prevent symbol-stub dominance)
2. **Semantic search** (`src/core/semantic.ts`): opt-in, configurable embeddings

**Ranking formula** (`retrieval.ts:fuseCandidates`):
```
score = lexical_normalized
      + 0.25 × log1p(degree) / log1p(10)   // graph degree prior
      + 0.15 × effective_confidence          // trust signal
      + 0.10 × exp(-age_days / 30)           // recency decay
```

Lexical component dominates (owns [0,1]); the three priors sum to at most 0.5, so they reorder near-ties but cannot outvote a clearly better BM25 match.

No RRF fusion. No learned reranker. No feedback loop. Graph degree is the only graph signal in retrieval — PPR, theme channel, and Hebbian PMI have no equivalent.

**Context compiler**: word-budget packet listing ranked cells with their effective confidence and any incoming challengers. Returns cell ids for drill-down rather than flooding context.

### Consolidation

**Daemon** (`src/core/daemon.ts`): rule-based background process. Runs stale-memory, contradiction, derivation, and eval passes outside the LLM. Writes back through the same admission gate.

No sleep phases. No LLM-mediated consolidation. The daemon is closer to Somnigraph's maintenance scripts than to NREM/REM.

**Programs** (`src/core/programs.ts`): standing operations on hyperedge bundles — `watch` (trips when effective confidence moves past delta), `drift` (watch + attribution), `quorum` (k-of-m sign-off as a graph object), `trend` (finite-difference calculus over run history). These are the "push" actuators that emit witnesses when conditions change.

### Lifecycle — Computed Confidence

The central architectural claim: effective confidence is NOT stored. It is computed from the live graph on every read (`src/core/evidence.ts`).

```typescript
// evidence.ts:82-136 (condensed)
export function effectiveConfidence(store, node, calibration?): ConfidenceBreakdown {
  const stated = statedConfidence(node);                   // from data.confidence.value
  const calibrationFactor = calibration?.get(producer) || 1;

  // One-hop traversal of incoming trust edges
  for (const relation of store.listRelations(node.id, "in", 100)) {
    if (relation.kind === "supports")   supportMass  += statedConfidence(source);
    if (relation.kind === "contradicts") challengeMass += statedConfidence(source);
    if (relation.kind === "concerns")    challengeMass += statedConfidence(source) * 0.5;
  }
  // Hyperedge checker/test verdicts carry VERIFIED_EDGE_WEIGHT = 1.25
  for (const edge of store.hyperedgesForNode?.(node.id, 100) ?? []) {
    if (edge.kind.endsWith("-supports"))    supportMass  += 1.25;
    if (edge.kind.endsWith("-contradicts")) challengeMass += 1.25;
  }

  const support   = 0.15 * Math.tanh(supportMass);   // SUPPORT_CEILING = 0.15
  const challenge = 0.60 * Math.tanh(challengeMass);  // CHALLENGE_CEILING = 0.60
  const effective = clamp01(stated * calibrationFactor + support - challenge);
}
```

**Calibration** (`src/core/calibration.ts:66-78`):
```typescript
const overconfidence = row.contradictedRate * (row.meanConfidenceContradicted ?? 0);
factors.set(row.actor, Math.max(0.5, 1 - overconfidence));
// Floor = 0.5; min 3 cells before calibration applies
```

Calibration keys on **overconfidence** (contradicted-while-confident), NOT raw Brier. This is principled: Brier also penalizes humble actors who state 0.4 on claims that survive. The overconfidence metric doesn't. But unlike Brier, overconfidence is NOT a proper scoring rule — an actor could game it by always stating 0.5. The `calibration.ts` code correctly computes Brier for the audit report; the actual DISCOUNT uses the overconfidence metric. The blog post conflates these two things, calling Brier "the scorecard" while the actual penalty is different.

**Three-axis question** (evidence vs currency vs salience): these are blended additively in the final ranking score, not fully orthogonal. Evidence = effective confidence (weight 0.15); currency = recency decay (weight 0.10); salience = graph degree (weight 0.25). Effective confidence is architecturally separate from ranking, but they share the same additive formula.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Push memory: corrections supersede stale facts automatically | `admission.ts`, `evidence.ts`, demo screencasts | **Validated at mechanism level.** Edge materialized at write time; demotion automatic at read time, O(edges-on-node), no LLM. Requires agent to explicitly declare `contradicts` — no auto-detection. |
| Confidence computed not stored | `evidence.ts` entire `effectiveConfidence()` function | **Validated.** No stored effective confidence field. Recomputed on every read from live graph. |
| SUPPORT/CHALLENGE asymmetry is correct (0.15/0.60) | Design choice, blog post | **Principled, unvalidated.** Asymmetry direction is correct (agreement cheap, contradiction informative). Specific constants are intuition, not ablation. No tuning study. |
| Calibration uses Brier as proper scoring rule | `calibration.ts:49-55` | **Partially accurate.** Brier IS computed for the audit. The DISCOUNT uses overconfidence (contradicted_rate × mean_confidence_when_contradicted), which is NOT a proper scoring rule. A writer stating 0.5 always has overconfidence=0 and is never discounted, regardless of whether 0.5 is a good calibration. |
| SENTINEL L1: 100% recall/precision, 0-tick latency (24 streams) | `docs/10_SENTINEL_BENCHMARK.md`, `scripts/sentinel-bench.mjs` | **True but trivial.** Deterministic value-flip detector ("Chicago"→"Denver") on 24 synthetic streams. Unit test masquerading as benchmark. |
| SENTINEL L2: 100% surfacing recall | L2 section, KB stand-in | **Honest about scope.** Surfacing floor is model-free and works. Detection (the hard part) deferred to KB stand-in/LLM — Recall's unique contribution is surfacing, not detecting. |
| SENTINEL L3: 100% cycle detection (24 triples) | `addDagOverlay` rejection | **Real and unique.** Cycle detection at write time has no equivalent in pull systems. 24 triples is a unit test, not a real-scale evaluation. |
| Pull systems score 0 on SENTINEL "by construction" | Structural argument | **Accurate and technically honest.** Pull systems lack standing programs. But SENTINEL tests mechanism presence, not mechanism quality. A system designed to always score 100% on its own benchmark is not equivalent to a system with independently measured quality. |

---

## Relevance to Somnigraph

### What Recall does that Somnigraph doesn't

**Ambient pre-prompt hook.** `UserPromptSubmit` injects relevant cells before each prompt, flagging superseded/stale ones with escalating severity. Somnigraph has no equivalent — `recall()` must be called explicitly. MEMORY.md says "use proactively" but that's a norm, not a mechanism. This gap is real and behavioral.

**Evidence declared at write time.** Recall's `evidence.contradicts/supports/depends_on` fields force the agent to declare relationships at write time, creating trust edges immediately. Somnigraph's `link()` creates edges at session time (optional), with graph construction deferred to NREM sleep. Recall's graph is denser immediately after write.

**Admission firewall with confidence attenuation.** Somnigraph's `remember()` accepts any priority (1-10) and confidence with no quality gate. Recall clips unsupported high-confidence claims (>0.7) to 0.7 and drops dangling trust references. This prevents pollution at write time.

**Rollback journal.** Every write generates a rollback entry. `recall rollback apply` undoes a write atomically. Somnigraph has no equivalent — a wrong `remember()` requires manual `forget()` + re-entry.

**Computed confidence** (the live design question). Somnigraph STORES confidence as part of the LightGBM reranker (features: `fb_last`, `fb_mean`, `fb_count`, `fb_time_weighted`, `hebbian_pmi`). Recall COMPUTES it from the live graph at read time. Somnigraph's confidence requires user feedback history to be meaningful. Recall's works from the first write, requires no feedback, and is immediately interpretable (you can trace every term). The tradeoff: Somnigraph's confidence is empirically validated (NDCG=0.8954 CV'd against 1885 queries); Recall's is principled but unvalidated.

**N-ary hyperedges and programs.** `watch`/`quorum`/`trend` programs are standing actuators that fire when effective confidence changes. Somnigraph has no equivalent standing computation — edges are built during sleep, not watched live.

### What Somnigraph does better

**Retrieval quality.** Somnigraph: BM25 + vector + PPR + theme via RRF, 31-feature LightGBM reranker (NDCG=0.8954, 1885 queries, CV'd), explicit feedback loop. Recall: BM25 + optional semantic, hand-tuned additive weights (0.25/0.15/0.10), no learning, no feedback. The retrieval gap is wide and measured.

**Validated confidence.** Somnigraph's scoring was tuned through wm1→wm38 (3673 trials, 9 phases) then replaced by learned reranker cross-validated with NDCG against ground truth. Recall's confidence constants (0.15/0.60/0.50 floor) are intuition with no ablation.

**LLM-mediated consolidation.** Somnigraph's three-phase sleep (NREM pairwise classification, edge discovery, merge/archive; REM gap analysis and taxonomy) discovers relationships the agent didn't declare. Recall's daemon is rule-based. Sleep's NREM step does what Recall expects the agent to do manually (declare `contradicts`/`supports`).

**Benchmark evidence.** Somnigraph: 85.1% LoCoMo (Opus judge), 95.4% R@10. Recall: no LoCoMo/LME numbers. SENTINEL is a self-referential internal benchmark.

**Decay integrated into scoring.** Somnigraph's `decay_rate` per category shapes retrieval through the reranker features. Recall's recency decay (30-day half-life) and `policy.expires_at` are correct in design but the weight (0.10) means recency can barely move rank against a strong BM25 match.

---

## Worth Stealing (ranked)

### 1. Admission firewall with confidence attenuation (Low effort, High impact)

**What**: In `impl_remember()`, if `evidence_count == 0 && verification == "unverified" && priority > 7` (mapping priority 1-10 to ~0.7 confidence threshold), emit a warning and cap priority at 7. Separately, check for near-duplicate titles (trigram Jaccard against recent 50 memories) and warn.
**Why**: Somnigraph accepts any priority at write time. High-priority memories with no evidence compete at full strength. The attenuation rule is ~10 lines, has no false positives by construction (only caps unverified high claims), and teaches the agent to write defensively.
**How**: `src/memory/tools.py` `impl_remember()` — post-validation check. Needs a lightweight near-dup scan (trigram set of title words, Jaccard against `db.py` recent node list). ~20 lines. No schema changes.

### 2. Write-time evidence declaration extending `remember()` (Medium effort, High impact)

**What**: Add optional `contradicts: [id]`, `supports: [id]`, `depends_on: [id]` fields to `remember()`. At write time, immediately create edges. Drop dangling references with a warning (matching `admission.ts:296`).
**Why**: Somnigraph's graph is built during NREM sleep — new memories have no edges until the next consolidation cycle. An agent that knows "this corrects that" at write time shouldn't have to wait for sleep to make the graph honest. This is the key mechanism that enables the push loop.
**How**: `src/memory/tools.py` `impl_remember()` accept optional `contradicts/supports/depends_on` lists. `src/memory/db.py` `insert_edge()` called in same transaction. Validate target ids exist; drop with warning otherwise. ~40 lines, no schema changes (edge table exists).

### 3. `challenges` annotation in recall() output (Low effort, Medium impact)

**What**: For each memory returned by `recall()`, add a field indicating whether it has incoming `contradicts` edges from active memories (i.e., is being challenged). Surface this as `challenged: true` alongside the normal result.
**Why**: Somnigraph `recall()` returns memories ranked by utility but doesn't flag contested ones. An agent acting on a challenged memory without knowing it's challenged makes the wrong call silently. This is a lightweight signal that doesn't require the full effective confidence formula — just "does this memory have any active contradicts edges?".
**How**: `src/memory/tools.py` `impl_recall()` — after retrieving results, query `db.py` for incoming contradiction edges on returned ids. Add `challenged` boolean to each result row. ~15 lines.

### 4. Rollback journal (Low effort, Medium impact)

**What**: For each `remember()` and `forget()` call, store a rollback entry (action type, memory id, before state, after state, timestamp). Expose `impl_rollback_list()` and `impl_rollback_apply()` MCP tools.
**Why**: No undo exists in Somnigraph. A mistaken `remember()` (wrong category, wrong content) requires manual `forget()` + re-entry. The rollback journal enables one-command recovery with provenance.
**How**: `src/memory/db.py` — add `rollback_entries` table (5 columns). `impl_remember()` and `impl_forget()` write entries. New MCP tools for list and apply. ~60 lines + schema migration.

### 5. Explicit `expires_at` per memory (Low effort, Low impact)

**What**: Add `expires_at TEXT` to memory schema. Accept via `remember()`. Sleep maintenance step flags any memory where `expires_at <= now` as stale, surfaces in recall output.
**Why**: Somnigraph has exponential decay (half-life per category) but no hard expiry. A memory about "Q2 planning deadline" doesn't hard-expire in Q3 — it just decays. Time-aware staleness is not age: the SENTINEL L4 benchmark shows age baselines fail in both directions (miss recent-but-expired, flag timeless-but-old).
**How**: `src/memory/db.py` schema migration. `tools.py` `impl_remember()` accepts `expires_at`. `scripts/sleep_nrem.py` maintenance step checks `expires_at <= now` and sets a `stale` flag. ~25 lines.

---

## Not Useful For Us

**N-ary hyperedges and programs.** Watch/quorum/trend programs are valuable for multi-agent team graphs where standing conditions need to fire without a query. Somnigraph is single-user; NREM sleep already handles contradiction detection offline. The complexity overhead doesn't justify the benefit.

**SENTINEL as a benchmark method.** SENTINEL tests mechanism presence (can a standing program fire?) not mechanism quality (are the confidence values right?). It scores 0 for pull systems by architecture, not by quality. Somnigraph's ground-truth judging + CV methodology is more honest for evaluating retrieval quality.

**TypeScript/Node.js stack.** Recall is Node.js 24+ only. No interoperability benefit with Somnigraph's Python/SQLite stack.

**Per-actor calibration at this scale.** Somnigraph is single-user, single-agent. Per-actor Brier scoring and the calibration factor are designed for multi-writer graphs (humans + multiple AI agents + CI jobs). The machinery adds overhead without benefit at single-user scale.

**Checker/Solver/Lattice ecosystem.** The broader Recall system (Checker for git-native verification, Solver for numeric compute, Lattice for code analysis) is enterprise tooling targeting engineering teams. Orthogonal to personal memory.

**The "no LLM in the loop" marketing.** Recall emphasizes that effective confidence requires no LLM. But the hard part — semantic contradiction detection (L2 SENTINEL) — still requires an LLM or KB. Somnigraph's NREM does LLM-mediated contradiction detection, which is more general. The "no LLM" claim covers only the surfacing layer, not the detection layer.

---

## Connections

**memv.md** — Third convergent arrival at write-time supersession. memv's `superseded_by` UUID, Kumiho's `Supersedes` edge, and Recall's `contradicts` + demotion are three independent systems reaching the same write-time reconciliation pattern. Strong evidence this is load-bearing. Recall's contribution: makes the demotion automatic (no query-time filter needed). memv's contribution: bi-temporal event/transaction time separation. The patterns are complementary, not competing.

**contradiction-reconciliation.md** — Chan et al. validate Recall's SENTINEL L2 claim from the opposite direction: NLI models can detect entailment contradictions ~20-40% of the time. This sets an honest ceiling for what L2 SENTINEL's LLM/Checker step achieves in production. Recall's surfacing floor (model-free) is solid; the detection ceiling (LLM-dependent) is imperfect, matching the paper.

**kumiho.md** — Closest architectural analog. Kumiho's tag re-pointing = Recall's effective confidence demotion; Kumiho's `Supersedes` edge = Recall's `contradicts` relation. Kumiho has formal AGM grounding (proved K*2–K*6); Recall has code-verified invariants (`evals.ts` invariant suite). Kumiho's prospective indexing (hypothetical future scenarios at write time) is the complement to Recall's admission-time write declaration — both invest structure at write time, in different directions.

**byterover.md / mempalace.md** — The cold-context challenge (see Summary Assessment below). ByteRover tops LoCoMo with BM25-only; MemPalace tops LongMemEval with verbatim matching. Both beat systems with sophisticated confidence tracking on static Q&A tasks.

**virtual-context.md** — Push-side architectural kin at a different layer. Virtual-context manages what fits in the context window via compression and paging; Recall manages what the agent believes via admission + effective-confidence. Both are substrate-level continuity layers. Recall's ambient hook (UserPromptSubmit mini-index) is the equivalent of virtual-context's paged working set: the substrate decides what the agent sees, not the agent deciding what to retrieve.

---

## Summary Assessment

Recall is an early-stage (v0.1.0, 13 stars) TypeScript memory substrate with sharp ideas about write discipline and trust transparency. Its strongest contribution is the combination of structured write proposals, admission-time quality gating, and graph-computed effective confidence — three things that together make corrections self-resolving at read time without an LLM. The ambient pre-prompt hook is architecturally distinctive: it makes memory push rather than pull by volunteering relevant cells before each prompt, not waiting for the agent to ask. No other system in the Somnigraph corpus has this mechanism.

The effective confidence formula (`evidence.ts:82-136`) is inspectable and deterministic, which is a genuine advantage over Somnigraph's LightGBM black box. Every term is traceable: stated × calibration + support − challenge. The asymmetric ceilings (0.15 support, 0.60 challenge) are the right shape. What the formula lacks is calibration against ground truth. The constants are intuition; Somnigraph spent two years of tuning studies to arrive at its scoring parameters. "Inspectable" and "well-calibrated" are not the same thing, and the formula has only been validated on synthetic SENTINEL streams, not against real-world retrieval quality.

**The cold-context challenge is the honest verdict on whether the machinery earns its complexity.** ByteRover (96.1% LoCoMo, BM25-only) and MemPalace (96.6% LongMemEval, verbatim-first) top their benchmarks without confidence machinery. Somnigraph is at 85.1% with a learned reranker and evidence-weighted graph. Write-path quality — the ByteRover finding — dominates retrieval sophistication on static Q&A tasks. Recall's confidence machinery earns its complexity in a different scenario: long-lived multi-writer agents where facts evolve and writers vary in reliability. That scenario has no public benchmark. SENTINEL is a self-referential proof that the mechanism fires, not an independent measurement of whether it fires correctly at real-world scale. The position: the confidence machinery is solving the right problem, with no evidence yet that it solves it well. The write-path discipline (structured proposals, admission attenuation, explicit evidence declaration) is the part that earns its complexity — it prevents the problem rather than measuring the solution.

For Somnigraph, the highest-priority steal is the admission firewall with confidence attenuation (~20 lines in `tools.py`). Second priority is write-time evidence declaration (extending `remember()` to accept `contradicts/supports` targets and immediately create edges, reducing NREM's workload and making the graph denser immediately). The ambient hook is architecturally compelling but requires hook infrastructure changes outside the memory server. The rollback journal is a pragmatic QoL improvement with no retrieval impact.
