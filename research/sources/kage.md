# Kage — Code-grounded verified memory for coding-agent teams (OKF-native, deterministic freshness, no LLM on the verdict path)

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

Kage is a memory layer for **coding agents working on a shared git repo**, not a personal
conversational memory. Its thesis is orthogonal to almost everything else in the corpus: a memory
is only useful if it is still *true about the code*, so every memory cites the source it describes
and Kage deterministically re-checks those citations at write, at recall, and when a diff moves the
cited code. No LLM sits on the trust verdict. TypeScript, stdio MCP server + CLI, ships as
`@kage-core/kage-graph-mcp`. Adopts Google's Open Knowledge Format (OKF) as its on-disk standard.

The engine is one 872 KB file: `mcp/kernel.ts` (with `mcp/index.ts` MCP wiring, `mcp/cli.ts`,
`mcp/okf.ts` the OKF adapter, `mcp/daemon.ts`).

### Storage & Schema
- Memories are **plain JSON "packets"** in `.agent_memory/packets/`, git-tracked and reviewed in
  the same PR as the code. `graph/`, `code_graph/`, `structural/`, `indexes/` are rebuildable
  (`kage refresh`). No database, no vector store required.
- Packet schema (`capture()`, kernel.ts:16485): `id, title, summary, body, type, scope, visibility,
  status, confidence, tags, paths` (cited files — the grounding), `stack, source_refs, context`
  (inferred *why/risk/trigger/verification*), `freshness` (ttl_days, last_verified_at,
  `path_fingerprints`, policy), `edges, quality` (votes, uses_30d, reports_stale, discovery_tokens),
  `created_at/updated_at, author_branch`.
- OKF export (`mcp/okf.ts`): `kage okf migrate` renders packets as OKF-conformant Markdown with
  YAML frontmatter; trust metadata rides in OKF-legal `x-kage-*` fields; round-trips byte-exact via
  a fenced `kage-state` JSON block. Can also *import* any third-party OKF bundle.

### Memory Types
14 fixed types (kernel.ts:15): `repo_map, runbook, bug_fix, decision, rationale, convention,
workflow, gotcha, reference, policy, issue_context, code_explanation, negative_result, constraint`.
Status lifecycle: `pending → approved → deprecated / superseded`. These are *engineering-artifact*
types, not Somnigraph's cognitive categories (episodic/semantic/procedural/reflection/meta).

### Write Path — the strongest part
`capture()` (kernel.ts:16399) is a real quality gate, not a passthrough:
1. **Private-span redaction** — `<private>…</private>` stripped from every text field *before* any
   validation or disk write.
2. **Serialized-dump rejection** — `isSerializedDumpTitle/Body` hard-rejects raw transcripts, tool
   output, and file-content dumps on every path (this is what previously bloated recall/graph).
3. **Secret/PII scan** — `scanSensitiveText` blocks the write.
4. **Citation validation (the hallucination gate)** — in strict mode, if *every* cited path is
   missing from the repo the write is rejected (kernel.ts:16446). Note: partial-missing is
   **warn-only**, not rejected — the "hallucinated citations rejected" claim holds only for the
   all-missing case.
5. **Ungrounded-chatter routing** — a rhetorical/frustrated message with no cited path is denied
   auto-approve and lands in the **pending inbox** tagged `needs-grounding`.
6. **Fingerprinting** — `memoryPathFingerprints` records SHA256 of each cited file at capture, and
   for TS/JS anchors to specific **symbol spans** (see below).

### Retrieval
Default recall (`recall` → `recallWithVectorScores`, kernel.ts:10516) is a **multi-signal additive
linear score**, not RRF and not a learned reranker:
- Lexical **BM25** (`scorePacketsBm25`, kernel.ts:9816) over weighted fields (title 4, tag 2.8,
  summary/path 2.4, type 1.8, body 1), k1=1.2 b=0.75, run three times: base terms, temporal terms,
  and a `recallQueryExpansion` semantic-synonym pass.
- A **sparse TF-IDF term-vector cosine** channel labeled "vector" (`scorePacketsVector`,
  kernel.ts:9847) — this is bag-of-words, **not** dense embeddings.
- Graph, path/type/tag, intent, and 30-day usage bonuses folded in via `recallBreakdown`; final =
  `textScore + graph + path_type_tag + intent + vector`, then `diversifyRecallEntries`.
- **Dense embeddings are opt-in** (`recallWithEmbeddings`, kernel.ts:10741): requires installing
  `@xenova/transformers` and running `kage embeddings build`; absent that artifact it silently falls
  back to the sparse path. So the default, dependency-free install has **no learned semantic
  channel**.
- **Just-in-time staleness gate**: before scoring, `recallStaleReason` filters out hard-stale
  packets (all citations deleted / TTL expired / reported stale / content-drifted); suppressed items
  are **recorded, not silently dropped**, and surfaced in a "Withheld (stale — not served)" section
  with a `kage reverify` hint.

### The differentiator — symbol-anchored, evidence-based freshness
`memoryPathFingerprint` (kernel.ts:3818) hashes each cited file; for TS/JS,
`symbolSpanHashesFromText` (via `extractSymbols`) hashes the span of each **named symbol** the packet
mentions and stores per-symbol SHA256 (kernel.ts:3920). At recall, `fingerprintPathContentChanged`
(kernel.ts:3959) marks a memory changed **only if one of its anchored symbols was edited or removed**
— unrelated edits elsewhere in the same file do *not* invalidate it. All deterministic. This is the
"0% stale-served" mechanism, and it is genuinely clever. Caveat: symbol anchoring is **TS/JS only**
(`ANCHOR_EXTENSIONS`, kernel.ts:3852); the multi-language code graph uses tree-sitter WASM
(py/go/rust/java/ruby), but staleness anchoring for those languages falls back to whole-file hashes
(coarser, more false-positive drift).

### Consolidation / Processing
Deterministic **compaction** (`compactProject`, kernel.ts:9531), not sleep and not LLM-mediated:
auto-deprecate hard-stale packets, prune individually-missing citations (re-fingerprint the rest),
and **report** near-duplicate clusters (merging is left to the agent). `detectContradictions`
(kernel.ts:3677) is deterministic too: shared cited path + distinctive-subject token overlap (≥0.5)
+ negation cue — no LLM. Auto-distillation turns session observations into pending draft packets.

### Lifecycle Management
TTL default 365 days; supersede chains record lineage (`supersedeMemory`, kernel.ts:20117);
`kageMemoryReconciliation` + `kage pr check` catch memory whose cited code changed at diff/PR time
and instruct the agent to reverify or supersede. Deprecate/superseded statuses hide from recall.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 96.17% R@5 / 98.72% R@10 on LongMemEval-S | `benchmarks/longmemeval-kage-retrieval.mjs` | **Retrieval recall, not QA.** Their own `docs/BENCHMARKS.md` states a plain BM25 baseline hits ~96.6% R@5 on this set and calls it "lexically tractable" — Kage does not beat BM25. **Not comparable to Somnigraph's 85.1% LoCoMo end-to-end QA.** |
| Hallucinated citations rejected at write time | `capture()` strict-mode gate, kernel.ts:16446 | Validated — but only when *all* cited paths are missing; partial-missing is warn-only. |
| 0% stale-served vs 100% for capture-everything stores | symbol-anchored fingerprint gate, kernel.ts:3959 + recall filter | Mechanism is real and deterministic. The "100% for others" figure is a constructed contrast, not a measured competitor benchmark. |
| Deterministic, no LLM on the verdict path | fingerprints + `detectContradictions` are pure functions | Validated by code — a genuine architectural choice, not a claim. |
| "Zero dependencies" | README badge | Overstated: `package.json` lists 5 runtime deps (MCP SDK, typescript, web-tree-sitter, tree-sitter-wasms, three.js). True sense: no vector DB / embedding model / API key for the *core sparse* path. |
| Trust score 100/100 | `kage benchmark --trust` self-test, `docs/TRUST.md` | Self-authored sandbox gates; measures Kage against its own harness, not competitors. |

The benchmark documentation is refreshingly honest — it explicitly refuses head-to-head tables and
tells the reader the retrieval number is a sanity check, not the headline. Credit where due.

---

## Relevance to Somnigraph

### What Kage does that Somnigraph doesn't
- **Evidence-based invalidation.** A memory cites an external artifact (source file/symbol) and is
  auto-withheld when that artifact drifts. Somnigraph's lifecycle (`db.py` decay, `scoring.py`) is
  purely **time-based** exponential decay with reheat-on-access — there is no notion of a memory
  becoming *false* because the world it described changed. This is a conceptually new axis.
- **Write-path quality gating.** `capture()` rejects dumps, scans secrets, and routes ungrounded
  captures to a pending inbox. Somnigraph's `tools.py remember()` auto-writes; STEWARDSHIP lists
  "write-path quality gating" as an explicit gap, and the Phase 18 sweep concluded write-path
  quality (not retrieval) is what LoCoMo leaders win on. Kage is independent corroboration.
- **Git-native team collaboration.** Memory is reviewed in the PR and shared across a team. Somnigraph
  is single-user by design — not a gap so much as a different problem.
- **Deterministic contradiction detection** as a cheap always-on filter, vs Somnigraph's LLM NREM
  pairwise classifier in `sleep_nrem.py`.

### What Somnigraph does better
- **Learned retrieval.** Somnigraph's 26-feature LightGBM reranker (`reranker.py`, NDCG 0.7958) and
  RRF fusion (k=14, Bayesian-tuned) are a real ranking model; Kage's default is a hand-weighted
  additive sum with a bag-of-words "vector" channel and no learning.
- **Real semantic retrieval by default.** sqlite-vec dense embeddings vs Kage's opt-in, dependency-
  gated embeddings that most installs never enable — Kage is effectively lexical.
- **Feedback loop.** Somnigraph has explicit utility ratings with measured Spearman r=0.70 and Hebbian
  co-retrieval PMI; Kage has only usage counts and up/down votes, unmeasured.
- **LLM-mediated consolidation.** Sleep's typed-edge inference and gap analysis do knowledge synthesis
  Kage's deterministic compaction cannot (it only prunes/deprecates/reports dupes).
- **End-to-end QA validation.** 85.1% LoCoMo QA vs Kage's retrieval-recall-only headline.

---

## Worth Stealing (ranked)

### 1. Evidence-based staleness: let a memory cite an external artifact and auto-invalidate on drift (High)
**What**: Store a content fingerprint (hash) of the source a memory is grounded in, and add a
recall-time gate that withholds (not deletes) the memory when the source changed, surfacing it as
"withheld — reverify" rather than silently serving a stale claim.
**Why**: Somnigraph's decay answers "is this old?" but never "is this still true?". For the subset of
Somnigraph memories that reference a concrete artifact (a file, a doc, a config, a commit), a
drift-triggered withhold is a lifecycle signal decay can't provide. Symbol-level anchoring
(invalidate only when the *named* thing changed, not any edit) is the non-obvious refinement.
**How**: New optional `grounding: {path, sha256, symbols[]}` field on memories in `db.py`; a
`stale_reason()` check in `scoring.py`/`tools.py recall()` mirroring Kage's `recallStaleReason`.
**Caveat**: applicability is narrow — most Somnigraph memories are conversational/personal and cite
no artifact. Scope this to memories that opt in with a citation, not a global mechanism.

### 2. Write-path quality gate with a pending inbox (Medium)
**What**: Before auto-approving a `remember()`, run cheap deterministic filters — reject serialized
dumps / tool-output pastes, scan for secrets, and route low-signal/ungrounded captures to a
`status=pending` inbox for later review instead of into live recall.
**Why**: Directly maps to the STEWARDSHIP "write-path quality gating" gap and the Phase 18 finding
that write quality, not retrieval, is the LoCoMo lever. A dump-rejection + pending-inbox pattern is
low-risk and independent of Somnigraph's data model.
**How**: A gate in `tools.py remember()` that sets `status="pending"` for captures failing a
heuristic (length/entropy/no-theme), reviewed during sleep — Somnigraph already has a pending
concept in the memory-server layer to build on.

### 3. Recorded (non-silent) suppression in the recall payload (Low)
**What**: When retrieval withholds a candidate (low score, decayed, contradicted), surface a compact
"withheld — reason" note rather than dropping it invisibly.
**Why**: Cheap transparency; helps the agent decide whether to reverify or dig deeper, and aids
debugging recall. Kage's "Withheld (stale — not served)" section is a clean UX pattern.
**How**: Extend the recall context block builder in `tools.py` to append a bounded withheld list.

---

## Not Useful For Us

### OKF adoption / git-native team sharing / PR review of memory
Kage's whole social model (multi-agent team memory reviewed in PRs) is orthogonal to Somnigraph's
single-user design. The OKF export format is nice standardization but solves a portability problem
Somnigraph doesn't have.

### Token-savings value ledger (`kage gains`)
Marketing/receipts feature. Somnigraph is a research artifact; a per-repo savings ledger doesn't
serve the documentation-first goals.

---

## Connections
- **Write-path discipline**: convergent with the Phase 18 sweep (ByteRover BM25-only, MemPalace
  verbatim, agentmemory write-time grounding — see `agentmemory.md`, `ai-memory-comparison.md`).
  Kage is a fourth independent system whose edge is write-time quality, not retrieval.
- **Deterministic-over-LLM**: contrasts with Somnigraph's LLM-mediated `sleep_nrem.py`; the
  cheap-deterministic-prefilter idea echoes candidate pre-selection debates elsewhere in the corpus.
- **Freshness/supersession**: the cite-and-invalidate lifecycle is a stronger, code-specific version
  of the supersession patterns noted in `memv`/`memos` analyses.

---

## Summary Assessment

Kage's core contribution is a genuinely novel lifecycle axis for the *coding-agent* niche:
**deterministic, evidence-based freshness**. Every memory cites the code it describes; a content
fingerprint (symbol-anchored for TS/JS) lets Kage withhold any memory whose grounding drifted, with
no LLM in the loop. Paired with a real write-path quality gate (dump rejection, secret scan,
ungrounded-chatter → pending inbox), this makes Kage the corpus's cleanest example of "trust the
store, not just retrieve from it." The benchmarking is unusually honest — it flags its own R@5 number
as retrieval recall on a lexically-tractable set that plain BM25 already wins.

The single most important takeaway for Somnigraph is the **evidence-based invalidation concept**: our
decay model can say a memory is old but never that it became false. For the opt-in subset of memories
that cite a concrete artifact, a drift-triggered withhold would add a signal decay structurally
cannot. The write-path gate is the more directly adoptable idea and corroborates a gap we already
flagged.

What's overhyped: the retrieval story. The default install is lexical BM25 + a bag-of-words "vector"
channel with no learned ranking and no dense embeddings — weaker than Somnigraph's reranker + RRF +
sqlite-vec on every retrieval-quality axis. The 96.17% R@5 is not comparable to our 85.1% LoCoMo QA,
and Kage's own docs say so. Its strengths are all on the *trust/write/lifecycle* side, and they only
apply where memories are code-grounded — which Somnigraph's personal, conversational memories mostly
are not. Verdict: two worth-considering ideas, both scope-limited; a MAYBE, not a DIVE.
