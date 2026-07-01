# fidelis — zero-LLM verbatim-fidelity retrieval layer built on top of mem0

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

fidelis (Hermes Labs, solo founder Roli Bosch; MIT; v0.0.91, pre-release) is **not a standalone
memory store**. It is a retrieval + QA-scaffold layer that wraps **mem0** (`mem0ai>=2.0.0,<3.0` is
a hard dependency; `chromadb>=0.5.0`, `ollama>=0.4.0`). All candidates come from `memory.search()`
on a mem0 `Memory` instance backed by a local Chroma vector store (`recall_b.py:317`,
`recall_hybrid.py:239`). The pitch: **"zero-LLM" retrieval** — the hot path never calls an LLM, so
the store returns your original passages verbatim rather than paraphrases.

### Storage & Schema
- mem0's Chroma collection at `~/.cogito/store` (dir name is the pre-rename codename "cogito").
  SQLite is mem0's metadata sidecar. Embeddings: Ollama `nomic-embed-text` (~280 MB, local).
- Memory unit is an **atomic fact text + embedding + score** — no typed schema. The evidence file's
  "8 schema fields" is an overclaim; real fields are text/id/embedding/score (3–4). No
  entities, no typed edges, no themes, no priority, no valid_from/valid_until.

### Memory Types
None. Flat verbatim fact store. No episodic/semantic/procedural distinction, no layers. (The
evidence file's "layered memory ❌ FALSE" is correct — the two retrieval *paths* are code paths,
not lifecycle tiers.)

### Write Path
Two write modes, **both LLM-mediated at write time** (the "zero-LLM" label is retrieval-only):
- `/add` → mem0's LLM fact-extraction using a custom technical-extraction prompt
  (`config.py:16-42`), run through a **local** Ollama model (`qwen3.5:0.8b` default). "Extract
  generously" — no salience/quality gate; it dedupes only via mem0's internal logic.
- `seed.py` / `/store` → a curation LLM extracts atomic facts from markdown chunks (split by
  heading, `_chunks_from_file`), then writes each fact **verbatim** bypassing mem0 extraction.
  State tracked by file-mtime hash in `seeded.json` (re-seed only changed files).
- `degrade.py` provides **graceful degradation**: writes queue locally when the upstream LLM is
  unreachable and replay with exponential backoff.
- No write-time dedup beyond mem0, no salience gating, no entity resolution.

### Retrieval (the actual product)
Two implementations, both zero-LLM by default:
- `recall_b.py` — **zero-LLM structural query decomposition**: strip stopwords + question
  scaffolding, expand via a precomputed `vocab_map`, generate up to 8 sub-queries
  (original → stripped phrase → vocab-expansion terms → bigrams → trigrams → tokens), run each as
  a separate mem0 vector search, **RRF-fuse** (k=60) across runs, then **cosine-rerank** the merged
  pool against the original-query embedding blended 0.3·RRF + 0.7·cosine. A **score floor**
  (cosine<0.25 AND top-vs-mean gap<0.1) returns empty rather than low-confidence junk.
- `recall_hybrid.py` — adds **BM25** (`bm25s`, optional; degrades to pure-dense if absent) over the
  candidate pool, plus nomic's `search_query:`/`search_document:` prefix convention, plus a regex
  query router (`classify_query`: skip / temporal-counting-"llm" / default). Fuses BM25 + dense +
  mem0's own ranking via RRF, blends with cosine.
- **Optional LLM tiers** (`filter`, `flagship`) are opt-in reranks. Crucially they use
  **integer-pointer output**: the LLM is shown numbered candidates and returns only a JSON array of
  indices (`_parse_indices_1based`); the server dereferences to the original stored text. The LLM
  **cannot rephrase memory content** — this is the "fidelity" guarantee. README admits the flagship
  tier currently escalates ~80% of queries vs the intended ~10% (an 8× cost miss).

### Consolidation / Processing
None in the sleep sense. `calibrate.py` is a **one-time offline** LLM pass that samples stored
memories and asks the LLM to emit a `vocab_map` (plain-English word → short technical terms that
appear verbatim in facts, e.g. `"freeze" → ["timeout","cascade","blocked"]`). A `snapshot` command
builds a compressed ~741-token index for session-context injection. No pairwise edge detection, no
merge/archive, no gap analysis.

### Lifecycle Management
None. No decay, no versioning/supersession, no time-travel, and per the evidence file no HTTP
delete/forget endpoint. Writes accumulate.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| 83.2% R@1, 98.3% R@5 on LongMemEval-S | `experiments/zeroLLM-FLAGSHIP-evidence/`, 470 questions | Plausible but **retrieval recall, not QA** — not comparable to our 85.1 LoCoMo QA |
| 73.0% end-to-end QA on LongMemEval-S (Wilson CI [68.7,77.0]) | same evidence dir, gpt/haiku reader | Plausible; **different benchmark** than our LoCoMo, so no head-to-head |
| "Zero-LLM" / $0 per query | Code confirms retrieval path makes no LLM call | True **for retrieval only** — write path uses a local LLM (mem0 extraction / seed curation) |
| Verbatim fidelity — no paraphrase | Integer-pointer reranker (`_parse_indices_1based`), verbatim `/store` | Validated — real structural guarantee |
| Beats Mem0 (~66–70), Zep 71.2, Supermemory 81.6 | README comparison table | Cites others' published numbers; self-number is single-run, single author |
| "Made by" / 486 tests passing | `tests/` | Plausible; test count not audited here |

**Sharpest correction**: fidelis is a **retrieval-and-scaffold wrapper over mem0**, not an
independent engine — the vector store, write-time extraction, and persistence are all mem0/Chroma.
The headline **83.2% is R@1 (retrieval recall)**, a different metric than Somnigraph's 85.1% LoCoMo
end-to-end QA, and the **73.0% QA is on LongMemEval-S**, not LoCoMo. No cell here is comparable to
our numbers without re-running the same benchmark.

---

## Relevance to Somnigraph

### What fidelis does that Somnigraph doesn't
- **Query-side vocabulary bridge (`vocab_map` + `calibrate.py`).** An offline LLM pass builds a
  static plain-English→technical-term expansion dict, applied at recall with zero runtime LLM cost
  (`recall_b._expand_with_vocab_map`). This targets exactly the failure Somnigraph's
  `docs/multihop-failure-analysis.md` names as the ~88% retrieval ceiling — the vocabulary gap
  between query language and stored language. Somnigraph attacks it **document-side** (synthetic
  bridge nodes injected during sleep, per `project_extraction_v6`); fidelis attacks it **query-side**.
- **Zero-LLM structural query decomposition** (`_build_subqueries`): stopword/scaffolding strip +
  n-gram sub-queries + per-subquery RRF. Somnigraph issues a single query to fts.py/embeddings.py;
  it has no recall-time query expansion or multi-subquery fusion.
- **Integer-pointer reranking**: when an LLM does rerank, it emits only indices, guaranteeing the
  returned text is byte-identical to the store. (Somnigraph already returns stored text verbatim, so
  the guarantee is implicit for us — but the pattern is a clean way to keep an LLM reranker
  fidelity-safe if we ever add one.)
- **Graceful write degradation** (`degrade.py`): local queue + exponential-backoff replay when the
  write-time LLM is down.

### What Somnigraph does better
- **Everything on the learning/graph/consolidation axes.** fidelis has no learned reranker (vs our
  26-feature LightGBM in `reranker.py`), no feedback loop (no equivalent to our EWMA/UCB utility
  ratings, Spearman r=0.70), no graph/PPR (`scoring.py`), no typed edges, no sleep consolidation
  (`sleep_nrem.py`/`sleep_rem.py`), no decay/lifecycle, no salience or write-path quality gating.
- Its retrieval is hand-tuned constants (RRF k=60, cosine weight 0.7, floors) calibrated on one
  benchmark; Somnigraph's fusion is Bayesian-optimized and its ranking is learned on 1032 real
  queries.

---

## Worth Stealing (ranked)

### 1. Query-side vocabulary bridge (`vocab_map`) (Low/Medium effort)
**What**: A precomputed dict mapping plain-English query words to the short technical terms that
appear verbatim in stored memories, built by a single offline LLM calibrate pass over sampled
memories, then applied at recall with zero runtime cost to add sub-queries.
**Why**: Somnigraph's own `multihop-failure-analysis.md` identifies the query↔store vocabulary gap
as the dominant retrieval ceiling (~88%). We currently only bridge it document-side (synthetic
nodes at sleep). A query-side expansion is complementary and cheap, and could be tested on the
LoCoMo multi-hop subset where the vocabulary gap was measured.
**How**: Add an offline builder (analogous to `calibrate.py`) that samples memories during a REM
sleep step and emits a themes→variants expansion table (we already have `theme_variants.json`
plumbing in `constants.py` / `DATA_DIR`). At recall in `fts.py`/`tools.py`, expand the query into
the FTS/vector channels using the table before RRF fusion. Guard with an ablation — query expansion
can add noise, which is why fidelis pairs it with a score floor.

### 2. Score-floor "return nothing" gate (Low effort)
**What**: If top cosine < floor AND top-vs-mean gap < threshold, return an empty result rather than
low-confidence memories (`recall_b._cosine_rerank`, floor 0.25 / gap 0.1).
**Why**: Somnigraph always returns its top-k; a confident-abstain gate could reduce injecting
irrelevant memories into context (relevant to the proactive-injection floor study in
`docs/proactive-injection.md`, which is itself about a floor that carries signal the cliff cutoff
misses).
**How**: A post-rerank check in `tools.py` recall path; expose the floor as a config constant and
sweep it against the existing feedback logs before enabling.

---

## Not Useful For Us

### Zero-LLM / $0-per-query positioning
Somnigraph already runs local-first and its retrieval hot path is non-LLM (BM25+vector+LightGBM);
the "zero-LLM" selling point is a marketing frame around a wrapper, not a capability we lack.

### mem0 dependency + integer-pointer fidelity as a *novel* guarantee
We store and return memories verbatim already; we don't paraphrase on read, so the integer-pointer
mechanism solves a problem (LLM-rewritten memory) we don't have. And building on mem0 is the
opposite of our architecture — we own the store.

### Snapshot/session-injection index and the LLM filter/flagship tiers
The flagship tier is self-admittedly miscalibrated (80% escalation vs 10% target); the snapshot is a
compaction trick, not a consolidation model.

---

## Connections

- **mem0**: fidelis is a *consumer* of mem0, not a competitor — same relationship several wrappers
  in our corpus have to an upstream engine. Cross-ref any mem0 analysis; the write-time extraction
  quality is mem0's, not fidelis's.
- **Write-path-quality thesis** (Phase 18 source sweep — ByteRover/agentmemory/MemPalace): fidelis
  is the mirror image. It bets everything on **verbatim faithful *retrieval*** and explicitly
  refuses to touch the write representation, whereas the Phase 18 finding was that the LoCoMo/LME
  leaders win on **write-path quality**. fidelis's own weak qtypes (temporal 58%, preference 37%)
  are exactly the ones that need write-time structuring it doesn't do — quiet corroboration that
  verbatim-only retrieval has a ceiling.
- **Vocabulary bridge**: convergent with Somnigraph's synthetic-bridge-node approach
  (`project_extraction_v6`) but from the query side — two independent systems arriving at "close the
  query↔store vocabulary gap," which strengthens the case that it's the real lever.

---

## Summary Assessment

fidelis's core contribution is a **clean, honestly-scoped zero-LLM retrieval layer** that returns
stored text verbatim, plus a genuinely nice **query-side vocabulary bridge** built by one offline
LLM pass. The engineering is tidy and the README is unusually candid about limitations (80%
flagship escalation, weak temporal/preference qtypes, mem0 dependency). But it is architecturally a
**thin wrapper over mem0 + Chroma**: no learned ranking, no feedback loop, no graph, no
consolidation, no decay, no lifecycle. On every axis Somnigraph treats as its differentiators,
fidelis has nothing.

The single most valuable takeaway is the **`vocab_map` query-expansion idea** — a low-cost,
offline-built, query-side attack on the exact vocabulary gap our own multi-hop analysis flagged as
the retrieval ceiling. It's complementary to our document-side synthetic bridges and worth an
ablation. Everything else (zero-LLM framing, integer-pointer fidelity, verbatim return) is either
something we already have or a solution to a problem our architecture doesn't have.

What's overhyped: the "83.2%" headline is R@1 retrieval recall on LongMemEval-S, not end-to-end QA
and not on our benchmark — not comparable to our 85.1 LoCoMo. The "zero-LLM" claim is retrieval-only;
writes still call a (local) LLM through mem0. Verdict: MAYBE — one idea (vocab_map) is worth a
revisit; the rest is skip.
