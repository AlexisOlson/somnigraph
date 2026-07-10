# Mnemos (arhuman) — cgo-free Go MCP server that indexes your files and answers with cited file#section + line ranges

*Generated 2026-07-03 by Sonnet agent reading the arhuman/mnemos repo*

**Name collision, read first**: this is a *different* project from the existing
`research/sources/ori-mnemos.md` analysis. That file covers **Ori-Mnemos**
(aayoawoyemi/Ori-Mnemos — TypeScript/Node, "Recursive Memory Harness," Q-value
learning, LinUCB stage selection, PPR graph traversal). This file covers
**arhuman/mnemos** — Go, a completely unrelated author, codebase, and design.
The two share only the bare name "Mnemos" (Greek for memory). No feature,
number, or code claim below should be conflated with the Ori-Mnemos analysis.

---

## Architecture

**Stack**: Go 1.25, single cgo-free binary (`CGO_ENABLED=0`), MIT license, author
Arnaud ASSAD ("Arhuman"). **Repo age verified from `git log`: initial commit
2026-06-29, most recent commit 2026-07-01, 17 commits total, latest tag
`v0.2.0`.** This is a brand-new project — about five days old at the time of
this analysis — not an established system. Star count is not meaningfully
assessable at this age and is omitted rather than guessed.

Core design: mnemos is a **document/codebase indexer with an MCP write-back
verb**, not a conversational agent-memory system. It scans a directory tree
(Markdown, Go source, plain text), chunks it, indexes it into SQLite
(FTS5 lexical, optional local vector embeddings), and serves it over MCP so an
agent can search/read/cite it — plus one write path (`mnemos.remember`) that
lets the agent add new OKF-formatted notes back into the same tree.

Module map (`internal/`): `app` (composition root), `config` (TOML/koanf),
`workspace` (single `MNEMOS_DIR` anchor, fixed subpaths), `cli` (Cobra),
`mcp` (MCP server + tool handlers), `memory` (the one verb layer both `cli` and
`mcp` call — search/read/context/list/remember/okfy/forget/move), `ingest`
(scanner, hasher, watcher, capture, OKF processing), `parse` (frontmatter +
format-specific parsing), `chunk` (token-aware splitting, golden-tested),
`storage` (SQLite via `modernc.org/sqlite`, pure Go, goose migrations), `search`
(bm25, hybrid RRF fusion), `embed` (no-op by default; ONNX behind a build tag),
`okf` (bundle conformance validator + auto-maintained `log.md`), `security`
(path confinement + secret scanning), `eval` (retrieval-quality evaluation).

### Storage & Schema

Single SQLite file. From `internal/storage/migrations/0001_init.sql`
(+ `0002_embeddings.sql` for vectors):

- **`documents`**: `id, uri, collection, content_hash, title, mime_type,
  size_bytes, modified_at, indexed_at, frontmatter_json`. `uri` is unique — a
  document's identity is its path relative to the ingest scan root (README
  calls out the resulting gotcha: two ingests with the same relative path
  silently overwrite each other).
- **`chunks`**: `id, document_id, ordinal, heading_path, content, tags,
  doc_type, token_count, start_line, end_line, metadata_json`. `start_line`/
  `end_line` are the backbone of the citation feature.
- **`chunks_fts`**: an FTS5 virtual table (content, heading_path, tags,
  doc_type) kept in sync via `chunks_ai`/`chunks_ad`/`chunks_au` triggers.
- **`links`**: `src_doc, dst_doc` — markdown cross-links captured as document-
  level edges. No weight, no type, no traversal at query time (see Retrieval).
- **`events`**: an append-only `id, type, payload_json, created_at` log —
  present in the schema but not wired into scoring or feedback anywhere in the
  code (no analogue to Somnigraph's `memory_events`-driven reranker features).
- **`embeddings`** (migration 0002): `chunk_id, model, dim, vector` — one
  little-endian float32 blob per chunk, only populated when the optional
  `embed` build is used and `mnemos reindex --embeddings` is run.

### Memory Types

None, in the Somnigraph sense (no episodic/semantic/procedural/reflection/meta
taxonomy). The closest analogue is a free-form `type` string in OKF frontmatter
(e.g. `idea`, `document`, `Task`) chosen by the caller of `mnemos.remember`
(`internal/mcp/tools_remember.go`). It is stored as `chunks.doc_type` and used
purely as an FTS5 column (weight 1.0, see Retrieval) — it does not change write
behavior, decay, or scoring beyond that.

### Write Path

`mnemos.remember` (`internal/mcp/tools_remember.go` → `internal/ingest/capture.go`):

1. `RenderOKF` serializes the note as an OKF markdown document: YAML
   frontmatter (`type`, `tags`, RFC3339 `timestamp`, `collection`) + blank line
   + body.
2. Filename is `"<type>-<YYYYMMDD>-<shorthash>.md"` where the hash is
   `xxhash64` of the body — identical same-day notes reuse a filename
   (idempotent re-capture); distinct notes diverge. Callers may instead pass an
   explicit `path` to place the note anywhere in the tree.
3. Content is **secret-scanned before write** (`internal/security/secrets.go`,
   see below) — findings block the write; only rule *names*, never matched
   values, are ever surfaced back to the caller.
4. `WriteFileAtomic` writes via same-directory temp file + rename.
5. The file is then ingested through the **identical** parse→chunk→store
   pipeline as any file on disk (`ingest.File` calls the same
   `Pipeline.IngestPath` the scanner/watcher use) — there is no separate "memory
   write" code path.

There is **no semantic deduplication, no near-duplicate check, no LLM
extraction/enrichment, no salience/quality gating** anywhere in this path
(confirmed by grep across `internal/` for `dedup`, `duplicate`, `confidence`,
`decay`, `contradiction`, `feedback`, `priority` — the only hits are an
unrelated "de-duplicated rule names" helper for secret-scan output and a
comment about not duplicating chunk display text). The write path's entire
sophistication is: render → secret-scan → write → index.

Secret scanner (`internal/security/secrets.go`): a fixed, ordered list of
regexes — AWS access key ID, PEM private-key blocks, GitHub tokens (classic +
fine-grained PAT), Slack tokens, Anthropic API keys, HuggingFace tokens, OpenAI
API keys, Google API keys, Stripe live keys, JWTs, DB connection-string
credentials, and a generic `key/secret/token/password = "..."` assignment
pattern. Specific-first ordering with a `seen` set so one secret isn't
double-reported under a broader rule.

### Retrieval

Default build is **lexical-only**. `internal/search/bm25.go`: user query is
tokenized into barewords, each token double-quoted (so no user input can ever
be interpreted as an FTS5 operator), joined with implicit AND, and run against
`chunks_fts` with per-column bm25 weights — `content 1.0, heading_path 2.0,
tags 1.5, doc_type 1.0` — plus a small deterministic `headingBoost` when a
query term also appears in the chunk's heading path. Results are ordered by
SQLite's (negative) bm25 rank; that's the entire lexical scoring formula — no
learned model, no metadata features.

Optional `embed` build tag adds semantic search (`internal/search/hybrid.go`):
`VectorRetriever` embeds the query with a local, pure-Go ONNX model
(`all-MiniLM-L6-v2`, 384-dim, via `gomlx`/`onnx-gomlx` — deliberately not the
cgo C++ ONNX Runtime) and does a **brute-force linear cosine scan** over the
`embeddings` table (no ANN index; ADR 0003 tracks this as a known scaling
limit, ~20ms at 100k chunks, an approximate index is "the next step" if ever
needed). `HybridRetriever` runs the lexical and vector retrievers concurrently
at an over-fetched candidate depth and fuses the two ranked lists with
**Reciprocal Rank Fusion**, `rrfK = 60` (`internal/search/rerank.go`) — despite
the filename, **`rerank.go` implements RRF fusion only; there is no learned
reranker, no feature extraction, no model of any kind**. If the vector side is
empty (no embed build, or no model installed), RRF over one non-empty list
degrades to plain lexical ranking.

Markdown cross-links are captured into the `links` table at ingest time but
are **explicitly not traversed at retrieval time** — the README states this
directly ("markdown links are captured as edges (stored, not yet traversed)").
There is no graph expansion, no PPR, no adjacency scoring of any kind in the
live retrieval path.

### Consolidation / Processing

None. No offline pass, no summarization, no cluster judgment, no contradiction
detection, no theme normalization. `mnemos watch` re-ingests changed files
(content-hash skip for unchanged ones) but this is incremental re-indexing, not
consolidation in the CLS/sleep sense.

### Lifecycle Management

Minimal, filesystem-CRUD-shaped, not memory-lifecycle-shaped:

- **`forget`** (`internal/ingest/forget.go`): de-indexes then deletes the file;
  idempotent; index cleared before disk mutation so a crash leaves the system
  self-healing on the next run.
- **`move`**: renames a file or directory subtree and re-indexes under the new
  path; inbound markdown links to the old path are **not** rewritten in this
  version (logged as a warning).
- **`watch`**: fsnotify + debounce, incremental reindex, removes deindexed
  files.

No decay, no confidence, no priority, no archival/dormancy, no versioning of
content over time (beyond the OKF `log.md` per-directory changelog, which
records Creation/Update/Deprecation bullets but does not feed retrieval).

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Zero dependencies... one cgo-free Go binary" | `go.mod` uses `modernc.org/sqlite` (pure Go); `gomlx`/`onnx-gomlx` for the optional embed build; Makefile forces `CGO_ENABLED=0` | Validated |
| "No vector database. No Ollama. No Python or Node service." | Confirmed — SQLite is the only store; embeddings (when enabled) run in-process via pure-Go ONNX inference, no external service | Validated |
| "Every result... citing the exact `file#section` and line range" | `chunks.start_line`/`end_line` stored per chunk; `model.Result` carries `HeadingPath`/`StartLine`/`EndLine`; CLI/MCP output both render them | Validated — this is the one feature the whole design is built around |
| Retrieval-quality table: Hit@1 0.83 lexical / 1.00 semantic on git-recipes; Hit@1 0.00 lexical / 0.57 semantic on onpage-seo | `internal/eval` auto-derives held-out query→doc pairs by stripping each example query from its own host chunk; author's own README caption: "smoke signals, not benchmarks" (N=6 and N=7 documents) | Self-reported, honestly caveated by the author, but N is far too small to generalize, and it is a **document-retrieval** eval, not a conversational-memory QA benchmark — not comparable to LoCoMo/PERMA-style numbers at all |
| "Safe by default": path confinement + secret scanning | `internal/security/paths.go` (confinement guard rejecting `..`, absolute escapes, symlink escapes, `.mnemos/` access) and `secrets.go` (regex scanner) both exist and are wired into the write path | Validated |
| Markdown links become graph "edges" | `links` table exists and is populated at ingest | Technically true but the README itself says they are "not yet traversed" — an inert edge list, not a working graph feature |
| Performance: lexical ~2.75ms/query, hybrid ~30ms at 10k chunks | `docs/architecture.md` cites a benchmark/profile run (`go test -bench`) with CPU/alloc breakdown attributing ~90% of hybrid bytes to blob decode | Plausible, self-measured, not independently reproduced here |

---

## Relevance to Somnigraph

### What Mnemos does that Somnigraph doesn't

- **Human-readable, git-diffable, file-per-memory storage.** Every "memory" is
  a literal OKF markdown file on disk with YAML frontmatter you can `cat`,
  `git diff`, or edit by hand — Somnigraph's memories are rows inside a SQLite
  DB, invisible outside the MCP tools or direct SQL.
- **Verifiable line-range citations as a first-class schema field**
  (`chunks.start_line`/`end_line`). Somnigraph has no analogous concept because
  its memories are freeform agent-written text, not derived from an ingested
  file — there is no "source location" to point back to. If Somnigraph ever
  grows a source/repo-ingestion mode, this is the schema shape to copy.
- **Multi-format ingestion** (Markdown, Go source, plain text, each with a
  dedicated chunker in `internal/chunk`). Somnigraph only ever stores explicit
  `remember()` calls; it never indexes a codebase or doc tree.
- **A conformance-checked portable interchange format** (the OKF validator,
  `internal/okf/validate.go`, with numbered E/W rule codes). Somnigraph's
  schema is internal-only and not designed for cross-tool portability.
- **Single self-contained binary deployment**, no Python/API-key dependency.
  Somnigraph requires a Python process, `sqlite-vec`, and an OpenAI API key for
  embeddings.
- **An auto-derived retrieval eval that needs zero manual annotation**
  (`internal/eval`: strip an in-corpus example query from its own host chunk,
  check whether retrieval still finds the document). Somnigraph's ground truth
  (`scripts/build_ground_truth.py` + `judge_ground_truth.py`) requires curating
  and LLM-judging real queries — much higher-effort, but also much more
  representative of actual usage (see Worth Stealing #1 for the tradeoff).

### What Somnigraph does better

- **A learned reranker vs. RRF-only fusion.** Mnemos's `internal/search/
  rerank.go` is Reciprocal Rank Fusion, full stop — no features, no model,
  nothing beyond what Somnigraph's `scoring.py` `rrf_fuse()` already did before
  it was superseded by `reranker.py`'s 31-feature LightGBM model
  (`docs/architecture.md` § The reranker).
- **A measured, working feedback loop.** Mnemos has no analogue at all to
  Somnigraph's `recall_feedback()` / EWMA-Bayesian blend in `scoring.py`
  (Stage 1, `FEEDBACK_COEFF = 5.15`) — nothing in mnemos closes any retrieval
  feedback loop.
- **Graph edges that actually participate in retrieval.** Mnemos's `links`
  table is populated but explicitly inert at query time; Somnigraph's
  `memory_edges` (typed, weighted, with `linking_context` embeddings) drive
  real PPR-based adjacency expansion in `scoring.py`.
- **Any temporal modeling.** No decay, no dormancy, no confidence, no priority
  in mnemos — Somnigraph's `decay.py`, per-memory `confidence`/`priority`/
  `decay_rate` columns, and shadow-load tracking have no counterpart here.
- **Offline consolidation.** Mnemos has zero equivalent to Somnigraph's
  NREM/REM sleep pipeline (`scripts/sleep_nrem.py`, `sleep_rem.py`) — no
  contradiction detection, no merge/archive, no summary generation.
- **A conversational-memory benchmark result.** Somnigraph reports 85.1%
  LoCoMo QA (Opus judge) on a memory-QA task; mnemos's numbers are a
  self-described "smoke signal" on document retrieval over its own 6-7
  document example bundles — a different task, at a different scale of
  evidence, entirely.

---

## Worth Stealing (ranked)

### 1. Auto-derived held-out retrieval smoke test (Medium effort)
**What**: `internal/eval`'s trick — take a memory whose content contains a
distinctive, quotable phrase, hide that exact phrase from the memory at index
time (or synthesize a query from the *rest* of the memory), then check whether
`recall()` still surfaces the source memory in the top-K. Zero manual
annotation required.
**Why**: Somnigraph already tried something adjacent — the "Single-shape
synthetic GT" experiment (`docs/architecture.md` § What didn't work) found
that naive synthetic anchors teach shape-memorization rather than retrieval
semantics, and that fix required careful (query, context) asymmetry
engineering. Mnemos's simpler doc-level version (hide a real excerpt, not
generate a synthetic one) sidesteps that specific pathology because the
"query" is always real prose the memory already contains, just relocated.
**How**: A lightweight nightly/CI smoke script, *not* a replacement for the
real GT pipeline: pick N recent memories, redact a sentence from the copy used
to build the query, run `recall()`, check the source memory's rank. Cheap
regression guard against retrieval regressions between real GT refreshes,
complementary to (not a substitute for) `build_ground_truth.py`.

### 2. Diff secret-scanning regex coverage against `privacy.py` (Low effort)
**What**: Mnemos's rule list (`internal/security/secrets.go`) includes several
patterns worth checking against Somnigraph's `privacy.py`: HuggingFace tokens
(`hf_...`), Stripe live keys (`sk_live_`/`rk_live_`), GitHub fine-grained PATs
(`github_pat_...`), and a generic `(?i)(api[_-]?key|secret|token|password)\s*
[:=]\s*['"][^'"]{16,}['"]` catch-all.
**Why**: `docs/architecture.md`'s P2 priority explicitly tracks
write-path secret redaction as already-shipped-but-worth-extending
(2026-07-01 session extended coverage to bearer tokens, PEM blocks, credit
cards, AWS/Google keys). This is a small, concrete list to diff against.
**How**: A 20-minute pass comparing mnemos's regex list to `privacy.py`'s
patterns; add whichever of the four above (if any) are missing.

Beyond these two, there is genuinely little else to take: mnemos's retrieval
stack (lexical FTS5 + optional brute-force-cosine + RRF) is strictly behind
what Somnigraph already built and moved past, and its write path has no
enrichment, dedup, or gating logic of any kind — it is, in effect, **the file-
storage half of a plain-markdown memory system with no learning machinery
bolted on**, closer in spirit to a lightweight RAG-over-your-repo tool than to
a persistent conversational memory system.

---

## Not Useful For Us

### The OKF bundle format / conformance validator
Somnigraph's memory model is not file-based and has no reason to adopt a
portable markdown interchange format; doing so would be a full architecture
change, not a feature addition.

### RRF-only "rerank.go"
Somnigraph already has (and moved beyond) exactly this fusion stage; mnemos's
version adds nothing new to learn from.

### Single-binary/no-Python deployment model
Not applicable — Somnigraph is intentionally a Python/MCP server; rewriting in
Go for deployment simplicity is out of scope for a research artifact.

### Markdown links as edges
This is the exact feature Somnigraph already has, and mnemos's version is
strictly weaker (untyped, unweighted, and explicitly not traversed at query
time — Somnigraph's edges are typed, weighted, and drive real PPR expansion).

---

## Connections

*(Every system named here was cross-read against its own source-analysis file
before being cited, not inferred from index summaries. Systems I did not read
are not named.)*

Mnemos lands in a well-populated corpus cluster: **plain-file/markdown store +
lexical (or lexical-first) retrieval, no learned reranker, no feedback loop, no
live graph traversal, no consolidation, no QA benchmark.** The recurring lesson
of that cluster, documented in `docs/similar-systems.md` (§ "Systems from the
carsteneu survey": the surveyed file-store systems "sit behind Somnigraph on
retrieval ... and win, where they win, on the write and consolidation paths"),
and made Priority 2 in `STEWARDSHIP.md`, is that its members earn their keep on
**write-path quality, not retrieval**. Placing mnemos against that lesson is the
point, and the placement is unusual enough to state precisely.

Closest kin, in order of similarity (each verified by reading the linked file):

- **engram-gentleman.md** — the single closest sibling. Also a single Go
  binary, also SQLite + FTS5 as the retrieval engine, also *no embeddings by
  default, no learned reranker, no feedback loop, no graph, no decay, no
  benchmark.* Where they diverge is instructive: engram invests in the **write
  path** (a rule-derived `topic_key` upsert with `revision_count`, a 15-minute
  hash-dedup window, and save-time FTS5+BM25-floor conflict surfacing that hands
  an LLM judge a shortlist) — mnemos has *none* of that (no dedup, no conflict
  detection, no upsert-by-identity; every `remember` is a fresh file). Mnemos
  instead spends its one increment of sophistication on the citation/line-range
  guarantee and the optional semantic build. Two near-identical Go/SQLite-FTS5
  systems making opposite choices about *where* to exceed the minimum.
- **acontext.md** (Acontext) — shares the "human-editable markdown files are the
  store" philosophy and deliberately-dumb retrieval (Acontext is grep +
  progressive disclosure, no ranking at all). But Acontext is the *inverse* of
  mnemos on the write path: its whole substance is an LLM distillation +
  skill-writing agent with an `applies_when` salience gate and update-vs-create
  routing. Mnemos is more retrieval-capable (real FTS5 bm25 vs grep) but far
  thinner on write curation.
- **memoir.md** — a git-backed versioned markdown store, sharing mnemos's "plain
  files, human-inspectable, portable" positioning. But memoir's retrieval is an
  LLM scanning every semantic path (weaker than mnemos's FTS5), while its write
  path is *rich* (a typed, confidence-gated conflict-merge policy). Same split
  again: memoir invests in write-path merge discipline; mnemos invests in
  neither dedup nor merge.
- **diffmem.md** (DiffMem) — further toward zero on retrieval infrastructure: no
  vectors, no BM25, no database at all, a multi-turn LLM agent shell-exploring a
  git repo. Mnemos's actual SQLite FTS5 index is a concrete step up in retrieval
  mechanism (and a step down in temporal/agentic flexibility).
- **nanobot.md** (Nanobot) — LLM-curated markdown files + git-versioned "Dream"
  consolidation, but *no retrieval engine* (whole-file injection + manual grep).
  Mnemos's FTS5 index is a step up on retrieval; Nanobot is a step up on
  write-path curation (its Dream prompt is an explicit salience/dedup gate) and
  on lifecycle (git diff/restore of consolidation output), both absent in mnemos.

The accurate placement: mnemos sits in the plain-file/lexical cluster, but is an
**outlier within it** — most members (acontext, memoir, nanobot, engram) are
there because they bet on write-path quality over retrieval sophistication, and
mnemos bets on *neither* (its write path has no dedup, salience gate, LLM
curation, or conflict detection at all). Its single genuine investment is on a
third axis those peers don't foreground: **verifiable source provenance**
(mandatory `file#section` + line-range citations). Two caveats worth carrying:
(1) that guarantee is cheap for mnemos precisely because it indexes files that
already have a stable identity (path + line number), which is exactly why it
does not transplant to Somnigraph, whose memories are agent-authored prose with
no source file to point back to (see Not Useful For Us); and (2) mnemos is alone
in this group in being explicitly scoped to **your project's docs/ADRs/source
code** rather than conversational history — "cited RAG over your repo with a
`remember()` bolt-on" is a fairer description than "agent-memory system."

---

## Summary Assessment

Mnemos is a five-day-old, single-author Go project (17 commits, v0.2.0,
initial commit 2026-06-29) that is best understood as a well-engineered,
honestly-documented **document/codebase RAG indexer with an MCP interface and
a write-back verb**, not a peer of the conversational agent-memory systems
this corpus otherwise surveys. Its core, genuinely good contribution is
architectural discipline: a single cgo-free binary, SQLite FTS5 bm25 as a
default that "just works," an optional pure-Go semantic layer that degrades
gracefully when absent, and — the actual headline feature — every retrieval
result carrying a verifiable `file#section` + line-range citation back to a
real file on disk.

For Somnigraph specifically, there is very little to adopt algorithmically.
Mnemos has no learned reranker (its `rerank.go` is RRF only, the exact stage
Somnigraph already surpassed), no feedback loop, no decay, no confidence/UQ,
no working graph traversal (links are captured but explicitly inert), and no
consolidation of any kind — confirmed by grepping the entire codebase for
those concepts and finding nothing. In effect it duplicates the file-storage
half of a plain-markdown memory system (a closer cousin to Acontext than to
anything else in the corpus) while adding no learning machinery Somnigraph
doesn't already have and exceed. The two things worth actually taking are
narrow and cheap: (1) its zero-annotation held-out retrieval smoke test as a
lightweight regression guard alongside — not instead of — Somnigraph's real
ground-truth pipeline, and (2) a short diff of its secret-scanning regex list
against `privacy.py` for a few patterns (HuggingFace tokens, Stripe live keys,
GitHub fine-grained PATs) that may be missing.

The one place mnemos is unambiguously ahead is honest self-assessment of its
own evidence: its README explicitly labels its Hit@1/Recall/MRR table "smoke
signals, not benchmarks" given N=6-7 documents, and explicitly states its
graph edges are "not yet traversed" rather than dressing up either as more
than it is. That discipline is worth noting, even where the underlying system
itself is too early and too thin to offer much else.
