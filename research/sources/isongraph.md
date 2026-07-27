# ISONGraph — a tabular, header-once serialization format for property graphs in LLM context windows (six language ports)

*Generated 2026-07-27 by Opus 5 agent reading the repo at commit `96c1ba6`*

---

## Overview

**Repo**: https://github.com/isongraph/isongraph (3 stars, last push 2026-07-24; HEAD `96c1ba6` dated 2026-07-23, **a single squashed commit** — no development history is recoverable). 117 tracked files (16 `.ts`, 15 `.md`, 10 `.cs`, 9 `.py`, 8 `.js`, 3 each `.hpp`/`.cpp`/`.rs`/`.ison`), ~45k LOC total.
**Paper**: none. The README ships a `@software{isongraph2025}` BibTeX block; there is no preprint, no evaluation writeup outside `benchmark/*/BENCHMARK.md`.
**License**: MIT (`LICENSE`, "Copyright (c) 2025 Mahesh Vaikri"). Single author.

**Problem addressed**: JSON, GraphML, JSON-LD and RDF spend context-window tokens re-stating field names on every record and wrapping values in punctuation. For a graph handed to an LLM as prompt text, that overhead is pure cost. ISONGraph proposes a tabular encoding where each node type and each relation type is one block with a single header row.

**Core approach**: A graph serializes to `nodes.<type>` and `edges.<REL>` blocks. Each block emits one header line of field names and then one whitespace-delimited row per entity. Edge endpoints are written as `:type:id` references. Values are bare unless they contain a delimiter, in which case they are double-quoted. A streaming variant (ISONL) packs each row onto one line as `block|fields|values`. Around the format sits a conventional in-memory property-graph library (O(1) node/edge maps, in/out adjacency indexes, BFS/DFS traversal, a Cypher-ish query language "ISONQL", a schema validator "ISONGraphantic", and a deterministic force-directed SVG renderer), ported to Python, TypeScript, JavaScript, Rust, C++ and C#.

**Key claims**:
- "The most token-efficient way to represent graph data in LLM context windows" (README:14) — **superlative, and false as stated**; see Benchmark Audit. A trivial variant that drops the `:type:` prefix from edge endpoints is 22% smaller (verified below).
- 68.6% token savings vs pretty JSON on a 7-dataset corpus — **fully reproducible; I reproduced every number in both token tables to the digit.**
- 90.0% LLM accuracy vs 87.0% for JSON, 89.0% for Cypher, 58.0% for RDF/Turtle — **asserted only.** No result artifact is committed, and the benchmark script's only output is a `.log` file that `.gitignore` excludes by pattern.
- "Bit-identical geometry across languages" for the visualizer (README:101) — untested here; note Rust and C++ have no viz module at all, so the claim covers 4 of 6 ports.
- "Multi-language" with "full feature parity" (README:117-122) — **the six ports do not agree on the wire format.** Three mutually incompatible quoting rules, each documented in source comments as "the standardized rule shared by all ISON ports."

---

## The Format

### Syntax and design

`social.ison`, verbatim (tabs shown as spans):

```
nodes.person
id	name	age	role
alice	Alice	30	Engineer
bob	Bob	25	Designer

edges.KNOWS
source	target	since
:person:alice	:person:bob	2020
:person:bob	:person:carol	2021
```

Primitives: **node type**, **node id**, **relation type**, and flat string/number/bool/null **properties** on both nodes and edges. There is no global header, no schema declaration in-band, no comment syntax in the library parsers, no explicit graph name (the VSCode parser invents a `graph.` line the libraries never emit), and no nesting. Adjacency is encoded purely relationally: an edge row names its two endpoints by `:type:id`; there are no adjacency lists, no inline nesting of neighbors, and no index. Multi-hop traversal by an LLM therefore requires joining edge rows to each other by hand.

Relative to JSON it drops: repeated keys (header once per block), all braces/brackets/colons/commas, quotes on values that don't need them, and the `"type"` field per node (hoisted into the block name). It keeps everything JSON keeps except nested structure.

`sampleX.ison` is a different, non-graph dialect (`object.customer`, `table.items`, `table.items_details` with `:1` back-references) belonging to the upstream core-ISON format from a separate repo (`ison.dev`). Neither the Python nor the TS library can produce it. It is present as documentation of the parent format, not of ISONGraph.

The serializer (`ison-graph/src/ison_graph/__init__.py:992` `to_ison`) is ~70 lines and self-contained. The **parser is not in this repo**: `from_ison` (line 1135) delegates to `ison_parser.loads`, from the external `ison-py` PyPI package. The repo therefore ships a writer whose reader lives elsewhere — the round-trip cannot be exercised from a clone alone.

### Tokenization behavior

The efficiency is **structural, not a tokenizer trick**, and I confirmed it is not tokenizer-specific: over the KG-100 corpus ISONGraph is 1,698 tokens under `o200k_base` and 1,705 under `cl100k_base` (68.6% vs 68.5% savings against the same-tokenizer JSON baseline). The savings come from three sources, in order of size: (1) not repeating field names per record, (2) not quoting or punctuating values, (3) hoisting `type` into the block header.

One genuinely non-obvious result fell out of my ablation: **the single-space delimiter is the right choice and tab is not.** Re-emitting byte-identical ISONGraph output with `\t` instead of `" "` costs 1,960 tokens instead of 1,698 — **+15%** — because BPE merges a leading space into the following token and does not merge a tab. This matters because the two shipped `.ison` samples and one of the two VSCode parsers are tab-delimited while every library serializer emits spaces. The repo does not appear to know this; the tab/space split is an accident, not a decision.

The `:type:id` reference syntax is the format's **largest remaining overhead**, and it is not ablated anywhere. Reserializing the same 7 datasets with bare integer endpoints (`1` instead of `:person:1`), holding everything else fixed, gives **1,280 tokens vs 1,650 for my typed-ref control** — the type prefixes cost ~22% of the whole document. BENCHMARK.md:122-125 justifies the syntax by comparing it to `{"type": "person", "id": 1}` ("7x more tokens") — but that construction appears in no format the benchmark actually tests. Its own JSON converters emit bare `"source":1`. The prefix earns its keep only when ids collide across node types; in all 7 benchmark datasets ids are globally unique, so on the corpus that produces the headline number it is pure cost.

### Round-tripping and lossiness

I ran the shipped Python serializer against adversarial values. It is lossier than the README's "Value Types" table (README:645-655) implies:

| Input (Python) | `to_ison()` emits | Problem |
|---|---|---|
| `[1, 2, 3]` | `[1, 2, 3]` **unquoted** | `str(list)` repr contains spaces → row splits into 3 fields. README advertises `[1,2,3]` as a value type; the writer cannot produce it. |
| `{"a": 1}` | `{'a': 1}` **unquoted** | Python repr, single quotes, embedded space → same corruption. |
| `"a\tb"` (string with tab) | `a<TAB>b` **unquoted** | Quoting rule (line 1077) checks `' ', '|', '"', '\n'` — not `\t`. Since real `.ison` files in this repo are tab-delimited, this is a live delimiter collision. |
| `"42"` (string) | `42` | Round-trips back as an integer. |
| `"true"` / `"null"` (strings) | `true` / `null` | Round-trip back as bool / None. |

Type identity is not preserved for strings that look like scalars, and containers are not representable at all. There is no in-band way to signal "this is a string" — nothing analogous to YAML's quoting-is-meaningful rule survives, because the writer only quotes on delimiters.

---

## Implementations

| Directory | Language | Core LOC | Modules | Tests |
|---|---|---|---|---|
| `ison-graph/` | **Python** (the reference; PyPI `ison-graph` 1.2.0) | 4,505 | core, query, schema, viz | 247 `def test_` |
| `ison-graph-ts/` | TypeScript | 4,606 | core, query, schema, viz | 167 `it()` |
| `ison-graph-js/` | JavaScript | 4,341 | core, query, schema, viz | 167 `it()` (near-verbatim port of the TS suite) |
| `ison-graph-rs/` | Rust | 5,768 | core, query, schema — **no viz** | 58 `#[test]` |
| `ison-graph-cpp/` | C++17, header-only | 4,211 | core, query, schema — **no viz** | 3 monolithic fns, 274 asserts |
| `ison-graph-cs/` | C# / .NET 8 | 5,135 | core, query, schema, viz | 367 `[Fact]` |

CI (`.github/workflows/ci.yml`) runs each language's own suite. **There is no conformance suite and no shared fixture corpus** — I grepped every test file for `social.ison`/`sampleX.ison` and found exactly one hit (`ison-graph-cs/tests/.../SerializationTests.cs:458`, which writes its own temp file of that name). Nothing cross-checks byte output between ports.

They do not agree. Three concrete divergences, all verified in source:

1. **Quoting trigger set.** Python (`__init__.py:1077`), TS (`index.ts:1123`) and JS (`index.js:918`) quote on `{space, |, ", \n, empty}`. Rust (`lib.rs:1080`) adds `\t`. C++ (`ison_graph.hpp:1026`) and C# (`Serialization.cs:27`) match Python's trigger set. All six carry a comment calling their own rule "the standardized rule (shared by all ISON language ports)."
2. **Escape set.** C++ and C# escape backslash (`\\`); Python, TS, JS and Rust do not. A property value containing both a space and a backslash serializes to two different strings depending on which port wrote it, and only one survives a parser that unescapes `\\`.
3. **Row ordering is nondeterministic in two ports.** Python (`dict`), C# (explicit `_nodeOrder` insertion lists) and JS/TS (`Map`) emit rows in insertion order. Rust stores `nodes: HashMap<String, HashMap<String, Node>>` and iterates `type_nodes.values()` directly (`lib.rs:1132`) — Rust's `HashMap` is randomly seeded per process, so **the same graph serializes to a different byte string on every run**. C++ uses `unordered_map` (`ison_graph.hpp:217`) — implementation-defined order. Both sort *type* names but not *rows*.

Two ports are also weaker in the type system: Rust (`lib.rs:100`) and C++ (`ison_graph.hpp:85`) declare properties as `HashMap<String,String>` / `map<string,string>`. They cannot hold an int, bool or null at all — everything is a string that happens to serialize the same way.

## Tooling

Two VS Code extensions, and they disagree with each other:

- **`vscode-isongraphviz/`** (140-line `parser.ts`): correct. `splitFields` handles space *and* tab delimiters plus quoted values with `\"`, `\n`, `\\` escapes. Ports the Python `compute_layout` Fruchterman-Reingold + seeded LCG for reproducible geometry, renders in a webview with a live-physics drag layer.
- **`vscode-isongraph/`** (266-line `parser/isonParser.ts`, D3-based editor): **cannot read the output of any ISONGraph library.** Line 52 gates row parsing on `line.includes('\t')` and line 53 splits on `'\t'` — space-delimited rows are silently skipped, producing an empty graph. Its committed sample (`samples/social.ison`, 58 tabs) is tab-delimited to match. Its `parseISONL` (line 190) expects `nodes.type id key=value` — a format no library emits; the real ISONL is `nodes.person|id name age|1 Alice 30`.

`CONTRIBUTING.md` says the VS Code extensions "live in separate repositories" and then lists both directories in its build table — a small sign that the repo was assembled rather than grown.

---

## Benchmark Audit

Two suites, both single-author, both dated ~7 months before the repo's only commit.

**What was measured.** `benchmark/KnowledgeGraph_Benchmark/benchmark_kg_100.py` (1,183 LOC) defines 7 hand-written datasets (6-13 nodes, 9-14 edges each — *tiny*) and 100 hand-written questions with hand-written expected answers, across 6 categories. For each dataset it renders 9-10 formats, counts tokens with `tiktoken`, then for every (question × format) pair sends one prompt to **DeepSeek Chat at temperature 0** and string-matches the reply. `benchmark/DataTraversal_Benchmark/benchmark_graph.py` (834 LOC) is the same machinery on 4 datasets / 50 questions / 5 formats. N is 100 and 50 single-shot generations per format; no repeats, no seeds, no CIs, one model.

**Token claims: verified.** I stubbed the missing `ison_parser` dependency, imported the repo's own converters, and re-ran the token pass. Every published number reproduces exactly:

| Dataset | Published ISONGraph | Mine | Published JSON | Mine |
|---|---|---|---|---|
| Social Network | 177 | 177 | 596 | 596 |
| Knowledge Graph | 285 | 285 | 832 | 832 |
| Movie Database | 335 | 335 | 959 | 959 |
| **KG-100 total** | **1,698** | **1,698** | **5,406** | **5,406** |
| **Traversal-50 total** | **639** | **639** | **2,039** | **2,039** |

Cypher (3,522), RDF/Turtle (4,166), GML (4,202), JSON-LD (8,191), GraphML (9,093) also reproduce to the digit. This half of the benchmark is honest and cheap to check. (ISON and TOON I could not reproduce — they need the external `ison-py` and `toon` packages.)

**Accuracy claims: not reproducible from anything committed.** `run_benchmark` builds a `results` list and *returns* it; nothing serializes it. The only persisted output is `benchmark_kg100_<ts>.log`, and `.gitignore` excludes `*.log`. So the 90.0% / 87.0% / 58.0% table exists solely as prose in `BENCHMARK.md`. Re-running costs 1,000 DeepSeek calls (~40 min by their estimate) and would produce different numbers anyway, since the datasets contain questions whose stated answers are contestable (KG-100 Q9 "How many hops from Jane to DataAPI?" expects 1 for a path the file itself describes as two edges — the doc calls this ambiguous *after* scoring it as a universal failure).

**The grader is the weakest link.** `validate_answer` (line 140) is substring matching without word boundaries:
- String answers pass if `expected_lower in response_lower`. Expected `"Eve"` matches a reply containing *"revenue"*, *"seven"*, *"level"* — and the same social-network dataset carries a `revenue` field.
- Boolean answers scan for `"no"` as a bare substring in the false-indicator list. Any reply containing *"not"*, *"none"*, *"Nolan"* trips it.
- List answers require only `all(item in response)` — a reply that names every node in the graph scores correct on every list question.
- Path answers claim ordering ("Path nodes found in order") but test unordered membership.
These biases are format-agnostic, so they inflate all ten formats roughly equally; they do not obviously manufacture ISONGraph's 3pp lead. But they mean the absolute numbers are not measuring what the table says.

**Baseline construction is asymmetric.** ISONGraph and ISON are produced by the author's own libraries; the other eight converters are ~40-line hand-rolls written for this file. Notably `to_isongraph` (line 281) wraps `add_edge` in a bare `except: pass`, so a rejected edge silently vanishes and reduces ISONGraph's token count — no other converter can lose data that way. RDF/Turtle's 58% is then blamed on RDF as a standard (BENCHMARK.md:134-141, "LLMs can't map `ex:person_1` back to Alice"), when it is a property of *this converter*: it renders edges as opaque-IRI triples with names stashed on separate subjects. A Turtle serialization using name-derived IRIs was never tried.

**The alternatives set omits the one family that competes.** Ten formats are tested; **none of them is a plain table.** No CSV, no TSV, no Markdown table, no adjacency list, no YAML. That is the entire mechanism ISONGraph relies on, so its headline claim was never contested. My check, on their datasets and tokenizer:

| Encoding | o200k tokens (KG-100) | vs ISONGraph |
|---|---|---|
| **ISONGraph (as shipped)** | **1,698** | — |
| Name-based adjacency lines (`Alice -follows-> Bob since=2020`) | 1,448 | **−14.7%** |
| Per-type TSV tables, bare ids | 1,395 | **−17.8%** |
| Same block structure, bare ids instead of `:type:id` | 1,280 | **−24.6%** |
| Mermaid `graph LR` | 2,025 | +19.3% |
| JSON Compact | 2,893 | +70.4% |

The last row of the "worth stealing" argument and the refutation of the superlative are the same measurement: ISONGraph's own design, minus the one feature it markets hardest, is a quarter smaller.

| Claim | Evidence | Assessment |
|---|---|---|
| 68.6% token savings vs JSON (KG-100) | Reproduced exactly by me from committed code | **Validated** |
| 68.7% savings vs JSON (Traversal-50) | Reproduced exactly | **Validated** |
| Savings are not a tokenizer artifact | 68.6% (o200k) vs 68.5% (cl100k), my run | **Validated** |
| "Most token-efficient way to represent graph data" | No tabular/CSV/adjacency baseline tested; three of my ad-hoc encodings beat it by 15-25% | **Refuted as stated** |
| 90.0% accuracy, best of 10 formats | Prose table only; results never serialized, log gitignored | **Unvalidated** |
| RDF/Turtle at 58% shows RDF is unsuitable for LLMs | Follows from one hand-written converter's IRI choice, not from RDF | **Questionable** |
| ISONGraph beats TOON by 42% tokens / 5pp accuracy | Token half plausible (I could not reproduce TOON — external dep); accuracy half unvalidated | **Partly unvalidated** |
| Multi-hop 80% vs 40-70% for alternatives | 8/10 vs 4-7/10 on a 10-question slice, single run, biased grader | **Statistically empty** (n=10) |
| Full feature parity across 6 languages | 3 distinct quoting rules; Rust/C++ lack viz and typed values; Rust/C++ row order nondeterministic | **Refuted** |
| Deterministic viz, bit-identical across languages | Plausible for Py/JS/TS/C# (seeded LCG + insertion-ordered maps); no cross-language test exists | **Unvalidated** |

---

## Memory-System Dimensions

Not a memory system. ISONGraph is an in-process data structure plus a text encoding; nothing here persists, ages, decides what to keep, or learns from use.

- **Storage & Schema**: in-memory hash maps only; `save()`/`load()` write a flat file. No database, no indexes beyond in/out adjacency, no concurrency story. `ISONGraphantic` (`schema.py`) is a fluent *validator* (types, min/max, enums, cardinality, acyclicity, no-orphans), run on demand — not a write-path gate.
- **Memory Types**: N/A. Node/edge types are user-defined labels with no semantics.
- **Write Path**: N/A. `add_node`/`add_edge` insert unconditionally; the only gate is exact `(rel_type, source, target)` uniqueness via `_edge_set`. No dedup, no salience, no redaction, no reconciliation.
- **Retrieval**: BFS/DFS traversal and an ISONQL filter language. No ranking, no scoring, no embeddings. The README's "Semantic Search (Embeddings)" section is explicitly labeled **planned** — the `ison-graph-embeddings` package is unreleased and appears nowhere in this repo.
- **Consolidation / Lifecycle**: none. No decay, no merge, no summarization, no archival, no temporal validity.
- **Benchmarks (LoCoMo / PERMA)**: N/A — never run, and the two suites here are the author's own hand-built corpora.

---

## Relevance to Somnigraph

### What ISONGraph does that Somnigraph doesn't

Somnigraph has **no graph serializer at all.** Edges in `memory_edges` are consumed only numerically — `src/memory/scoring.py:228` `personalized_pagerank` builds an adjacency dict for power iteration, `_expand_adjacency_legacy` (line 398) does novelty-scored BFS, and betweenness enters the reranker as a scalar feature. No code path renders the graph as text. Everything that reaches an LLM is prose:

- `src/memory/formatting.py:7` `format_memory_compact` — one memory per line, `[category] summary (themes)`.
- `src/memory/formatting.py:21` `format_memory_full` — three lines per recall hit.
- `scripts/sleep_nrem.py:116` `CLASSIFY_TEMPLATE` — one *pair* per block, restating five field labels twice per pair, batched to `BATCH_TOKEN_LIMIT`.
- `scripts/sleep_rem.py` taxonomy/summary/audit prompts — lists of memories, again per-item labeled.

So ISONGraph occupies a slot Somnigraph has left empty. The gap is real but narrow: Somnigraph's per-item cost is dominated by free-text `content`, not by field labels, so a header-once table saves maybe 10-20 tokens on a 200-600 token NREM pair block.

### What Somnigraph does better / where the comparison doesn't apply

Nearly everything about persistence, ranking and lifecycle — none of which ISONGraph attempts. On the axis they share (graph representation), Somnigraph's typed-edge model (`supports`/`contradicts`/`evolves`/`revision`/`derivation`, with confidence and provenance) is richer than ISONGraph's untyped `rel_type` string, and its edges are *derived* by LLM classification during NREM rather than hand-authored. Somnigraph's benchmark discipline is also in a different category: paired bootstrap CIs, pre-registered hypotheses, judged-vs-raw credit brackets, 170 registered masks in WP6. ISONGraph reports point estimates from n=1 with a substring grader and does not commit the results.

---

## Worth Stealing (ranked)

The honest total here is **one small measurement and one small format habit** — not an architecture. Somnigraph does not currently serialize graphs into prompts, so the format itself has no consumer to steal it for.

### 1. The space-vs-tab delimiter measurement (Low effort, ~15 minutes)
**What**: Under `o200k_base`, a leading space merges into the following BPE token and a tab does not. Byte-identical tabular content costs **+15%** in tabs (1,960 vs 1,698 on the KG-100 corpus, my measurement).
**Why**: Somnigraph's own budgeting code (`count_tokens` gating in `tools.py:118`, `sleep_nrem.py:340` batch packing) is sensitive to per-line overhead, and any future table-shaped prompt would face this exact choice. It is the one non-obvious thing this repo demonstrates — and it does so accidentally, since its samples are tabs and its serializers are spaces.
**How**: no code change; a two-line note in `docs/architecture.md` next to the token-budget discussion. Worth banking because it is cheap and durable.

### 2. Header-once framing for the NREM pair batch (Low effort, likely not worth it)
**What**: `CLASSIFY_TEMPLATE` repeats `stored {date}, source: {source}` / `[{category}]` / `Themes:` for both memories in every pair. A per-batch header (`# pair_index | side | date | source | category | themes | content`) with one row per memory would state those labels once per batch instead of 2N times.
**Why**: `sleep_nrem.py` runs thousands of pairs per pass and each batch pays the label tax repeatedly.
**How**: rewrite `format_pair`/`build_batches` in `scripts/sleep_nrem.py`. **Caveat that probably kills it**: memory `content` is multi-line free prose, which is exactly what a whitespace-delimited row cannot hold — you'd need quoting/escaping, which reintroduces the tokens you saved and adds the lossiness class documented above. The saving is single-digit percent on a prompt whose cost is content, and it perturbs a prompt that H10/H12 results are calibrated against. **Recommend not doing this** unless a future sleep stage needs to put *structure* (edges, cluster membership) rather than *prose* into a prompt.

### 3. Nothing else
No retrieval idea, no lifecycle idea, no evaluation idea. The ISONQL query language, the schema validator, the viz layer and the six-way port are all orthogonal to a single-user SQLite memory server.

---

## Not Useful For Us

- **The format as a storage layer.** Somnigraph stores in SQLite with FTS5 and sqlite-vec; a flat text file with no indexes, no transactions and a type-lossy round trip is strictly worse.
- **The library.** In-memory hash maps with BFS duplicate what `scoring.py` already does over SQL, without the PPR weighting or the reranker integration.
- **ISONQL / ISONGraphantic.** Somnigraph has no user-facing graph query surface and doesn't want one; schema validation is handled by the `remember()` tool contract.
- **The benchmark harness.** Substring grading, n=1, single model, no committed artifacts — the opposite of the discipline in `experiments/` and `docs/benchmarks.md`. Adopting anything from it would be a regression.

---

## Connections

- [`token-savior.md`](token-savior.md) — the nearest neighbor by *goal*. token-savior's headline is "−77% to −80% active tokens," achieved by compacting **tool output** rather than by inventing a serialization format. Both are token-efficiency plays; token-savior's is measured on a task benchmark (tsbench) with a downstream signal, ISONGraph's on a hand-built corpus with a substring grader. Interesting contrast in where the two authors chose to spend the compression budget.
- [`graphs-rag-at-scale.md`](graphs-rag-at-scale.md) — **strong convergence.** Tadayon & Gupta (arXiv:2603.22340) independently find that a Labeled Property Graph with Cypher beats an RDF-triple pipeline for LLM-mediated retrieval over semi-structured data. ISONGraph reaches the same LPG-over-RDF ordering (90% vs 58%) on a much weaker instrument. Two independent arrivals at "LPG > RDF for LLMs" is worth more than either alone — though the paper attributes it to query expressiveness while ISONGraph attributes it to identifier legibility.
- [`graphrag.md`](graphrag.md) — GraphRAG builds a graph from text via LLM extraction, then serializes *community reports* (prose summaries) rather than graph structure into context. That is the dominant design and the reason ISONGraph's niche is thin: production graph-RAG systems mostly avoid putting raw graph structure in the prompt at all.
- [`graphiti.md`](graphiti.md), [`kumiho.md`](kumiho.md), [`cortexgraph.md`](cortexgraph.md) — graph-native memory systems whose graphs live in Neo4j/FalkorDB and are traversed by query, never serialized wholesale into a window. Kumiho in particular pairs its property graph with formal AGM belief-revision semantics; ISONGraph has no semantics above the byte level.
- [`hipporag.md`](hipporag.md) — PPR over a knowledge graph as the retrieval primitive, the direct ancestor of Somnigraph's `personalized_pagerank`. Same reason ISONGraph's encoding is unneeded: the graph is *scored*, not *shown*.
- [`graph-aware-late-chunking.md`](graph-aware-late-chunking.md) — graph structure used to shape chunk selection, again without ever rendering the graph as prompt text.

The convergent finding across this cluster: **every serious graph-memory system consumes its graph numerically and emits prose.** ISONGraph is the only one betting that the LLM should read the adjacency table directly, and it validates that bet on 6-to-13-node graphs where any encoding fits.

---

## Summary Assessment

The core contribution is real but small: a clean, minimal, header-once tabular encoding for property graphs, ported six ways, with an honest and fully reproducible token-count measurement attached. I re-derived every published token number from the committed code and they are exact — that half of the benchmark is better than most repos manage. The format is also well-shaped for its purpose: block-per-type, single-space delimiter, no punctuation, human-legible names instead of opaque identifiers.

Everything above that floor is overclaimed. "The most token-efficient way to represent graph data" is refuted by the first three alternatives I tried in twenty minutes — plain per-type TSV is 18% smaller, and the format's own design minus its `:type:id` reference syntax is 25% smaller. The comparison set was chosen from serialization *standards* (JSON, GraphML, RDF, GML, JSON-LD, Cypher) and omits the entire tabular family that the design actually competes with, so the superlative was never tested. The accuracy claims — the ones that would make this more than a formatting choice — rest on 100 single-shot DeepSeek generations graded by unanchored substring matching, with no committed artifact and a `.gitignore` rule that guarantees the log can never be committed. The "multi-language format library" framing does not survive reading the six serializers side by side: three incompatible quoting rules, two ports that cannot represent a non-string value, and two ports (Rust, C++) whose row order is nondeterministic, all under source comments asserting a shared standard. For a format library, non-byte-identical output across implementations is the failure mode, and there is no conformance suite that would have caught it.

For Somnigraph the takeaway is one measurement, not an adoption: **space delimiters beat tab delimiters by ~15% under o200k for tabular prompt content.** Bank it and move on. The deeper signal is negative and worth stating — this cluster of analyses now contains six graph-memory systems, and none of the serious ones put raw graph structure into a context window. Somnigraph's choice to consume edges numerically (PPR, betweenness, novelty-scored adjacency) and emit prose is the majority position, and ISONGraph's evidence that the minority position is better does not survive scrutiny at n=1 on 10-node graphs. If a future sleep stage ever needs to *show* an LLM a subgraph rather than score one, revisit the header-once idea — but design the delimiter and reference syntax from the token measurement, not from this repo's defaults.
