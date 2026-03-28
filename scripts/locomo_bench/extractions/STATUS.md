# Graph Extraction Status

*2026-03-24*

## What's done

**v6 extraction complete for all 10 conversations.** Single-pass Opus
call per conversation, producing claims, thin entities, and segments
(with cross-speaker evaluations). Files: `conv{0-9}_v6.json`.

| Conv | Entities | Claims | Segments | Cross-speaker evals | Sessions |
|------|----------|--------|----------|-------------------|----------|
| 0 | 17 | 76 | 40 | 36 | 19 |
| 1 | 9 | 54 | 41 | 15 | 19 |
| 2 | 20 | 106 | 51 | 21 | 32 |
| 3 | 21 | 72 | 33 | 8 | 29 |
| 4 | 36 | 101 | 49 | 15 | 29 |
| 5 | 13 | 87 | 42 | 9 | 28 |
| 6 | 39 | 115 | 57 | 16 | 31 |
| 7 | 31 | 83 | 47 | 14 | 30 |
| 8 | 13 | 69 | 50 | 17 | 25 |
| 9 | 22 | 67 | 30 | 13 | 30 |
| **Total** | **221** | **830** | **440** | **164** | **272** |

**Quality notes from conv 0 validation:**
- Outdoor/nature vocabulary: 10+ claims with camping/hiking/nature terms
- Cross-speaker evaluations: 36 detected (vs 1 opinion_of claim in prior v5 run)
- Session coverage: all 19 sessions have claims and segments
- Mild thinning in later sessions but no catastrophic fatigue

**Known issues to check:**
- Conv 3: only 8 cross-speaker evaluations (lowest). May be underextracted.
- Conv 9: 30 segments for 30 sessions (1 per session). Likely undersegmented.
- Conv 4/6: large entity counts (36, 39). Check for duplicates.

## What comes next (in order)

### 1. Validate extractions across all 10 conversations

Quick spot-checks on conversations with hard failures:
- Conv 3: Is "Jo" captured as Joanna alias? Are turtle care claims present?
- Conv 4: Is MinaLima grounded (NYC store)? Are leadership/seminar claims present?
- Conv 5: Are cooking + dog adoption claims present for the indoor activity failure?
- Conv 7: Is Mount Talkeetna grounded to Alaska?
- Conv 6: Are "lonely" inference and "elementary school → studied together" capturable?

### 2. Script pipeline (zero LLM cost)

Build these from the extraction JSON:

**a. Entity facets** — Group claims by (entity, topic). Clustering options:
- Hand-built taxonomy (~20 categories) + keyword matching (cheapest)
- TF-IDF on claim object fields per entity (better)
- Embedding clustering on claim retrieval_text (best, uses existing pipeline)
Threshold: 3+ claims per facet. Concatenate retrieval_text into facet node.

**b. Relationship nodes** — Group cross_speaker_evaluations by (observer,
subject). Concatenate into directed relationship nodes with
traits_attributed array. Template likely_questions.

**c. Entity co-reference edges** — Turns mentioning the same entity
get edges. Use entity aliases + evidence_turn_ids from claims.

**d. EXTRACTED_FROM edges** — Every synthetic node links to its source
turns via evidence_turn_ids.

**e. Distribute bridging_terms to turns** — Map each segment's
bridging_terms back to constituent turns. This enables ablation step 0
(vocabulary augmentation without synthetic nodes).

**f. Entity dedup** — Fuzzy match entity names/aliases across the
extraction. Flag probable duplicates.

**g. Opinion_of validation** — For each speaker pair, count
cross_speaker_evaluations. Flag conversations with <3 per direction.

### 3. Ingest into benchmark DBs

The benchmark creates a fresh SQLite DB per conversation via
`ingest.py`. Need to:
- Insert synthetic nodes (claims, segments, facets, relationships) as
  memories in the DB with appropriate fields
- Generate embeddings for all synthetic nodes (retrieval_text field)
- Insert edges into memory_edges table
- Track node_type metadata for each inserted memory

### 4. Run retrieval eval

Use separate retrieval pools (turns vs synthetic), merged via RRF.
No reranker changes yet — synthetic nodes scored by raw vector+BM25.

`eval_retrieval.py` against augmented DBs. Measure:
- R@10 on 18 hard multi-hop failures
- R@10 on 67 successful multi-hop questions (regression)
- R@10 on all 1,531 non-adversarial questions
- Per-question: which node type was retrieved for each recovered failure

### 5. Ablation (per the plan in extraction_prompt.md)

0. Turns + bridging_terms only (no synthetic nodes)
1. Turns only (baseline)
2. Turns + claims
3. Turns + claims + entity facets
4. Turns + claims + facets + relationships
5. Turns + claims + facets + relationships + segments
6. Full (+ bridging_terms, latent_concepts in BM25)

Stopping criterion: node type earns its place if ≥2 hard failures
recovered without ≥3 regressions.

## Key files

- `scripts/locomo_bench/extraction_prompt.md` — v6 prompt (final)
- `scripts/locomo_bench/extractions/conv{0-9}_v6.json` — raw extractions
- `scripts/locomo_bench/extractions/archive_v1/` — prior iteration outputs
- `docs/multihop-failure-analysis.md` — the 18 hard failures we're targeting
- `docs/extraction-consultation.md` — original problem statement
- `docs/extraction-consultation-v3.md` — v3 review round
- `docs/extraction-consultation-v4.md` — v4 review round
- `docs/extraction-consultation-v5.md` — v5 review round

## Design decisions log

- **v1**: 4-level hierarchy (segments, sessions, conversation, entities). One pass.
- **v2**: Added relationships, HyPE, transformation protocol. Two passes.
- **v3**: Added claims, entity facets, session deltas. Three passes.
- **v4**: Per-session Pass 1, thin entities, structural enforcement. Three passes.
- **v5**: Collapsed back to single pass (cost constraint). Relationships as LLM output.
- **v6 (current)**: Single pass, scoped tasks. Cross-speaker evaluations extracted
  locally in segments (not globally in relationships). Relationships and facets
  built by script. Entities and claims use global scope; segments use local scope.

Key insight from v5→v6: moving evaluative extraction from a global task
(relationship synthesis across full conversation) to a local task
(detect within each segment) dramatically improved capture rate
(1 opinion_of → 36 cross-speaker evaluations on conv 0).
