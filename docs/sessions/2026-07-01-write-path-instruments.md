# 2026-07-01 — Write-path measurement instruments

Session 2 of the stewardship arc. Ships the write-path *measurement* instruments named in STEWARDSHIP Priority 2, step 1. Governing principle: **measure before you threshold** — this session instruments, it does not gate. Nothing here changes retrieval behavior or what gets stored, except that secret/PII redaction coverage was extended.

Branch: `write-path/instruments`, based on `main` after Session 1's reranker-arc docs (`c64f6ab`) were merged in (fast-forward, at Alexis's direction — the orchestrator had expected Session 1 already on main; it was still on `docs/reranker-arc`).

## What shipped

### Task 1 — Shadow-mode near-dup logging (`src/memory/tools.py`)

`impl_remember()` now logs exactly one `write_shadow` event to `memory_events` per write attempt that reaches the embedding step. Each event records:

- `outcome`: `inserted` / `superseded` / `dedup_rejected` (the three real code paths — richer than the "inserted/deduped/rejected" the prompt sketched; the extra distinction between supersede and reject is free and useful for threshold-setting).
- `new_id`: the newly stored memory's id, or `null` when the write was rejected.
- `neighbors`: up to the 3 nearest **same-category** neighbors, each `{id, distance, sim}` (cosine distance and `sim = 1 - distance`).
- `similarity_score` column: the nearest neighbor's similarity, so the histogram reader needs no JSON parse for the common case.

It **reuses the existing dedup KNN pass** — no second vector search. The top-3 are filtered from the dedup candidate set to same-category rows (the dedup KNN is a global k-NN capped at the same-category count, so filtering makes the logged distribution match the same-category decision the gate actually makes). The event uses a `_write` sentinel `memory_id`, mirroring the existing `_recall` sentinel for query-scoped events, because `memory_events.memory_id` is `NOT NULL` and a rejected write has no new id to attach.

First-of-category writes (no same-category memory exists yet) log an empty neighbor list and null `similarity_score` — structurally un-dedupable, and the histogram reader counts them in a separate "no neighbor" row.

Validation-error early returns (invalid category/status) fire *before* the embedding and log nothing — there is no near-dup measurement possible without an embedding. So "one event per call" means one per call that reaches the write attempt.

**Stretch: `scripts/write_shadow_histogram.py`** buckets the accumulated similarities (0.05 buckets) split by outcome, so a future session reads the distribution without raw SQL.

#### Isolation verification (required by the task)

Grepped every reader of `memory_events` across `src/` and `scripts/`. **Result: every behavioral consumer filters by a specific `event_type`** — reranker (`event_type IN ('retrieved','feedback')`, `= 'feedback'`, `= 'retrieved'`), scoring's Hebbian/Beta reads (`= 'feedback'`, `= 'retrieved'`), `session`/burstiness, sleep pipeline (`recall_miss`, `recall_miss_repaired`, `retrieved`), the GT/tuning/probe scripts, and `db.py`'s migration read. `write_shadow` matches none of them, so it is invisible to all scoring, retrieval, GT, and sleep logic. The **only** unfiltered reads are `stats.py`'s two display-only queries (`SELECT count(*)` and `GROUP BY event_type`), which simply surface a new `write_shadow: N` row — desirable visibility, not behavior. No consumer needed a filter fix.

### Task 2 — Secret/PII redaction audit + extension (`src/memory/privacy.py`)

**Audit result: STEWARDSHIP P2's "remember() will embed a leaked API key today" was factually wrong.** `_strip_sensitive()` runs at the top of `impl_remember` (and `impl_update`) *before* `build_enriched_text` / `embed_text`, so a redacted secret never enters the content stored, the summary, the FTS index, or the embedding vector. Corrected that sentence in STEWARDSHIP P2 (honest accounting, Priority 1, outranks last session's framing).

Extended pattern coverage to match Memora's set, keeping the existing `[REDACTED_*]` marker convention:

- PEM private-key blocks (`-----BEGIN … PRIVATE KEY----- … -----END`, DOTALL, redacted first so nothing inside leaks past a later single-line pattern)
- Bearer tokens (`Authorization: Bearer …`)
- AWS access key IDs (`AKIA…`), Google API keys (`AIza…`)
- Credit-card numbers anchored to major-network prefixes (Visa/MC/Amex/Discover) to avoid clobbering ordinary long digit strings
- More connection-string schemes (mongodb/mongodb+srv, redis, amqp) added to the existing postgres/mysql/mssql/snowflake set

Existing patterns (OpenAI/generic `sk|pk|api` keys — also covers `sk-ant-…` and `sk-proj-…`, JWTs, GitHub tokens, Slack tokens, `password=…`) were kept unchanged.

### Task 3 — memory_meta masquerading-defaults: STOPPED, code untouched

The backlog item was to NaN-encode the `memory_meta`-missing default block (`category=1, priority=5, …`) in `reranker.py` and `train_reranker.py` **if** it is genuinely dead in production. The task said: if it *can* fire in production, STOP and leave the code untouched.

**It can fire.** Two independent triggers:

1. **Pending-status candidates.** `_insert_memory` writes a vector + FTS entry for `status='pending'` memories. `impl_recall`'s vector/FTS searches have no status filter, and — critically — while `all_ids` is filtered to active, the *unfiltered* `vec_ranked`/`fts_ranked` dicts are what get passed to `rerank()`. `rerank()` builds `candidate_ids` from those unfiltered dicts, so a pending memory near the query enters the candidate pool. `_load_memory_meta()` loads only `status='active'`, so `memory_meta.get(mid)` is `None` → the masquerading-defaults branch executes.
2. **Superseded candidates.** The `remember()` supersede path sets the old memory's `status='deleted'` but, unlike `impl_forget`, does **not** delete its vector/FTS/rowid rows. Those linger in the search tables and can also surface as candidates → same `None` → same branch.

The effect is **masked** in production because `recall()` filters final results to `status='active'` (line ~690), so these non-active candidates' scores are computed and then discarded — they never reorder active results. So the earlier "dead-code in practice" claim was defensible only in the "no observable effect" sense; the branch genuinely executes. Because it runs under a deployed model, NaN-encoding it now would change live feature computation for those candidates — exactly what the guardrail forbids. Left `reranker.py` and `train_reranker.py` untouched. Corrected the architecture.md § Sentinel-encoded missing values note to state this precisely.

**Clean fix for a future session:** filter `candidate_ids` to active memories at the top of `rerank()` (or filter the dicts in `impl_recall` before the call). That makes the block genuinely dead *and* stops wasting compute scoring discarded pending/superseded candidates — after which NaN-encoding is safe. It changes the live candidate set, so it wants its own session and possibly a retrain. (Adjacent latent bug worth fixing there: the supersede path should clean vec/FTS/rowid like `forget()` does, so deleted memories stop lingering in the search index.)

## Verification

No test suite exists. Verified against a throwaway scratch DB (`SOMNIGRAPH_DATA_DIR` = temp dir). fastembed is not installed on this machine (it's Alexis's work-machine backend), so the check ran on the `openai` backend with the key loaded from the prod key file into a *fresh* scratch dir — never touching the prod DATA_DIR. ~7 embedding API calls total.

Drove `impl_remember` through every path and asserted programmatically (all PASS):

- **normal write** (first of category) → one `write_shadow`, `outcome=inserted`, empty neighbors, `_write` sentinel.
- **unrelated second write** → `inserted` with ≥1 captured neighbor carrying id/distance/sim and a populated `similarity_score`.
- **near-duplicate, higher priority** → `superseded`, records superseded id + new id.
- **near-duplicate, not higher priority** → `dedup_rejected`, `new_id` null, records matched id.
- **secret-bearing write** (fake `sk-proj-…`, `AKIA…`, Bearer token, credit card, PEM block) → none of the five markers survive into content, summary, or embedded FTS text; content shows `[REDACTED_*]` markers.
- **recall** still returns results on the scratch DB.

Also ran a full-package import + `py_compile` smoke test via `uv run` (production loads `src/memory/` through a directory junction, so the tree must import cleanly at all times) — clean. `write_shadow_histogram.py` ran against the scratch DB and rendered the bucketed table correctly.

## Files touched

- `src/memory/tools.py` — shadow-neighbor capture + `_log_write_shadow` helper + three emit points (CRLF).
- `src/memory/privacy.py` — extended `_PRIVACY_PATTERNS` (CRLF).
- `scripts/write_shadow_histogram.py` — new histogram reader (LF).
- `docs/architecture.md` — new § Shadow-mode write instrumentation; corrected the Sentinel-encoded-missing-values dead-code note (CRLF).
- `STEWARDSHIP.md` — corrected P2 redaction sentence; marked P2 step 1 shipped (LF).
- `docs/sessions/2026-07-01-write-path-instruments.md` — this file (LF).

`reranker.py` and `train_reranker.py` deliberately **not** touched (Task 3).

## Reversibility

All reversible. Task 1 is additive logging behind a new event type isolated from every consumer; deleting the `write_shadow` rows and reverting `tools.py` fully removes it. Task 2 is a pattern-list extension; over-redaction (false positives) is the only risk — the credit-card and bearer patterns are the most likely to over-match, mitigated by prefix/keyword anchoring. Task 3 changed no code.

## Surprises

- The redaction "gap" the survey and last session's STEWARDSHIP treated as open was already closed for the write path — redaction has run pre-embedding all along. The real gap is coverage breadth, not call-order. Good reminder to read the code before repeating a doc's claim.
- Task 3's "dead code" wasn't dead. The pending + superseded candidates reaching the reranker is a real (if effect-masked) execution path, and the supersede-path vec/FTS leak is a latent index-hygiene bug hiding behind it.

## What's next

- **Step 2 blocker:** shadow logging needs weeks of live writes to accumulate a real production near-dup distribution before Write Guard thresholds can be set from it (not guessed). Read `write_shadow_histogram.py` output before designing the gate.
- Future session: filter reranker candidates to active + clean vec/FTS on supersede, then NaN-encode the (now genuinely dead) masquerading-defaults block. Own session; possible retrain.
