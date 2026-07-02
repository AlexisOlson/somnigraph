# 2026-07-01 — Reranker restore (arc step 2.5): index hygiene + anti-recurrence; retrain HARD-STOPPED

**Branch:** `fix/reranker-restore` (off `main` @ 5900abd). Committed, not merged, not pushed.

Arc step 2.5 set out to restore the production reranker (on formula fallback since
2026-04-07) and fix index hygiene. The hygiene, NaN-encoding, and anti-recurrence
work shipped and is verified. **The retrain hard-stopped**: the exact V5+3b recipe is
unreproducible from surviving artifacts, and per the task's explicit instruction I did
not improvise a substitute and call it V5+3b. The model restore is deferred pending a
recipe decision.

## Task 0 — embedding backend resolved (a standing confusion, now closed)

The background premise ("production DB is fastembed 384d") is the **work-machine**
picture. On **this (home) machine**:

- `~/.claude/data/memory.db` → `memory_vec` is `float[1536]` = **OpenAI**.
- The MCP server (`~/.claude/settings.json`) launches via `uv run src/memory_server.py`
  with only `SOMNIGRAPH_DATA_DIR` set → `SOMNIGRAPH_EMBEDDING_BACKEND` defaults to
  `openai`, matching the DB. `~/.claude/data/openai_api_key` is present.
- `fastembed` is genuinely absent from the repo venv here — and that is **correct**.
  Installing it would flip the backend to 384d and `db.py`'s dim guard would hard-fail.

Per Alexis: fastembed is the work-computer config (local ONNX, no API key); home is
OpenAI. `SOMNIGRAPH_EMBEDDING_BACKEND` is the intended configurability knob, so
`migration-notes.md`'s "production runs on fastembed" is accurate for the work machine
and needs no correction. **The retrain must use `openai` to match the 1536d DB**; the
key file travels with the scratch copy.

## Task 1 — index hygiene (SHIPPED, commit 87189ab)

The `remember()` supersede path flipped `status='deleted'` without dropping the memory's
vec/FTS/rowid rows (unlike `forget()`), so superseded rows lingered in the search tables
and re-entered the reranker candidate pool as `status='deleted'` phantoms. `recall()`
masks the effect by filtering final results to active, but the missing-meta branch still
fired and compute was wasted.

- **1a** — extracted `_drop_search_rows(db, memory_id)`; called from the supersede path
  and refactored `forget()` to share it (single source of truth).
- **1b** — `db.py` `_init_schema` gained an idempotent startup backfill that prunes
  rowid_map/vec/FTS rows for deleted/missing memories (mirrors `consolidate()`'s orphan
  prune). Self-heals on the first post-merge start; no-op on a clean store.
- **1c** — `rerank()` now filters the candidate pool to active before scoring, via
  membership in the active-only `memory_meta` precompute (no extra query, no
  SQL-variable limit). Makes the missing-meta branch truly dead for live scoring.

**Consumer check for 1c:** `probe_recall.py` goes through `impl_recall` (covered);
`audit_reranker_pathology.py`/`diagnose_pathology.py` call `rerank()` (covered);
`train_reranker.py` does its own extraction against the DB (covered by the backfill).
The dedup KNN in `impl_remember` (checks active+pending) is untouched.

**Lingering-row count on the live DB: 0.** `consolidate()` had already pruned the 9
existing deleted memories. The backfill is defense-in-depth for the supersede bug going
forward, not a cleanup of current damage.

**Verified** on a scratch copy of the live DB: a simulated lingering superseded row is
pruned by the migration (rowid_map/vec/FTS all cleared), `memory_meta` stays active-only
(1162 keys after flipping one of 1163), and the candidate filter keeps only the active id.

## Task 2 — NaN-encode the memory_meta-missing block (SHIPPED, commit 9841a26)

The missing-memory branch fabricated a real-looking memory (category=1, priority=5,
age_days=0, token_count=200, edge_count=0, theme_count=0, confidence=0.5,
diversity=0.5, decay=default). Per the codified missing-encoding policy these masquerade
as real measurements; NaN-encoded them in **both** `src/memory/reranker.py` and
`scripts/train_reranker.py` (feature parity is a hard invariant). The memory-present path
is untouched, including the legitimate 0.0 for present memories with no usable query
terms. After Task 1c this branch is dead in live scoring and dead in training after the
backfill, so this changes no scored candidate today — it is policy compliance + parity.

## Task 4 — anti-recurrence (SHIPPED, commit d26f38e)

The silent fallback was invisible because graceful fallback logged nothing prominent and
no tool reported the live scorer. Two small changes (~24 lines):

- `reranker.py`: the no-model path logs a prominent `WARNING` ("RERANKER DISABLED ...
  FORMULA FALLBACK") instead of an info line; new `scorer_status()` returns a one-line
  identity. Logged string kept ASCII (Windows cp1252 would swallow an em-dash).
- `stats.py`: `memory_stats()` gains a "Retrieval Scorer" section reporting
  `scorer_status()` — the state is now checkable on demand.

Verified against scratch (no model): the warning fires and `memory_stats()` reports
`formula fallback (no model at ...)`.

## Task 3 — retrain (HARD STOP, no model produced)

**The exact V5+3b recipe is unreproducible.** V5+3b (`docs/sessions/2026-05-09-v5-3-*`)
was: `--pinned-boost 5.0`, 31 features, on a **cleaned 1885-query GT** built by re-emitting
after a 200-query real-recall adversarial probe, trained with a **sample-weights sidecar**
(pin-cap + per-mode holdout mechanics from v5-1/v5-2). Recipe inputs on disk:

- `gt_calibrated.json` — **present but wrong**: 1032 queries, dated Mar 20 (the
  26-feature production-era GT), not the V5+3b 1885-query GT.
- **Sample-weights sidecar** (`--sample-weights`) — **absent**. All tuning_studies
  artifacts are March-dated; no V5-era sidecar exists.
- **200 V5+3 probe events** — **absent** from the live `memory_events` (the May V5 work
  ran on a scratch DATA_DIR that is gone; live shows 1 superseded event, not the probe
  injection).
- `reranker_model.pkl` / `reranker_features.pkl` — present but the **stale March
  26-feature** model/cache. Converting the pkl to `.txt` would deploy a
  feature-misaligned model (26 vs the live extractor's 31) — explicitly not done.

Live signal that *does* exist (for a future fresh GT): 45,493 feedback rows with utility,
46,529 retrieved events, 3,328 distinct queries, Feb 20 → Jul 2.

Per Task 3's HARD STOP ("if any recipe input is missing or the recipe is ambiguous, stop
and report — do not improvise a different recipe and call it V5+3b"), no retrain was run.
The restore is deferred to a recipe-decision follow-up. Options surfaced to Alexis:
(1) defer retrain, ship the rest [chosen by default when no response]; (2) restore
baseline on the surviving March 1032-query GT, labeled honestly as *not* V5+3b (caveat:
March-era GT vs the grown current corpus); (3) fresh GT from the 45k live feedback rows
via the LLM-judge pipeline (best-calibrated, biggest scope — its own session).

## Post-merge deployment steps (for whenever a model IS trained)

The reranker loader (`constants.py:21-22`) expects two files in
`~/.claude/data/tuning_studies/`:

1. Copy `<scratch>/tuning_studies/reranker_model.txt` and `reranker_features.json` →
   `~/.claude/data/tuning_studies/` (the training run writes both via
   `train_reranker._export_for_mcp()`, called unconditionally after training).
2. Restart the MCP server (production loads `src/memory/` from this working tree via the
   directory junction, so the merged branch code is already live; only the model files
   are missing).
3. Confirm `memory_stats()` reports `reranker: 31 features, loaded from ...` (not
   `formula fallback`).

No writes to the live `~/.claude/data` happened this session. The scratch copy is under
the session scratchpad and is test-contaminated (one memory flipped to deleted for the
hygiene test) — do not reuse it for training; make a fresh copy.

## Surprises

1. **The DB is OpenAI, not fastembed.** The task's central environment premise was the
   work-machine picture; reading the actual `memory_vec` DDL (1536d) closed it in one query.
2. **Zero lingering rows on the live DB.** The supersede bug is real in code, but
   `consolidate()`'s orphan-prune had already cleaned the 9 deleted memories — so the
   observable damage was nil and the fix is forward-looking.
3. **The retrain is the blocked task, not the code.** The hygiene + anti-recurrence were
   straightforward; the actual model restore is gated on a GT that no longer exists.

## Caveats

- 1c couples "active" to `memory_meta` membership. This holds because `_load_memory_meta`
  is built from `status='active'` and `invalidate_cache()` fires on remember/forget. If a
  future change makes `_load_memory_meta` include non-active rows, the filter would leak —
  noted here as the coupling to watch.
- The other `status='deleted'`-without-cleanup paths (`consolidate()`'s near-dup merge,
  `review_pending` discard of indexed pending memories) still exist, but are covered by
  the same startup backfill + `consolidate()` orphan-prune. Left scoped; worth a follow-up
  to route them through `_drop_search_rows` too.
- NaN-encoding is inert until a model is trained under it; the next retrain must extract
  features with this branch present (both files already match).

## Reversibility

All three commits are additive and independent. `git revert` any one cleanly. No live data
was modified. Production remains on formula fallback exactly as before this session; the
only immediate live effect after merge is the startup backfill (a no-op on the currently
clean store) and the louder fallback warning.

## Files touched

- `src/memory/tools.py` — `_drop_search_rows` helper; supersede-path cleanup; `forget()` refactor.
- `src/memory/db.py` — startup index-hygiene backfill migration.
- `src/memory/reranker.py` — active candidate-pool filter; NaN-encoded missing-meta block; `scorer_status()`; loud fallback warning.
- `scripts/train_reranker.py` — NaN-encoded missing-meta block (parity).
- `src/memory/stats.py` — "Retrieval Scorer" section in `memory_stats()`.
- Docs (this task): `architecture.md`, `experiments.md`, `STEWARDSHIP.md`, this session file.

## What's next

- **Decide the retrain recipe** (defer / March-baseline / fresh-live-GT) and run it in a
  scoped session, then deploy per the steps above.
- Optional follow-up: route `consolidate()` merge + `review_pending` discard through
  `_drop_search_rows` for immediate (not just backfilled) cleanup.
