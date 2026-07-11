# Fork runbook — accumulation-regime sleep fork (~2026-07-15)

Frozen 2026-07-11 per the pre-fork review ([`findings-prefork-review.md`](findings-prefork-review.md)).
All decisions pre-registered; do not tune mid-flight. Every long/LLM step runs in
Alexis's terminal. **The live store is never written by anything below** — the one
read of it is the quiesced BASE backup in Phase 0.

## Arms and cells

```
BASE  = quiesced backup of the live store (2 weeks unprocessed accumulation)
A     = fork of BASE, untouched (frozen arm)
B     = fork of BASE, slept:  sleep.py --deep --probes 0   → checkpoint = B-mid
                              sleep_consolidate.py -v      → final      = B-full
```

Scored cells (6): `{A, B-mid, B-full} × V6` then `{A×R_A, B-mid×R_B, B-full×R_B}`
where R_A/R_B are same-recipe retrains on the respective arm's store, one frozen GT.

Pre-registered contrasts (analyze_sleep_fork.py, `--config reranker`, paired bootstrap):

1. `B-full(V6) − A(V6)` — naive deploy. Bench prediction: judged-null.
2. `B-full(R_B) − A(R_A)` — sleep under sleep-aware scoring. Bench: +0.091✱ NDCG.
3. Interaction = (2) − (1). Bench: +0.072✱.
4. `A(R_A) − A(V6)` — pure recalibration dividend, live analogue of N4's ~95% claim.
5. `B-full(V6) − B-mid(V6)` and `B-full(R_B) − B-mid(R_B)` — consolidate's marginal
   (first production-shaped measurement of H12; its downweight is V6-visible via
   `base_priority`, invisible to raw RRF by construction).

Pre-registered expectations/biases:
- Dormancy inert: 0 dormanted in B (verified by gate below). No scrub anywhere (D3).
- Sleep-created summaries/consolidation nodes are GT-unlabeled → implicit negatives:
  a bias **against** B. Do not patch mid-flight.
- NREM duplicate rate not comparable to bench (prompt truth-fix N3).
- NREM edge yield is a single draw (capped/failed pairs unrecoverable); REM taxonomy
  restructuring is LLM-nondeterministic → arm-B reruns won't reproduce exactly.
- Phase 4 repair (≤10 organic misses) and theme review are in-arm production
  behavior; their events are not V6-visible.

## Config-parity table

| Item | Live/production | Bench-validated | Fork value |
|---|---|---|---|
| `SOMNIGRAPH_DATA_DIR` | `~/.claude/data` (set by wrapper) | isolated per-arm dirs | **arm dir, explicit on every command** |
| `SOMNIGRAPH_EMBEDDING_BACKEND` | unset → `openai` 1536d | unset | **unset** (db.py dim-guard backstops) |
| `OPENAI_API_KEY` | env or `DATA_DIR/openai_api_key` | file in store dir | file rides in BASE copy (fallback path moves with DATA_DIR) |
| `PYTHONUTF8` | wrapper sets IOENCODING | `precheck.py` sets it | **`1`, every command** (repro'd crash without it in review smokes) |
| `SOMNIGRAPH_NREM_PARALLEL` | default 2 | 2 | **default 2, valid only post-reboot**; else set `1` |
| `SOMNIGRAPH_SUMMARY_MIN_MEMBERS` | default 5 | 5 (H7) | default (31/33 live clusters clear it) |
| `SOMNIGRAPH_CONSOLIDATED_MEMBER_WEIGHT` | default 0.8 | 0.8 (H12) | default (set nowhere in repo — verified) |
| `sleep.py` mode | `--deep` (daily habit) | 07-02 fork `--deep` | **`--deep`** (N2) |
| `sleep.py --probes` | default 10 | 0 in bench | **`0` explicit** (D2; H-1) |
| `sleep.py --allow-live` | wrapper passes it | n/a | **never passed** (refusal guard active) |
| `sleep_nrem.py --legacy-merge` | off (link default) | off since H3 | off (unreachable via sleep.py anyway) |
| `sleep_consolidate.py --legacy-archive` | off | off (H12) | **off** |
| `fork.py --scrub-access-counts` | n/a | bench-only workaround | **NOT used** (D3) |
| eval `--configs` | — | `reranker` | `reranker` |
| eval `--gt-path` | stale default | explicit | **explicit** (never default) |
| eval `--recall-budget` | default 10000 | 5000 registered | **`5000` explicit** |
| eval `--recall-limit` | 20 | 20 | 20 |
| eval `--snapshot` | — | — | **never passed** (absent-snapshot copies the live DB) |
| eval `--model-path` | — | new this branch | retrain cells only: `<arm>\tuning_studies\reranker_model.txt` |
| Snapshot swap target | — | `~/.somnigraph/benchmark/production_snapshot.db` | same, `Copy-Item -Force` after `wal_checkpoint(TRUNCATE)` |

Paths below use `$FORK = C:\Users\Alexis\somnigraph-fork0715` (any dir passing
`fork.py`'s isolation guard works; not under `~/.claude/data`, not bare `~/.somnigraph`).

## Phase -1 — preconditions (day before)

1. **Reboot.** Then verify the pagefile actually grew (review measured 5,613 MB
   allocated against the 12/32 GB registry setting):
   ```powershell
   Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize
   # AllocatedBaseSize must be >= 12288. If not: $env:SOMNIGRAPH_NREM_PARALLEL = "1"
   ```
2. Branch `fix/prefork-review` merged to `main`; run from `main`.
3. Disk space: BASE + 2 arms ≈ 3× the live data dir (check `~/.claude/data` size;
   `tuning_cache/` rides along).

## Phase 0 — BASE (quiesced backup; the only live-store read)

```powershell
# STOP the memory MCP server first: close all Claude Code sessions (the server is
# session-launched). Verify no python holds memory.db open before backing up.
$FORK = "C:\Users\Alexis\somnigraph-fork0715"
New-Item -ItemType Directory -Force "$FORK\BASE"

# Whole-dir copy EXCEPT the DB files (sidecars: openai_api_key, taxonomy.json,
# rem_thin_migration_done, theme_decisions.json, personal JSONs, tuning_studies/...)
robocopy "C:\Users\Alexis\.claude\data" "$FORK\BASE" /E /XF memory.db memory.db-wal memory.db-shm

# The DB itself via the sqlite3 backup API (WAL-safe), then integrity gate:
python -c "import sqlite3; s=sqlite3.connect(r'C:\Users\Alexis\.claude\data\memory.db'); d=sqlite3.connect(r'$env:USERPROFILE\somnigraph-fork0715\BASE\memory.db'.replace('$env:USERPROFILE','C:\Users\Alexis')); s.backup(d); d.close(); s.close(); print('backup done')"
python -c "import sqlite3; print(sqlite3.connect(r'C:\Users\Alexis\somnigraph-fork0715\BASE\memory.db').execute('PRAGMA integrity_check').fetchone())"
# Gate: ('ok',)
```

## Phase 1 — fork the arms (NO scrub — D3)

```powershell
cd C:\Users\Alexis\repos\somnigraph
uv run experiments/sleep-bench/fork.py fork "$FORK\BASE" "$FORK\A"
uv run experiments/sleep-bench/fork.py fork "$FORK\BASE" "$FORK\B"
```

## Phase 2 — GT: built once from BASE, frozen

```powershell
$env:PYTHONUTF8 = "1"
$env:SOMNIGRAPH_DATA_DIR = "$FORK\BASE"
uv run python scripts/build_gt_from_feedback.py --db "$FORK\BASE\memory.db" `
  --output "$FORK\gt_fork0715.json" `
  --vec-input-overrides-output "$FORK\gt_fork0715_vec_inputs.json" `
  --sample-weights-output "$FORK\gt_fork0715_weights.json"
Get-FileHash "$FORK\gt_fork0715.json" -Algorithm SHA256   # record in manifest
```

**Holdout slice (pre-registered):** primary eval slice for cross-model contrasts
(2, 3, 4) = out-of-sample queries only — the probe-hard class (weight 0 in the
schedule already) **plus abstinence-window queries** (feedback logged after
2026-07-01). Zero the window queries' training weight in
`gt_fork0715_weights.json` (identify them from BASE's `memory_events` feedback
timestamps; **verify the sidecar's schema before editing** — it is the v2
modes/holdout format, see `~/.claude/data/tuning_studies/v6_bundle/TRAINING.md`).
Full-GT numbers remain valid for within-model contrasts (1, 5); report both,
caveat the window slice's n (likely a few hundred).

## Phase 3 — sleep arm B (Alexis's terminal; multi-hour)

```powershell
$env:PYTHONUTF8 = "1"
$env:SOMNIGRAPH_DATA_DIR = "$FORK\B"
cd C:\Users\Alexis\repos\somnigraph
uv run scripts/sleep.py --deep --probes 0
# Banner must show: data dir = $FORK\B  (fix 6). If it shows ~/.claude/data, STOP.
```

**Gates after sleep.py (all read-only, `mode=ro` URI):**

```powershell
# G1: zero failed classification batches (report prints "failed_batches: N/M"; also:)
Select-String -Path "$FORK\B\sleep_logs\nrem_deep_*.log" -Pattern "failed_batches"
# G2: no PPR-invisible edges (edge-embedding health, H-4):
python -c "import sqlite3; db=sqlite3.connect('file:C:/Users/Alexis/somnigraph-fork0715/B/memory.db?mode=ro', uri=True); print('NULL-embedding context edges:', db.execute(\"SELECT COUNT(*) FROM memory_edges WHERE linking_context IS NOT NULL AND linking_context != '' AND linking_embedding IS NULL\").fetchone()[0])"
# Gate: 0. If >0: credentials failed mid-NREM; fix key, re-run NREM (backfill self-heals), re-gate.
# G3: dormancy inert as pre-registered (D3):
python -c "import sqlite3; db=sqlite3.connect('file:C:/Users/Alexis/somnigraph-fork0715/B/memory.db?mode=ro', uri=True); print('dormant:', db.execute(\"SELECT COUNT(*) FROM memories WHERE status='dormant'\").fetchone()[0])"
# Gate: unchanged vs BASE (expected delta 0). A large jump = predicate surprise; STOP and investigate.
# G4: REM ran incrementally, not cold-start: summaries were refreshed, not mass-deleted
#     (active summary count should be ~32 + new, not regenerated-from-zero; check the rem log for the
#      one-time-migration line — it must NOT appear).

# B-mid checkpoint (consolidate's own signal — N1):
python -c "import sqlite3; db=sqlite3.connect(r'C:\Users\Alexis\somnigraph-fork0715\B\memory.db'); db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close()"
Copy-Item "$FORK\B\memory.db" "$FORK\Bmid_memory.db"
```

## Phase 4 — consolidate arm B (Alexis's terminal; Opus, one pass)

```powershell
# Same env as Phase 3 (DATA_DIR = $FORK\B, PYTHONUTF8=1)
uv run scripts/sleep_consolidate.py -v
```

Gates: report's `Rejected: N` must match the per-line `REJECTED` count (fix 8 makes
the classes visible); `Skipped ... majority-consolidated` expected 0 on this first
pass; note nodes created / members downweighted (protected-heavy live clusters →
low downweight counts are correct).

## Phase 5 — V6 cells (all six cells same day, this order — M-5)

Per cell: checkpoint → swap snapshot → eval → verify identity line.

```powershell
$env:PYTHONUTF8 = "1"; $env:SOMNIGRAPH_DATA_DIR = "$FORK\BASE"   # eval reads snapshot; DATA_DIR only feeds key fallback
function Swap-Snapshot($dbfile) {
  python -c "import sqlite3,sys; db=sqlite3.connect(sys.argv[1]); db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close()" $dbfile
  Copy-Item -Force $dbfile "C:\Users\Alexis\.somnigraph\benchmark\production_snapshot.db"
}

Swap-Snapshot "$FORK\A\memory.db"
uv run scripts/locomo_bench/eval_retrieval.py --dataset production --configs reranker `
  --gt-path "$FORK\gt_fork0715.json" --recall-budget 5000 --recall-limit 20 `
  --output "$FORK\eval_A_v6.jsonl"
# Gate per cell: log line "Scorer identity: C:\Users\Alexis\.claude\data\...reranker_model.txt (sha256=...)"
# — deployed V6 for these three cells. The run HARD-FAILS if no model loads (fix 1). Record each SHA.

Swap-Snapshot "$FORK\Bmid_memory.db"
uv run ... --output "$FORK\eval_Bmid_v6.jsonl"     # same flags
Swap-Snapshot "$FORK\B\memory.db"
uv run ... --output "$FORK\eval_Bfull_v6.jsonl"    # same flags
```

## Phase 6 — retrains (AFTER all V6 cells; `_export_for_mcp` overwrites arm artifacts)

Identical recipe both arms, per `v6_bundle/TRAINING.md` (only the store differs):

```powershell
$env:PYTHONUTF8 = "1"
$env:SOMNIGRAPH_DATA_DIR = "$FORK\A"
uv run scripts/train_reranker.py --gt "$FORK\gt_fork0715.json" `
  --vec-input-overrides "$FORK\gt_fork0715_vec_inputs.json" `
  --sample-weights "$FORK\gt_fork0715_weights.json" --pinned-boost 5.0

$env:SOMNIGRAPH_DATA_DIR = "$FORK\B"
uv run scripts/train_reranker.py  <identical flags>

Get-FileHash "$FORK\A\tuning_studies\reranker_model.txt", "$FORK\B\tuning_studies\reranker_model.txt" -Algorithm SHA256
```

## Phase 7 — retrain cells (same day as Phase 5)

```powershell
Swap-Snapshot "$FORK\A\memory.db"
uv run scripts/locomo_bench/eval_retrieval.py --dataset production --configs reranker `
  --gt-path "$FORK\gt_fork0715.json" --recall-budget 5000 --recall-limit 20 `
  --model-path "$FORK\A\tuning_studies\reranker_model.txt" `
  --output "$FORK\eval_A_rA.jsonl"
Swap-Snapshot "$FORK\Bmid_memory.db"
uv run ... --model-path "$FORK\B\tuning_studies\reranker_model.txt" --output "$FORK\eval_Bmid_rB.jsonl"
Swap-Snapshot "$FORK\B\memory.db"
uv run ... --model-path "$FORK\B\tuning_studies\reranker_model.txt" --output "$FORK\eval_Bfull_rB.jsonl"
# Gate: each cell's "Scorer identity" SHA must equal the Phase-6 hash for that arm.
```

## Phase 8 — contrasts

```powershell
$env:PYTHONUTF8 = "1"
uv run experiments/sleep-fork/analyze_sleep_fork.py --config reranker `
  --baseline "$FORK\eval_A_v6.jsonl" --slept "$FORK\eval_Bfull_v6.jsonl" `
  --gt "$FORK\gt_fork0715.json" --db-a "$FORK\A\memory.db" --db-b "$FORK\B\memory.db" `
  --out "$FORK\contrast1_naive.json"
# ... one run per contrast pair (2: A_rA vs Bfull_rB; 4: A_v6 vs A_rA with db-a=db-b=A;
#     5: Bmid vs Bfull under each model). The scorer_note in each output now records
#     both sides' model SHAs — verify they match the intended cells before reading numbers.
# Window-slice versions: filter per-query rows by the pre-registered query list, or
# rerun analyze on slice-filtered JSONLs. Report full-GT and window-slice per the plan.
```

Results + manifest (all SHAs, cell timestamps, gate outcomes) → this directory;
findings doc + session file per STEWARDSHIP retrospective.

## Abort criteria

- Any gate fails and its listed remedy doesn't clear it same-day → stop, keep arms
  on disk (they're cheap), diagnose in a fresh session. The one-shot condition is
  BASE, and BASE keeps: arms can be re-forked from it as long as no one sleeps the
  live store in the interim.
- `sleep.py` banner ever shows the live data dir → hard stop (should be impossible:
  refusal guard + wrapper-only `--allow-live`).
