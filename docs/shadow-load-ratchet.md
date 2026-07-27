# `shadow_load` only ever increases

**Status:** open, unfixed, not yet triaged against the priority list. Found
2026-07-27.

## Summary

`memories.shadow_load` is documented as "how much newer similar memories
suppress this one." In practice the stored value never decreases. Once a
memory records a nonzero suppression score, that number stays even after the
newer memory that caused it is archived, superseded, or no longer similar
enough to suppress anything.

A stored value therefore means "this memory was suppressed at some point," not
"this memory is suppressed now." A stored `0.000` means "no value was ever
written here," which is not the same as "no suppression," and the two cannot be
told apart from the column alone.

## Why it happens

The column is written in exactly one place, `scripts/sleep_nrem.py`, during the
NREM sleep pass:

```python
# lines 237-238 — only nonzero scores are collected
if shadow > 0.0:
    shadow_updates[rel_id] = max(shadow_updates.get(rel_id, 0.0), shadow)

# line 743 — the only UPDATE, fed exclusively from that dict
db.execute("UPDATE memories SET shadow_load = ? WHERE id = ?", (shadow_val, mid))
```

The `shadow > 0.0` guard means a score that has fallen to zero is never
collected, so it is never written. A memory whose suppression has genuinely
ended simply does not appear in `shadow_updates`, and its old value survives.
The `max()` prevents any decrease within a single run. No other code path
writes, decays, or resets the column.

Two further limits on coverage:

1. Suppression scores are computed only for memories that appear as an *older*
   neighbor of a memory the pass is currently processing
   (`sleep_nrem.py:229-238`). A memory that nothing newer resembles is never
   scored at all.
2. Deep mode skips memories whose neighborhood has not changed since they were
   last processed (`sleep_nrem.py:222-226`), so even eligible memories can be
   passed over.

Both paths call `_find_related_memories(..., limit=5)`, so which neighbors are
visible depends on a five-result retrieval window.

## Evidence

Production database (1,377 active memories), last NREM run 2026-07-01, measured
2026-07-27. `_compute_shadow_load` was re-run read-only over the 1,158 memories
predating that run and compared against the stored column.

| Stored vs recomputed | n |
|---|---|
| both above zero | 522 |
| stored above zero, recomputed zero | 66 |
| stored zero, recomputed above zero | 52 |
| **disagreement** | **118 / 1,158 (10.2%)** |

Where the 118 come from:

| Cause | n |
|---|---|
| The suppressing memory was written after the last NREM run | 28 |
| The suppressing memory predates that run, but the memory was never scored | 24 |
| Recomputation agrees once the neighbor window widens from 5 to 40 | 23 |
| Still disagrees at any window — the stored value is simply stale | 43 |

Only 28 of 118 are explained by the unusually long gap since the last run. The
rest would appear after any interval.

Separately, all 219 memories written since that run hold a stored value of
exactly `0.000`, while 64 of them (29.2%, highest 0.68) score above zero when
computed fresh.

## Why it matters

- The bias is one-directional and grows with time, so it is worst for
  long-lived stores, which is the intended use case.
- `scripts/tune_memory.py:476-479` reads the column into `shadow_map`, so
  tuning work built on it is reading stale values.
- `docs/roadmap.md` lesson 7 lists the shadow penalty among three intuitive
  scoring signals that proved useless or counterproductive. If that result was
  measured against this column, the signal was never cleanly tested and the
  negative finding may describe the instrument rather than the idea.

## Open questions before changing anything

- Is the behavior intentional? "Was ever redundant" is a reasonable thing to
  track. It is just not what the column's name, its docstring, or its consumers
  suggest it holds.
- If the value should be current, the options are: recompute and write every
  memory during NREM (costs a full pass over the store), decay stored values,
  or drop the column and compute on demand.
- Does the lesson 7 result need re-running against a correctly maintained
  column?

## Reproducing

Reads stored embeddings only — no API calls, no writes.

```python
import sys, sqlite3
sys.path.insert(0, "src")
from memory.db import get_db
from memory import _find_related_memories, _compute_shadow_load

db = get_db(); db.row_factory = sqlite3.Row
last = db.execute("SELECT MAX(started_at) FROM sleep_log").fetchone()[0]
rows = db.execute(
    "SELECT id, content, shadow_load FROM memories "
    "WHERE status='active' AND created_at <= ?", (last,)
).fetchall()

for r in rows:
    rel = _find_related_memories(db, r["id"], r["content"] or "")
    fresh = _compute_shadow_load(db, r["id"], rel)
    stored = r["shadow_load"] or 0.0
    if (stored > 0) != (fresh > 0):
        print(r["id"], stored, fresh)
```

Run with `SOMNIGRAPH_DATA_DIR` pointing at the database under test.
