"""Strip the standard-B probe/repair footprint from B-noprobe, leaving ONLY the
NREM/REM consolidation. Isolates standard-consolidation-only for the decisive arm.

The probe reaches formula scoring through THREE channels — all removed here:
  1. feedback events  -> feedback_raw (tools.py:619). Deleted (all B-only events).
  2. retrieved events -> Hebbian co-retrieval window (tools.py:642). Deleted.
  3. memory_edges.weight -> PPR (tools.py:683/700). Two sub-cases:
       (a) shared edges (id present in A): restored to A's pre-sleep weight.
       (b) new consolidation edges (created by NREM/REM before the probe) that the
           probe later co-retrieval-bumped: reset to their sleep-creation weight
           (= earliest probe-window weight_before for that rowid).
  Plus: probe-CREATED edges (co_retrieval / recall_feedback, created during the probe
  window) are deleted outright — they don't exist in a consolidation-only store.

Kept intact: the 871 NREM/REM consolidation edges (809 sleep + 62 sleep_rem), the 765
A-edges sleep removed stay removed, refreshed summaries. Result = the store exactly as
it stood after REM, before the probe ran.

Run on a FRESH copy of B's memory.db (edge_weight_change events must still be present).
"""
import sqlite3, json

A  = r'D:\somnigraph-exp\sleepfork-92fdacb-1782966620\A-frozen\memory.db'
BN = r'D:\somnigraph-exp\sleepfork-92fdacb-1782966620\B-noprobe\memory.db'

c = sqlite3.connect(BN)
c.row_factory = sqlite3.Row
c.execute("ATTACH DATABASE ? AS adb", (A,))

max_a_event = c.execute("SELECT max(id) FROM adb.memory_events").fetchone()[0]
probe_start = c.execute("SELECT min(created_at) FROM memory_events WHERE id>?", (max_a_event,)).fetchone()[0]
print("A max event id:", max_a_event, "| probe window start:", probe_start)

# --- capture creation weights of probe-bumped edges BEFORE deleting the events ---
bumped_first_before = {}  # rowid -> earliest weight_before (== pre-probe/creation weight)
for r in c.execute("SELECT context FROM memory_events WHERE event_type='edge_weight_change' AND id>? ORDER BY id", (max_a_event,)):
    try:
        ctx = json.loads(r["context"])
    except Exception:
        continue
    rid = ctx.get("edge_rowid")
    if rid is not None and rid not in bumped_first_before:
        bumped_first_before[rid] = ctx.get("weight_before")
print("distinct probe-bumped edge rowids:", len(bumped_first_before))

# baseline counts
fb_before = c.execute("SELECT count(*) FROM memory_events WHERE event_type='feedback'").fetchone()[0]
a_fb = c.execute("SELECT count(*) FROM adb.memory_events WHERE event_type='feedback'").fetchone()[0]

# 1+2) delete ALL B-only events (feedback, retrieved, edge_weight_change, recall_*, probe_target)
c.execute("DELETE FROM memory_events WHERE id > ?", (max_a_event,))

# 3-probe-created) delete new edges created during/after the probe window
del_edges = c.execute("""
    DELETE FROM memory_edges
     WHERE id NOT IN (SELECT id FROM adb.memory_edges)
       AND created_at >= ?
""", (probe_start,)).rowcount

# 3a) restore shared-edge weights to A's pre-sleep value
c.execute("""
    UPDATE memory_edges
       SET weight = (SELECT a.weight FROM adb.memory_edges a WHERE a.id = memory_edges.id)
     WHERE id IN (SELECT id FROM adb.memory_edges)
""")

# 3b) reset probe-bumped consolidation (new, pre-probe) edges to creation weight
reset = 0
for rid, w0 in bumped_first_before.items():
    # only touch edges that still exist and are NEW (id not in A) -> consolidation edges
    row = c.execute("SELECT id FROM memory_edges WHERE rowid=?", (rid,)).fetchone()
    if row is None:
        continue  # was a probe-created edge already deleted, or a shared edge (handled by 3a)
    in_a = c.execute("SELECT 1 FROM adb.memory_edges WHERE id=?", (row["id"],)).fetchone()
    if in_a:
        continue  # shared edge already restored to A in step 3a
    c.execute("UPDATE memory_edges SET weight=? WHERE rowid=?", (w0, rid))
    reset += 1
c.commit()

# ---- verify ----
fb_after = c.execute("SELECT count(*) FROM memory_events WHERE event_type='feedback'").fetchone()[0]
ret_after = c.execute("SELECT count(*) FROM memory_events WHERE event_type='retrieved'").fetchone()[0]
a_ret = c.execute("SELECT count(*) FROM adb.memory_events WHERE event_type='retrieved'").fetchone()[0]
bonly_left = c.execute("SELECT count(*) FROM memory_events WHERE id>?", (max_a_event,)).fetchone()[0]
div = c.execute("SELECT count(*) FROM memory_edges e JOIN adb.memory_edges a ON a.id=e.id WHERE e.weight IS NOT a.weight").fetchone()[0]
edges_bn = c.execute("SELECT count(*) FROM memory_edges").fetchone()[0]
edges_a = c.execute("SELECT count(*) FROM adb.memory_edges").fetchone()[0]
probe_edges_left = c.execute("SELECT count(*) FROM memory_edges WHERE created_by IN ('co_retrieval','recall_feedback') AND id NOT IN (SELECT id FROM adb.memory_edges)").fetchone()[0]
print("\n--- after complete strip ---")
print(f"feedback events: {fb_after} (A={a_fb}, match={fb_after==a_fb})")
print(f"retrieved events: {ret_after} (A={a_ret}, match={ret_after==a_ret})")
print(f"B-only events remaining: {bonly_left} (target 0)")
print(f"probe-created edges deleted: {del_edges} (target 31)")
print(f"consolidation edges weight-reset to creation: {reset} (expected ~359)")
print(f"shared edges with weight != A: {div} (target 0)")
print(f"new probe-created edges remaining: {probe_edges_left} (target 0)")
print(f"total edges: BN={edges_bn} A={edges_a} net={edges_bn-edges_a} (consolidation kept)")
c.close()
print("COMPLETE STRIP DONE")
