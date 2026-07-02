import sqlite3, json
from collections import Counter
A = r'D:\somnigraph-exp\sleepfork-92fdacb-1782966620\A-frozen\memory.db'
B = r'D:\somnigraph-exp\sleepfork-92fdacb-1782966620\B-slept\memory.db'
ca=sqlite3.connect(A); cb=sqlite3.connect(B)
for c in (ca,cb): c.row_factory=sqlite3.Row

a_ids=set(r[0] for r in ca.execute("select id from memory_edges"))
b_rows=cb.execute("select rowid,id,edge_type,created_by,created_at,weight from memory_edges").fetchall()
b_ids=set(r["id"] for r in b_rows)
new=[r for r in b_rows if r["id"] not in a_ids]
gone=a_ids - b_ids
print(f"A edges={len(a_ids)} B edges={len(b_rows)} | new(in B not A)={len(new)} | gone(in A not B)={len(gone)} | net={len(b_rows)-len(a_ids)}")

# probe window start: first B-only event time
maxa=ca.execute("select max(id) from memory_events").fetchone()[0]
probe_start=cb.execute("select min(created_at) from memory_events where id>?", (maxa,)).fetchone()[0]
print("probe/repair window start:", probe_start)

print("\nnew edges by created_by:", Counter(r["created_by"] for r in new))
print("new edges by edge_type:", Counter(r["edge_type"] for r in new))
# split new edges by created_at vs probe_start
before=[r for r in new if (r["created_at"] or "") < probe_start]
after =[r for r in new if (r["created_at"] or "") >= probe_start]
print(f"\nnew edges created BEFORE probe (consolidation): {len(before)}")
print(f"new edges created DURING/after probe window (probe artifacts): {len(after)}")
print("  after-window by created_by:", Counter(r["created_by"] for r in after))
print("  after-window by edge_type:", Counter(r["edge_type"] for r in after))
print("  before-window by created_by:", Counter(r["created_by"] for r in before))

# of the consolidation (before) new edges, how many were probe-bumped later?
bumped=set()
for r in cb.execute("select context from memory_events where event_type='edge_weight_change' and id>?", (maxa,)):
    try: rid=json.loads(r["context"]).get("edge_rowid")
    except Exception: rid=None
    if rid is not None: bumped.add(rid)
before_rowids={r["rowid"] for r in before}
print(f"\nconsolidation(before) new edges probe-bumped: {len(before_rowids & bumped)} of {len(before)}")
