import sqlite3, json
A  = r'D:\somnigraph-exp\sleepfork-92fdacb-1782966620\A-frozen\memory.db'
B  = r'D:\somnigraph-exp\sleepfork-92fdacb-1782966620\B-slept\memory.db'      # intact, has probe events
BN = r'D:\somnigraph-exp\sleepfork-92fdacb-1782966620\B-noprobe\memory.db'    # stripped

ca=sqlite3.connect(A); cb=sqlite3.connect(B); cn=sqlite3.connect(BN)
for c in (ca,cb,cn): c.row_factory=sqlite3.Row

# (a) retrieved-event count: BN should equal A (probe retrieval events removed from hebb window)
ra=ca.execute("select count(*) from memory_events where event_type='retrieved'").fetchone()[0]
rn=cn.execute("select count(*) from memory_events where event_type='retrieved'").fetchone()[0]
rb=cb.execute("select count(*) from memory_events where event_type='retrieved'").fetchone()[0]
print(f"[hebb channel] retrieved events: A={ra}  B={rb}  B-noprobe={rn}  (BN==A: {rn==ra})")
# also confirm within 30d window specifically (that's what hebb reads)
from datetime import datetime, timezone, timedelta
lb=(datetime.now(timezone.utc)-timedelta(days=30)).isoformat()
ra30=ca.execute("select count(*) from memory_events where event_type='retrieved' and query is not null and query!='' and created_at>?", (lb,)).fetchone()[0]
rn30=cn.execute("select count(*) from memory_events where event_type='retrieved' and query is not null and query!='' and created_at>?", (lb,)).fetchone()[0]
print(f"[hebb channel] retrieved-in-30d-window: A={ra30}  B-noprobe={rn30}  (match: {ra30==rn30})")

# (b) new consolidation edges (id in B not in A): were any probe-bumped?
a_edge_ids=set(r[0] for r in ca.execute("select id from memory_edges"))
# map new edges -> rowid in B(=BN)
new_edges=cb.execute("select rowid,id,weight,edge_type,created_at from memory_edges").fetchall()
new_edges=[r for r in new_edges if r["id"] not in a_edge_ids]
new_rowids={r["rowid"] for r in new_edges}
print(f"\n[ppr channel] new consolidation edges (id not in A): {len(new_edges)}")

# probe-window edge_weight_change events in B reference edge_rowid in context
maxa=ca.execute("select max(id) from memory_events").fetchone()[0]
bumped_rowid_first_before={}
for r in cb.execute("select id,context from memory_events where event_type='edge_weight_change' and id>? order by id", (maxa,)):
    try: ctx=json.loads(r["context"])
    except Exception: continue
    rid=ctx.get("edge_rowid")
    if rid is None: continue
    if rid not in bumped_rowid_first_before:
        bumped_rowid_first_before[rid]=ctx.get("weight_before")
bumped_new = new_rowids & set(bumped_rowid_first_before)
print(f"[ppr channel] new edges probe-BUMPED: {len(bumped_new)}")
if bumped_new:
    # compare BN current weight vs sleep-creation weight (earliest weight_before)
    mism=0
    for rid in list(bumped_new)[:2000]:
        cur=cn.execute("select weight from memory_edges where rowid=?", (rid,)).fetchone()
        cur=cur[0] if cur else None
        creation=bumped_rowid_first_before[rid]
        if cur!=creation: mism+=1
    print(f"[ppr channel] of bumped new edges, BN weight != sleep-creation weight: {mism} -> RE-STRIP NEEDED" if mism else "[ppr channel] bumped new edges already at creation weight (ok)")
else:
    print("[ppr channel] no new edges were probe-bumped -> new-edge weights are sleep-creation values (ok)")
