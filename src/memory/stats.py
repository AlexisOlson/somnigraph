"""Memory store diagnostic stats."""

import json

from memory.decay import effective_priority
from memory.events import _row_get


def compute_stats(db) -> str:
    """Compute diagnostic overview of the memory store.

    Returns formatted string with counts, access stats, decay distribution,
    and token budget without loading memory content.
    """
    # Counts by status and category
    status_counts = db.execute(
        "SELECT status, category, count(*) as cnt FROM memories "
        "GROUP BY status, category ORDER BY status, category"
    ).fetchall()

    # Total tokens for active memories
    token_total = db.execute(
        "SELECT coalesce(sum(token_count), 0) FROM memories WHERE status = 'active'"
    ).fetchone()[0]

    total_active = db.execute(
        "SELECT count(*) FROM memories WHERE status = 'active'"
    ).fetchone()[0]

    # Top accessed (by recall_count — actual retrieval, not startup loading)
    top_accessed = db.execute(
        "SELECT id, summary, content, recall_count, reflect_count, category "
        "FROM memories WHERE status = 'active' "
        "ORDER BY (recall_count + reflect_count) DESC LIMIT 5"
    ).fetchall()

    # Never accessed (neither recalled nor reflected)
    never_accessed = db.execute(
        "SELECT count(*) FROM memories WHERE status = 'active' "
        "AND recall_count = 0 AND reflect_count = 0"
    ).fetchone()[0]

    # Recently accessed (last 7 days)
    recent_accessed = db.execute(
        "SELECT count(*) FROM memories WHERE status = 'active' "
        "AND last_accessed > datetime('now', '-7 days')"
    ).fetchone()[0]

    # Decay distribution
    active_rows = db.execute(
        "SELECT base_priority, last_accessed, decay_rate, category, flags FROM memories WHERE status = 'active'"
    ).fetchall()

    at_floor_count = 0
    for row in active_rows:
        row_flags = json.loads(row["flags"]) if row["flags"] else []
        ep = effective_priority(
            row["base_priority"], row["last_accessed"],
            decay_rate=row["decay_rate"], category=row["category"],
            flags=row_flags,
        )
        if ep < row["base_priority"] * 0.5:
            at_floor_count += 1

    # Date range
    oldest = db.execute(
        "SELECT created_at FROM memories WHERE status = 'active' "
        "ORDER BY created_at ASC LIMIT 1"
    ).fetchone()
    newest = db.execute(
        "SELECT created_at FROM memories WHERE status = 'active' "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchone()

    # Feedback stats (utility-based)
    fb_rows = db.execute("""
        SELECT memory_id, context FROM memory_events
        WHERE event_type = 'feedback'
    """).fetchall()

    fb_memories = {}  # memory_id -> list of parsed contexts
    utility_sum = 0.0
    durability_pos = 0
    durability_neg = 0
    fb_count = 0
    for r in fb_rows:
        mid_fb = r["memory_id"]
        try:
            ctx = json.loads(r["context"]) if r["context"] else {}
        except (json.JSONDecodeError, TypeError):
            ctx = {}
        if mid_fb not in fb_memories:
            fb_memories[mid_fb] = []
        fb_memories[mid_fb].append(ctx)
        utility_sum += ctx.get("utility", 0.0)
        dur = ctx.get("durability", 0.0)
        if isinstance(dur, (int, float)):
            if dur > 0.01:
                durability_pos += 1
            elif dur < -0.01:
                durability_neg += 1
        fb_count += 1

    # Decay rate distribution
    custom_decay = db.execute(
        "SELECT count(*) FROM memories WHERE status = 'active' AND decay_rate IS NOT NULL"
    ).fetchone()[0]
    default_decay = total_active - custom_decay

    # Event log stats
    event_count = db.execute("SELECT count(*) FROM memory_events").fetchone()[0]
    event_types = db.execute("""
        SELECT event_type, count(*) as cnt FROM memory_events
        GROUP BY event_type ORDER BY cnt DESC
    """).fetchall()

    # Sleep status
    last_sleep_row = db.execute(
        "SELECT completed_at FROM sleep_log ORDER BY completed_at DESC LIMIT 1"
    ).fetchone()
    unprocessed = db.execute(
        "SELECT count(*) FROM memories WHERE status = 'active' "
        "AND last_sleep_processed IS NULL"
    ).fetchone()[0]

    # Confidence distribution
    conf_stats = db.execute("""
        SELECT AVG(confidence) as avg_conf,
               SUM(CASE WHEN confidence < 0.4 THEN 1 ELSE 0 END) as low_count,
               SUM(CASE WHEN confidence > 0.8 THEN 1 ELSE 0 END) as high_count
        FROM memories WHERE status = 'active'
    """).fetchone()

    # Format output
    lines = ["## Memory Store Stats\n"]

    lines.append("### Counts by Category & Status")
    current_status = None
    for row in status_counts:
        if row["status"] != current_status:
            current_status = row["status"]
            lines.append(f"\n**{current_status}**:")
        lines.append(f"  {row['category']}: {row['cnt']}")

    lines.append(f"\n### Token Budget")
    lines.append(f"Active memories: {token_total} tokens")

    lines.append(f"\n### Access Stats")
    lines.append("Top recalled:")
    for row in top_accessed:
        summary = row["summary"] or row["content"][:50]
        rc = row["recall_count"] or 0
        fc = row["reflect_count"] or 0
        parts = []
        if rc: parts.append(f"{rc} recall")
        if fc: parts.append(f"{fc} reflect")
        count_str = " + ".join(parts) if parts else "0"
        lines.append(f"  [{row['category']}] {summary} -- {count_str}")

    lines.append(f"Never recalled or reflected: {never_accessed}")
    lines.append(f"Accessed in last 7 days: {recent_accessed} / {total_active}")

    # Feedback section
    if fb_memories:
        lines.append(f"\n### Retrieval Feedback")
        lines.append(f"Memories with feedback: {len(fb_memories)}")
        lines.append(f"Total feedback events: {fb_count}")
        if fb_count:
            lines.append(f"Mean utility: {utility_sum / fb_count:.3f}")
        if durability_pos or durability_neg:
            lines.append(f"Durability signals: {durability_neg} stale, {durability_pos} enduring")

    if event_count:
        lines.append(f"\n### Event Log")
        lines.append(f"Total events: {event_count}")
        for row in event_types:
            lines.append(f"  {row['event_type']}: {row['cnt']}")

    lines.append(f"\n### Decay Distribution")
    lines.append(f"Near floor (effective < 50% of base): {at_floor_count} / {total_active}")
    lines.append(f"Custom decay rates: {custom_decay} / {total_active}")
    lines.append(f"Using category defaults: {default_decay} / {total_active}")

    lines.append(f"\n### Confidence Distribution")
    if conf_stats and conf_stats["avg_conf"] is not None:
        lines.append(f"Mean: {conf_stats['avg_conf']:.2f}")
        lines.append(f"Low confidence (<0.4): {conf_stats['low_count'] or 0}")
        lines.append(f"High confidence (>0.8): {conf_stats['high_count'] or 0}")
    else:
        lines.append("No confidence data available")

    lines.append(f"\n### Sleep Status")
    if last_sleep_row:
        lines.append(f"Last sleep: {last_sleep_row['completed_at'][:16]}")
    else:
        lines.append("Last sleep: never")
    lines.append(f"Unprocessed memories: {unprocessed}")

    lines.append(f"\n### Date Range")
    if oldest:
        lines.append(f"Oldest: {oldest['created_at'][:10]}")
    if newest:
        lines.append(f"Newest: {newest['created_at'][:10]}")

    return "\n".join(lines)
