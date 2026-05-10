"""Adversarial target selection for probe_recall.

Reads the latest `audit_*.json` from `data/pathology_audits/` (produced by
`scripts/audit_reranker_pathology.py --output ''`) and returns a list of
target memory entries shaped to match `probe_recall.select_targets`'s output.

Adversarial differs from coverage-fill on three axes:
  - selection: pathology memories (reranker_rank - best_channel_rank > N),
    not coverage-underserved memories
  - cap: pathology-flagged memories cap at 8 pins (vs 4 for coverage), giving
    the reranker two bundles' worth before declaring it a feature ceiling
  - usage: returns a flat candidate list with weights that bias toward
    higher-gap pathologies; probe_recall.process_group still does the actual
    bundled crafting

The selector consciously does NOT do persistence detection (audit-N-still-on-list
counting). The 8-pin cap already prevents infinite probing of unfixable
memories — once a pathology hits cap=8, it stops getting probed regardless
of audit history. If we need finer-grained "stop probing because this won't
heal" detection later, it layers on top of consecutive audit JSONs.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from memory.constants import DATA_DIR
from memory.db import get_db


PATHOLOGY_AUDITS_DIR = DATA_DIR / "pathology_audits"
MAX_PATHOLOGY_PINS = 8


def latest_audit_path() -> Path | None:
    """Return the most recent audit JSON, or None if none exist."""
    if not PATHOLOGY_AUDITS_DIR.exists():
        return None
    audits = sorted(PATHOLOGY_AUDITS_DIR.glob("audit_*.json"))
    return audits[-1] if audits else None


def load_pathology_set(audit_path: Path | None = None) -> dict:
    """Load the latest pathology audit. Returns {} if none available."""
    if audit_path is None:
        audit_path = latest_audit_path()
    if audit_path is None:
        return {}
    return json.loads(audit_path.read_text())


def select_pathology_targets(
    num_groups: int,
    *,
    memory_content: dict,
    audit: dict | None = None,
    exclude: set[str] | None = None,
    content_chars: int = 600,
) -> tuple[list[dict], dict]:
    """Select up to `num_groups` pathology memories as probe targets.

    Args:
        num_groups: budget — return at most this many target entries
        memory_content: {mid: {summary, themes, category}} from probe_recall
        audit: optional pre-loaded audit dict; defaults to latest on disk
        exclude: memory IDs to skip (already picked by another selector)
        content_chars: per-target content snippet length

    Returns:
        (target_list, info) where target_list mirrors the shape produced by
        probe_recall.select_targets's eligible-list pass; info reports
        provenance for logging.
    """
    if audit is None:
        audit = load_pathology_set()
    info = {
        "audit_path": str(latest_audit_path()) if latest_audit_path() else None,
        "audit_timestamp": audit.get("audit_timestamp"),
        "audit_pathologies": len(audit.get("pathologies", [])),
        "skipped_at_cap": 0,
        "skipped_excluded": 0,
        "skipped_inactive": 0,
        "selected": 0,
    }

    pathologies = audit.get("pathologies", [])
    if not pathologies:
        return [], info

    # Per-memory pin count (probe_target events) — same source the
    # coverage-fill cap reads, just with a higher MAX.
    db = get_db()
    pin_counts = Counter()
    pathology_mids = {p["memory_id"] for p in pathologies}
    for r in db.execute(
        "SELECT memory_id, COUNT(*) AS n FROM memory_events "
        "WHERE event_type = 'probe_target' GROUP BY memory_id"
    ):
        if r["memory_id"] in pathology_mids:
            pin_counts[r["memory_id"]] = r["n"]

    excluded = exclude or set()
    eligible = []
    for p in pathologies:
        mid = p["memory_id"]
        if mid in excluded:
            info["skipped_excluded"] += 1
            continue
        if pin_counts.get(mid, 0) >= MAX_PATHOLOGY_PINS:
            info["skipped_at_cap"] += 1
            continue
        if mid not in memory_content:
            info["skipped_inactive"] += 1
            continue
        eligible.append((p, mid))

    if not eligible:
        db.close()
        return [], info

    # Load full content for the eligible candidates only
    candidates = []
    for p, mid in eligible:
        row = db.execute(
            "SELECT content FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        if row:
            candidates.append({
                "id": mid,
                "themes": memory_content[mid].get("themes", "[]"),
                "summary": memory_content[mid].get("summary", ""),
                "category": memory_content[mid].get("category", ""),
                "content": row["content"][:content_chars],
                "_pathology_gap": p["gap"],
                "_pathology_pin_count": pin_counts.get(mid, 0),
            })
    db.close()

    if not candidates:
        return [], info

    # Selection: weighted sampling without replacement (Efraimidis-Spirakis),
    # weight = gap (larger gap = bigger pathology = pick first), softened by
    # 1/(1+pins) so memories already partway pinned cede priority.
    def _weight(c: dict) -> float:
        gap = max(c["_pathology_gap"], 1)
        pins = c["_pathology_pin_count"]
        return gap / (1.0 + pins)

    keyed = []
    for c in candidates:
        w = max(_weight(c), 1e-9)
        u = random.random()
        keyed.append((u ** (1.0 / w), c))
    keyed.sort(key=lambda x: x[0], reverse=True)
    selected = [c for _, c in keyed[:num_groups]]
    info["selected"] = len(selected)
    return selected, info
