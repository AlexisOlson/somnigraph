"""LoCoMo evidence-turn ground truth for the Stage A retrieval rig.

Conceptually the same thing eval_retrieval.py's `eval_locomo()` builds
(question -> evidence-turn memory ids, via a (conv_id, dia_id) -> memory_id
map), but written fresh as a standalone, store-status-agnostic module. See
`resolve_dia_map` for why "status-agnostic" is a real correctness choice
here, not just independence for its own sake.

`locomo_bench.ingest.{load_locomo,extract_turns,extract_questions}` are
reused directly -- pure data loading, not eval logic (allowed per this
rig's scope).

Corrected-GT note: the locomo-audit corrections (`locomo_bench/
corrected_gt.py`) only ever patch the *golden answer text* used for QA
judging (Stage C, not built here). The one error category that could touch
*evidence pointers* -- WRONG_CITATION -- is explicitly left unpatched
(corrected_gt.py's own docstring: "WRONG_CITATION errors don't affect
answer scoring and are left as-is"). So Stage A's evidence-based retrieval
GT needs no correction layer; `gt_answer` below is carried through only as
debug context, not as a scored field.
"""

import json
import re
import sqlite3
from pathlib import Path

from locomo_bench.ingest import extract_questions, extract_turns, load_locomo


def resolve_dia_map(
    data_dir: Path, conv_idx: int, conv_turns: list[dict],
) -> dict[tuple[int, str], str]:
    """(conv_id, dia_id) -> memory_id.

    Prefers a persisted `<data_dir>/dia_map.json` (exact -- written at
    ingest time). Falls back to content-text matching against the DB.

    DELIBERATE DIVERGENCE from `scripts/locomo_bench/run.py`'s
    `_rebuild_dia_map_from_turns`: that helper only matches
    `status = 'active'` rows. On a POST-SLEEP store that is the wrong
    filter for GT construction -- an evidence turn that REM dormant-ed or
    consolidate archived still has a stable memory_id (only its `status`
    changed); the (conv_id, dia_id) -> memory_id identity is a fact about
    ingest, not about current retrievability. Filtering by status here
    would silently drop exactly the questions sleep touched (their
    evidence set becomes empty, so eval_locomo-style callers skip the
    question entirely) -- which would bias any pre/post-sleep comparison
    toward *under-counting* sleep's effect on evidence turns it acted on.
    Retrievability is correctly handled downstream instead, by
    `store.load_token_map` (active-only) and by `impl_recall` itself (only
    returns active memories) -- so an archived evidence turn still scores
    0 for NDCG the honest way (it's unretrievable), not by being deleted
    from the question's evidence set entirely.
    """
    dia_map_path = Path(data_dir) / "dia_map.json"
    if dia_map_path.exists():
        raw = json.loads(dia_map_path.read_text())
        out = {}
        for k, v in raw.items():
            conv_s, dia_id = k.split(":", 1)
            out[(int(conv_s), dia_id)] = v
        return out

    db_path = Path(data_dir) / "memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    content_to_id: dict[str, str] = {}
    for row in conn.execute(
        "SELECT id, content FROM memories WHERE source != 'extraction'"
    ):
        content_to_id[row["content"]] = row["id"]
    conn.close()

    dia_map: dict[tuple[int, str], str] = {}
    unmatched = 0
    for turn in conv_turns:
        content = f"[{turn['speaker']}] {turn['text']}"
        mem_id = content_to_id.get(content)
        if mem_id:
            dia_map[(turn["conv_id"], turn["dia_id"])] = mem_id
        else:
            unmatched += 1
    if unmatched:
        print(
            f"[gt] WARNING: {unmatched}/{len(conv_turns)} turns had no "
            f"content-text match in {db_path} (no dia_map.json found; used "
            "fallback content-match reconstruction). Likely turns whose "
            "content sleep rewrote/merged, or a genuine content collision. "
            "Persisting the recovered map for reuse on the next run."
        )
    dia_map_path.write_text(
        json.dumps({f"{k[0]}:{k[1]}": v for k, v in dia_map.items()})
    )
    return dia_map


def build_query_set(
    locomo_data: Path, data_dir: Path, conv_idx: int,
) -> list[dict]:
    """[{query_id, question, category, gt_answer, evidence_ids}], skipping
    questions with no resolvable evidence (matches eval_locomo's `if not
    evidence_short: continue`). `query_id` is `"{conv_idx}:{qa_idx}"` --
    stable across reruns and forks of the same conversation, and the join
    key a later Stage B/C would use.
    """
    conversations = load_locomo(Path(locomo_data))
    all_turns = extract_turns(conversations)
    all_questions = extract_questions(conversations)
    conv_turns = [t for t in all_turns if t["conv_id"] == conv_idx]
    conv_questions = [q for q in all_questions if q["conv_id"] == conv_idx]

    dia_map = resolve_dia_map(data_dir, conv_idx, conv_turns)

    out = []
    for q in conv_questions:
        evidence_ids = set()
        for eid in q.get("evidence", []):
            mem_id = dia_map.get((q["conv_id"], eid))
            if mem_id:
                evidence_ids.add(mem_id)
        if not evidence_ids:
            continue
        out.append({
            "query_id": f"{conv_idx}:{q['qa_idx']}",
            "question": q["question"],
            "category": q["category"],
            "gt_answer": str(q["answer"]),
            "evidence_ids": sorted(evidence_ids),
        })
    return out


_SESSION_RE = re.compile(r"^session-\d+$")


def build_session_map(data_dir: Path) -> dict[str, str]:
    """memory_id -> 'session-N', read from each memory's `themes` column.

    LoCoMo ingest (`locomo_bench/ingest.py`) stamps every turn with a
    `f"session_{turn['session']}"` theme at write time, but `_insert_memory`
    routes themes through `memory.themes.normalize_themes`, which
    auto-hyphenates underscore variants -- so what actually lands in the DB
    is `session-N` (verified against a live ingested store: `["Gina",
    "session-1"]`), not `session_N`. This IS the turn/session mapping the
    LongMemEval effective_k correction needs; no separate join is required,
    but the hyphen is a real gotcha, not a typo -- match it exactly.
    Memories with no session theme (e.g. a REM summary spanning several
    sessions) are simply absent; `metrics.effective_k` treats an unmapped
    id as its own singleton "session" (conservative -- it can't inflate
    coverage).
    """
    db_path = Path(data_dir) / "memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    session_map: dict[str, str] = {}
    for row in conn.execute(
        "SELECT id, themes FROM memories WHERE themes IS NOT NULL"
    ):
        try:
            themes = json.loads(row["themes"])
        except (json.JSONDecodeError, TypeError):
            continue
        for t in themes:
            if isinstance(t, str) and _SESSION_RE.match(t):
                session_map[row["id"]] = t
                break
    conn.close()
    return session_map
