"""Store-pointing + ID-resolution helpers for the Stage A retrieval rig.

Deliberately NOT reusing scripts/locomo_bench/run.py's `setup_isolated_db` --
that helper also clears `locomo_bench.eval_retrieval._conv_cache`, i.e. it is
already wired to the eval harness this rig is designed to be independent of.
This is a small (~20 line), from-scratch equivalent: point memory.constants /
memory.db / memory.reranker at an isolated data dir and reset the caches that
depend on the old path. Same idea, no dependency.
"""

from pathlib import Path

_KEY_LOCATIONS = [
    Path.home() / ".somnigraph" / "openai_api_key",
    Path.home() / ".claude" / "secrets" / "openai_api_key",
    Path.home() / ".claude" / "data" / "openai_api_key",
]


def resolve_openai_key() -> str | None:
    import os
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    for key_path in _KEY_LOCATIONS:
        if key_path.exists():
            return key_path.read_text().strip()
    return None


def point_at_store(data_dir: Path) -> None:
    """Point the memory.* modules at an existing isolated store.

    Must be called once per process before any `memory.db` / `memory.tools`
    calls. Does NOT touch MODEL_PATH -- that's scorers.apply_scorer's job
    (the scorer axis is orthogonal to which store we're pointed at).
    """
    import os

    import memory.constants
    import memory.db
    import memory.embeddings
    import memory.reranker

    data_dir = Path(data_dir)
    if not (data_dir / "memory.db").exists():
        raise FileNotFoundError(
            f"no memory.db at {data_dir} -- this rig reads an existing store, "
            "it does not ingest. Build the store first (precheck.py, fork.py, "
            "or this rig's own validation ingest helper)."
        )

    key = resolve_openai_key()
    if key:
        os.environ["OPENAI_API_KEY"] = key

    memory.constants.DATA_DIR = data_dir
    memory.db.DB_PATH = data_dir / "memory.db"
    memory.db._schema_initialized = False  # idempotent re-check, cheap
    memory.embeddings._openai_client = None  # pick up the (possibly new) key
    memory.reranker._cache = {
        "model": None, "feature_names": None, "memory_meta": None,
        "meta_mem_count": -1, "loaded": False, "failed": False,
    }


def resolve_full_ids(data_dir: Path, short_ids: list[str]) -> dict[str, str]:
    """short (8-char) memory id -> full memory id, via a direct sqlite3 read.

    Independent of any in-process memory.db connection/cache -- correct
    whether the store was touched by this process or a prior subprocess.
    """
    import sqlite3

    if not short_ids:
        return {}
    conn = sqlite3.connect(str(Path(data_dir) / "memory.db"))
    try:
        out = {}
        for short in short_ids:
            row = conn.execute(
                "SELECT id FROM memories WHERE id LIKE ?", (short + "%",)
            ).fetchone()
            if row:
                out[short] = row[0]
        return out
    finally:
        conn.close()


def edge_embedding_health(data_dir: Path) -> dict:
    """Count contextful edges that have no linking_embedding.

    NOTE (corrected 2026-07-04): this is an edge-HYGIENE signal, NOT a
    scoring-validity gate. It was originally believed that unembedded contextful
    edges are invisible to PPR via a context-relevance gate
    (`if has_context and not edge["linking_embedding"]: continue`), so a store
    full of them would read as zero-edge -- a false null. That gate exists ONLY
    in scoring._expand_adjacency_legacy, which NOTHING in src/ calls. The actual
    live + bench expansion path -- impl_recall(expand=True) -> scoring.
    expand_via_ppr -- SELECTs only (source_id, target_id, weight, flags) and
    traverses every non-contradiction edge by WEIGHT; it never reads
    linking_embedding (verified: scoring.py:316-320 + full fn to 395;
    tools.py:748-751). So an unembedded edge still contributes to PPR -- there
    is no false null. REM (sleep_rem.py) creates contextful edges without
    embedding them; NREM (sleep_nrem.py) embeds. The blind count reflects that
    REM/NREM asymmetry, a cosmetic inconsistency, not a scoring hazard.

    Returns counts, not a verdict.
    """
    import sqlite3

    conn = sqlite3.connect(str(Path(data_dir) / "memory.db"))
    try:
        def scalar(q: str) -> int:
            return conn.execute(q).fetchone()[0]

        total = scalar("SELECT COUNT(*) FROM memory_edges")
        contextful = scalar(
            "SELECT COUNT(*) FROM memory_edges "
            "WHERE linking_context IS NOT NULL AND linking_context != ''"
        )
        blind = scalar(
            "SELECT COUNT(*) FROM memory_edges "
            "WHERE linking_context IS NOT NULL AND linking_context != '' "
            "AND linking_embedding IS NULL"
        )
        return {
            "total_edges": total,
            "contextful_edges": contextful,
            "blind_edges": blind,  # contextful but unembedded -> invisible to PPR
            "blind_frac": (blind / contextful) if contextful else 0.0,
        }
    finally:
        conn.close()


def load_token_map(data_dir: Path) -> dict[str, int]:
    """id -> token_count, for currently-active memories only.

    Matches eval_retrieval.py's `eval_production` token_map convention: a
    memory dormancy/consolidate archived is correctly absent here (and thus
    unretrievable / contributes 0 to NDCG) -- that's intended behavior, not
    a bug (see docs/proposals/sleep-pipeline-optimization.md §3.6).
    """
    import sqlite3

    conn = sqlite3.connect(str(Path(data_dir) / "memory.db"))
    try:
        rows = conn.execute(
            "SELECT id, token_count FROM memories WHERE status = 'active'"
        ).fetchall()
        return {r[0]: (r[1] or 100) for r in rows}
    finally:
        conn.close()
