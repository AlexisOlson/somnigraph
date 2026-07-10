#!/usr/bin/env python3
"""Fresh-ingest one LoCoMo conversation into an isolated store, persisting
`dia_map.json` alongside `memory.db` -- experiments/sleep-bench/ingest_baseline.py

`precheck.py` already fresh-ingests (via the same locomo_bench helpers) but
never persists `dia_map.json`, so a store built by precheck.py alone falls
back to `gt.py`'s content-text reconstruction on first Stage A use (it
works, but it's a fallback, not free, and content-text matching degrades
once sleep rewrites/merges turn content). This script is the "build me a
rig-ready baseline" entry point: ingest + `dia_map.json` in one step, so
`retrieve.py`'s `gt.resolve_dia_map` hits the fast, exact path immediately
-- and every fork of this baseline (via fork.py's copytree) carries the
exact map with it.

Reuses only pure data-loading helpers (`locomo_bench.ingest.
{load_locomo,extract_turns,ingest_conversation}`) -- no eval/scoring logic.

Usage:
    uv run experiments/sleep-bench/ingest_baseline.py --conv 1 \
        --data-dir ~/.somnigraph/benchmark/sleepbench-rig-validate/conv1

Whole-corpus store (all conversations in ONE combined store, for corpus-scale
sleep experiments; dia_map keys are conv-scoped so the merged map stays exact):
    uv run experiments/sleep-bench/ingest_baseline.py --convs all \
        --data-dir ~/.somnigraph/benchmark/sleepbench-full/frozen
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from precheck import assert_isolated_data_dir  # reuse the isolation guard, don't duplicate it

DEFAULT_LOCOMO_DATA = Path.home() / ".claude" / "repos" / "locomo" / "data" / "locomo10.json"
DEFAULT_EMBED_CACHE = Path.home() / ".claude" / "data" / "bench_locomo_embeddings.pkl"


def _point_at_fresh_store(data_dir: Path) -> None:
    import os

    import memory.constants
    import memory.db
    import memory.embeddings
    import memory.reranker

    from store import resolve_openai_key

    data_dir.mkdir(parents=True, exist_ok=True)
    key = resolve_openai_key()
    if key:
        os.environ["OPENAI_API_KEY"] = key
    memory.constants.DATA_DIR = data_dir
    memory.db.DB_PATH = data_dir / "memory.db"
    memory.db._schema_initialized = False
    memory.embeddings._openai_client = None
    memory.reranker._cache = {
        "model": None, "feature_names": None, "memory_meta": None,
        "meta_mem_count": -1, "loaded": False, "failed": False,
    }


def _parse_convs(spec: str) -> list[int]:
    """'0,2,5' or '0-9' (inclusive) -> sorted unique conversation indices."""
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        elif part:
            ids.add(int(part))
    if not ids:
        sys.exit(f"[ingest-baseline] empty --convs spec: {spec!r}")
    return sorted(ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conv", type=int, help="single LoCoMo conversation index")
    ap.add_argument("--convs", help='multiple conversations into ONE combined store: "all", "0-9", or "0,2,5"')
    ap.add_argument("--data-dir", required=True, help="isolated store to create (must not exist yet)")
    ap.add_argument("--locomo-data", default=str(DEFAULT_LOCOMO_DATA))
    ap.add_argument("--embed-cache", default=str(DEFAULT_EMBED_CACHE))
    args = ap.parse_args()

    if (args.conv is None) == (args.convs is None):
        sys.exit("[ingest-baseline] pass exactly one of --conv or --convs")

    data_dir = Path(args.data_dir).expanduser().resolve()
    assert_isolated_data_dir(data_dir)

    if (data_dir / "memory.db").exists():
        sys.exit(
            f"[ingest-baseline] {data_dir}\\memory.db already exists -- remove it first "
            "(or use fork.py's reset) for a fresh ingest."
        )

    _point_at_fresh_store(data_dir)

    from locomo_bench.ingest import extract_turns, ingest_conversation, load_locomo

    conversations = load_locomo(Path(args.locomo_data))
    all_turns = extract_turns(conversations)

    if args.convs is not None:
        if args.convs.strip().lower() == "all":
            conv_ids = sorted({t["conv_id"] for t in all_turns})
        else:
            conv_ids = _parse_convs(args.convs)
    else:
        conv_ids = [args.conv]

    merged_map: dict = {}
    for cid in conv_ids:
        conv_turns = [t for t in all_turns if t["conv_id"] == cid]
        if not conv_turns:
            sys.exit(f"[ingest-baseline] no turns found for conversation {cid}")
        print(f"[ingest-baseline] conversation {cid}: {len(conv_turns)} turns -> {data_dir}")
        dia_map = ingest_conversation(
            conv_turns, embed_cache_path=Path(args.embed_cache), enrich=False
        )
        merged_map.update(dia_map)
        print(f"[ingest-baseline] conversation {cid}: {len(dia_map)} memories inserted")

    print(f"[ingest-baseline] total: {len(merged_map)} memories across {len(conv_ids)} conversation(s)")
    dia_map_path = data_dir / "dia_map.json"
    dia_map_path.write_text(json.dumps({f"{k[0]}:{k[1]}": v for k, v in merged_map.items()}))
    print(f"[ingest-baseline] wrote {dia_map_path}")


if __name__ == "__main__":
    main()
