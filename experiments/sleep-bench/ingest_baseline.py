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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conv", type=int, required=True, help="LoCoMo conversation index")
    ap.add_argument("--data-dir", required=True, help="isolated store to create (must not exist yet)")
    ap.add_argument("--locomo-data", default=str(DEFAULT_LOCOMO_DATA))
    ap.add_argument("--embed-cache", default=str(DEFAULT_EMBED_CACHE))
    args = ap.parse_args()

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
    conv_turns = [t for t in all_turns if t["conv_id"] == args.conv]
    if not conv_turns:
        sys.exit(f"[ingest-baseline] no turns found for conversation {args.conv}")

    print(f"[ingest-baseline] conversation {args.conv}: {len(conv_turns)} turns -> {data_dir}")
    dia_map = ingest_conversation(
        conv_turns, embed_cache_path=Path(args.embed_cache), enrich=False
    )
    print(f"[ingest-baseline] {len(dia_map)} memories inserted")

    dia_map_path = data_dir / "dia_map.json"
    dia_map_path.write_text(json.dumps({f"{k[0]}:{k[1]}": v for k, v in dia_map.items()}))
    print(f"[ingest-baseline] wrote {dia_map_path}")


if __name__ == "__main__":
    main()
