"""
Phase 0 born-saturation precheck — experiments/sleep-bench/precheck.py

Spec: docs/proposals/sleep-pipeline-optimization.md §3.2, §3.8.

Fresh-ingests one LoCoMo conversation into an isolated benchmark store, then
runs the sleep pipeline stage by stage (NREM deep -> REM -> consolidate
--dry-run), counting what each stage actually did. The acceptance gate answers
one question before any of Phase 1-3's pre/post-sleep measurement is trusted:
does a fresh ingest give sleep real work, or is the corpus born saturated the
other way (too clean for NREM/consolidate to find anything)?

    PASS iff (NREM edges created > ~100) AND (consolidate --dry-run proposes
    > 0 archives/merges).

SAFETY — read this before running anything:
    The live store (~/.claude/data) is under a hard no-sleep constraint until
    ~2026-07-15 (an abstinence experiment is accumulating unconsolidated
    memories there). This script NEVER touches it. The isolation guard below
    is the first thing that runs, before any ingest or subprocess call, and
    it hard-exits if the resolved data dir is the live store, is nested
    inside/around it, or is the bare `~/.somnigraph` root with no explicit
    subdir. Every sleep-stage subprocess this script launches is given
    SOMNIGRAPH_DATA_DIR explicitly pointed at the isolated store — nothing
    here ever relies on an ambient/inherited data dir.

Usage:
    uv run experiments/sleep-bench/precheck.py --conv 0
    uv run experiments/sleep-bench/precheck.py --conv 0 --fresh
    uv run experiments/sleep-bench/precheck.py --conv 0 --stop-after ingest
    uv run experiments/sleep-bench/precheck.py --conv 0 --stop-after nrem --nrem-limit 5

Stages (each clearly delimited so they can be run in one shot or resumed
stage-by-stage across separate invocations — the store is idempotent: ingest
is skipped if memory.db already exists, unless --fresh):
    ingest      -> fresh LoCoMo conversation into the isolated store (cheap)
    nrem        -> uv run scripts/sleep_nrem.py <standard|deep> [--limit N]  (Sonnet, $$)
    rem         -> uv run scripts/sleep_rem.py [--limit N]                  (Opus, $$$)
    consolidate -> uv run scripts/sleep_consolidate.py --dry-run -v         (Opus, $$ *)

    * consolidate's LLM judge only runs per cluster found by gather_clusters(),
      which requires topic_id (set by REM's taxonomy step). Before REM has
      run, gather_clusters() always returns zero clusters, so --dry-run before
      REM costs zero LLM calls — safe to smoke-test any time pre-REM.
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_BENCH_ROOT = Path.home() / ".somnigraph" / "benchmark" / "sleepbench-precheck"
LIVE_DATA_DIR = Path.home() / ".claude" / "data"
BARE_SOMNIGRAPH_ROOT = Path.home() / ".somnigraph"

RESULTS_PATH_DEFAULT = Path(__file__).resolve().parent / "precheck_results.json"


# ---------------------------------------------------------------------------
# SAFETY GUARD — must run before anything else touches a store.
# ---------------------------------------------------------------------------


def assert_isolated_data_dir(data_dir: Path) -> None:
    """Hard-exit if data_dir is (or contains, or is contained by) the live
    store, or is the bare ~/.somnigraph root. This is a mechanism, not a
    comment — it runs unconditionally as the first thing precheck.py does,
    and fork.py imports and reuses it for every fork/reset call.
    """
    resolved = data_dir.resolve()
    live = LIVE_DATA_DIR.resolve()
    bare = BARE_SOMNIGRAPH_ROOT.resolve()

    def _bail(reason: str) -> None:
        banner = "!" * 78
        sys.stderr.write(
            f"\n{banner}\n"
            f"REFUSING TO RUN -- sleep-bench isolation guard tripped\n"
            f"  target data dir : {resolved}\n"
            f"  reason          : {reason}\n"
            f"  live store      : {live}\n"
            f"                    (HARD no-sleep constraint until ~2026-07-15 --\n"
            f"                     an abstinence experiment is accumulating there)\n"
            f"This harness only operates on a dedicated, explicit isolated benchmark\n"
            f"store. Pass --data-dir pointing at an isolated subdir, e.g.:\n"
            f"  {DEFAULT_BENCH_ROOT / 'conv0'}\n"
            f"{banner}\n"
        )
        sys.exit(1)

    nested_in_live = False
    try:
        resolved.relative_to(live)
        nested_in_live = True
    except ValueError:
        pass

    contains_live = False
    try:
        live.relative_to(resolved)
        contains_live = True
    except ValueError:
        pass

    if nested_in_live or contains_live:
        _bail(f"equals or is nested with the live store ({live})")

    if resolved == bare:
        _bail(
            "is the bare default ~/.somnigraph root -- an explicit isolated "
            "subdir is required (e.g. .../benchmark/sleepbench-precheck/conv0)"
        )

    print(f"[guard] isolation OK -- target is not the live store: {resolved}")


# ---------------------------------------------------------------------------
# Subprocess env — every sleep-stage call gets an explicit, pinned env.
# ---------------------------------------------------------------------------


def resolve_openai_key() -> str | None:
    """Mirror locomo_bench.run.setup_isolated_db's key lookup so subprocess
    sleep scripts (which read DATA_DIR/openai_api_key if OPENAI_API_KEY is
    unset) don't fail just because the isolated store has no key file."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    for key_path in [
        Path.home() / ".somnigraph" / "openai_api_key",
        Path.home() / ".claude" / "secrets" / "openai_api_key",
        Path.home() / ".claude" / "data" / "openai_api_key",
    ]:
        if key_path.exists():
            return key_path.read_text().strip()
    return None


def build_subprocess_env(conv_dir: Path) -> dict:
    """SOMNIGRAPH_DATA_DIR pinned to the isolated store, PYTHONUTF8=1 (sleep
    scripts print box-drawing chars; cp1252 crashes without it on Windows),
    SOMNIGRAPH_EMBEDDING_BACKEND left untouched (unset on this machine ->
    OpenAI 1536d, matching the live store's backend)."""
    env = os.environ.copy()
    env["SOMNIGRAPH_DATA_DIR"] = str(conv_dir)
    env["PYTHONUTF8"] = "1"
    key = resolve_openai_key()
    if key:
        env["OPENAI_API_KEY"] = key
    return env


def run_sleep_script(script_name: str, script_args: list[str], conv_dir: Path) -> str:
    """Run a sleep-stage script as `uv run scripts/<script_name> <args>` with
    an explicit isolated env. Streams output live (both to our stdout and
    into the returned capture buffer, for report-parsing) and raises on
    non-zero exit."""
    script_path = SCRIPTS_DIR / script_name
    cmd = ["uv", "run", str(script_path)] + script_args
    env = build_subprocess_env(conv_dir)
    print(
        f"  $ SOMNIGRAPH_DATA_DIR={conv_dir} PYTHONUTF8=1 "
        f"uv run scripts/{script_name} {' '.join(script_args)}"
    )
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    captured = []
    for line in proc.stdout:
        print(line, end="")
        captured.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{script_name} exited with code {proc.returncode}")
    return "".join(captured)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def do_ingest(conv_dir: Path, conv_idx: int, locomo_data: Path, embed_cache: Path | None) -> dict:
    """Fresh-ingest one LoCoMo conversation via the existing locomo_bench
    helpers (setup_isolated_db + ingest_conversation) -- not reimplemented."""
    from locomo_bench.run import setup_isolated_db
    from locomo_bench.ingest import load_locomo, extract_turns, ingest_conversation

    setup_isolated_db(conv_dir)

    conversations = load_locomo(locomo_data)
    all_turns = extract_turns(conversations)
    conv_turns = [t for t in all_turns if t["conv_id"] == conv_idx]
    if not conv_turns:
        sys.exit(f"[ingest] ERROR: no turns found for conversation {conv_idx} in {locomo_data}")

    print(f"[ingest] conversation {conv_idx}: {len(conv_turns)} turns -> {conv_dir}")
    dia_map = ingest_conversation(conv_turns, embed_cache_path=embed_cache, enrich=False)
    print(f"[ingest] done: {len(dia_map)} memories inserted")
    return dia_map


# ---------------------------------------------------------------------------
# Store-state counting (direct sqlite3 -- independent of any in-process
# memory.* monkeypatching, correct whether the DB was touched in-process or
# by a subprocess).
# ---------------------------------------------------------------------------


def count_store_state(db_path: Path) -> dict:
    conn = sqlite3.connect(str(db_path))
    try:
        def scalar(query: str) -> int:
            return conn.execute(query).fetchone()[0]

        return {
            "active_memories": scalar("SELECT COUNT(*) FROM memories WHERE status='active'"),
            "total_memories": scalar("SELECT COUNT(*) FROM memories"),
            "deleted_memories": scalar("SELECT COUNT(*) FROM memories WHERE status='deleted'"),
            "dormant_memories": scalar("SELECT COUNT(*) FROM memories WHERE status='dormant'"),
            "active_details": scalar(
                "SELECT COUNT(*) FROM memories WHERE status='active' AND layer='detail'"
            ),
            "active_summaries": scalar(
                "SELECT COUNT(*) FROM memories WHERE status='active' AND layer='summary'"
            ),
            "topics_assigned": scalar(
                "SELECT COUNT(DISTINCT topic_id) FROM memories "
                "WHERE status='active' AND layer='detail' AND topic_id IS NOT NULL"
            ),
            "edges_total": scalar("SELECT COUNT(*) FROM memory_edges"),
            "edges_entity": scalar(
                "SELECT COUNT(*) FROM memory_edges WHERE linking_context LIKE 'Entity auto-link%'"
            ),
            "edges_derived_from": scalar(
                "SELECT COUNT(*) FROM memory_edges WHERE edge_type='derived_from'"
            ),
        }
    finally:
        conn.close()


def diff_snapshots(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in after if k in before}


def print_snapshot(label: str, snap: dict) -> None:
    print(f"\n[state] {label}:")
    for k, v in snap.items():
        print(f"    {k}: {v}")


# ---------------------------------------------------------------------------
# Report parsing — the "- key: value" report blocks each sleep script prints.
# For NREM/REM this is a cross-check against the DB-delta ground truth. For
# consolidate --dry-run it is the ONLY source: dry-run never mutates the DB
# by construction (execute_actions only calls archive_memory/rewrite_memory
# etc. when `not dry_run`), so the "proposed merges/archives" the acceptance
# gate needs only exist in the printed report.
# ---------------------------------------------------------------------------

_KV_RE = re.compile(r"^- ([^:\n]+):\s*(.+)$", re.MULTILINE)


def parse_kv_report(stdout: str) -> dict:
    return {m.group(1).strip(): m.group(2).strip() for m in _KV_RE.finditer(stdout)}


def leading_int(value: str | None, default: int = 0) -> int:
    if not value:
        return default
    m = re.match(r"-?\d+", value.strip())
    return int(m.group()) if m else default


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------


def write_results(results: dict, results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[results] wrote {results_path}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 0 born-saturation precheck for the sleep-bench harness"
    )
    p.add_argument("--conv", type=int, default=0, help="LoCoMo conversation index (default: 0)")
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override the isolated store dir (default: "
        "~/.somnigraph/benchmark/sleepbench-precheck/conv<N>)",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="rmtree the isolated store and re-ingest even if it already exists",
    )
    p.add_argument(
        "--stop-after",
        choices=["ingest", "nrem", "rem", "consolidate"],
        default="consolidate",
        help="Stop after this stage (default: run the full precheck through "
        "consolidate --dry-run and print the gate verdict)",
    )
    p.add_argument("--nrem-mode", choices=["standard", "deep"], default="deep")
    p.add_argument(
        "--nrem-limit",
        type=int,
        default=0,
        help="Cap memories gathered by NREM (0 = no limit; pass a small number "
        "e.g. 5 for a cheap wiring smoke test)",
    )
    p.add_argument(
        "--rem-limit", type=int, default=0, help="Cap clusters processed by REM (0 = no limit)"
    )
    p.add_argument(
        "--consolidate-limit",
        type=int,
        default=0,
        help="Cap clusters processed by consolidate (0 = no limit)",
    )
    p.add_argument(
        "--locomo-data",
        type=str,
        default=None,
        help="Path to locomo10.json (default: locomo_bench.config.BenchConfig default)",
    )
    p.add_argument(
        "--embed-cache",
        type=str,
        default=None,
        help="Path to the shared embedding cache (default: BenchConfig default)",
    )
    p.add_argument(
        "--results-path",
        type=str,
        default=None,
        help=f"Where to write JSON results (default: {RESULTS_PATH_DEFAULT})",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    from locomo_bench.config import BenchConfig

    bench_defaults = BenchConfig()
    locomo_data = Path(args.locomo_data) if args.locomo_data else Path(bench_defaults.locomo_data)
    embed_cache = Path(args.embed_cache) if args.embed_cache else Path(bench_defaults.embed_cache)
    results_path = Path(args.results_path).expanduser() if args.results_path else RESULTS_PATH_DEFAULT

    conv_dir = (
        Path(args.data_dir).expanduser()
        if args.data_dir
        else DEFAULT_BENCH_ROOT / f"conv{args.conv}"
    )

    # --- SAFETY GUARD: first thing, before any I/O on the target store ---
    assert_isolated_data_dir(conv_dir)

    print(f"[precheck] conversation {args.conv}")
    print(f"[precheck] isolated store: {conv_dir}")

    if args.fresh and conv_dir.exists():
        print(f"[fresh] removing existing store: {conv_dir}")
        shutil.rmtree(conv_dir)

    db_path = conv_dir / "memory.db"
    results: dict = {
        "conv": args.conv,
        "data_dir": str(conv_dir),
        "started_at": now_iso(),
        "stages": {},
        "counts": {},
    }

    # ---- Stage: ingest ----
    if not db_path.exists():
        dia_map = do_ingest(conv_dir, args.conv, locomo_data, embed_cache)
        results["stages"]["ingest"] = {"fresh": True, "memories_inserted": len(dia_map)}
    else:
        print(f"[ingest] reusing existing store ({db_path}) -- pass --fresh to rebuild")
        results["stages"]["ingest"] = {"fresh": False, "reused": True}

    snap_pre_sleep = count_store_state(db_path)
    results["counts"]["pre_sleep"] = snap_pre_sleep
    print_snapshot("pre-sleep (post-ingest)", snap_pre_sleep)

    if args.stop_after == "ingest":
        write_results(results, results_path)
        return

    # ---- Stage: NREM ----
    nrem_args = [args.nrem_mode]
    if args.nrem_limit:
        nrem_args += ["--limit", str(args.nrem_limit)]
    print(f"\n[nrem] running sleep_nrem.py {' '.join(nrem_args)} ...")
    nrem_stdout = run_sleep_script("sleep_nrem.py", nrem_args, conv_dir)

    snap_post_nrem = count_store_state(db_path)
    results["counts"]["post_nrem"] = snap_post_nrem
    print_snapshot("post-NREM", snap_post_nrem)

    nrem_delta = diff_snapshots(snap_pre_sleep, snap_post_nrem)
    results["stages"]["nrem"] = {
        "args": nrem_args,
        "reported": parse_kv_report(nrem_stdout),
        "db_delta": nrem_delta,
    }

    if args.stop_after == "nrem":
        write_results(results, results_path)
        return

    # ---- Stage: REM ----
    rem_args = []
    if args.rem_limit:
        rem_args += ["--limit", str(args.rem_limit)]
    print(f"\n[rem] running sleep_rem.py {' '.join(rem_args)} ...")
    rem_stdout = run_sleep_script("sleep_rem.py", rem_args, conv_dir)

    snap_post_rem = count_store_state(db_path)
    results["counts"]["post_rem"] = snap_post_rem
    print_snapshot("post-REM", snap_post_rem)

    clusters_match = re.search(r"Found (\d+) topic clusters", rem_stdout)
    results["stages"]["rem"] = {
        "args": rem_args,
        "reported": parse_kv_report(rem_stdout),
        "topic_clusters_found": int(clusters_match.group(1)) if clusters_match else None,
        "db_delta": diff_snapshots(snap_post_nrem, snap_post_rem),
    }

    if args.stop_after == "rem":
        write_results(results, results_path)
        return

    # ---- Stage: consolidate (dry-run, -v) ----
    consolidate_args = ["--dry-run", "-v"]
    if args.consolidate_limit:
        consolidate_args += ["--limit", str(args.consolidate_limit)]
    print(f"\n[consolidate] running sleep_consolidate.py {' '.join(consolidate_args)} ...")
    consolidate_stdout = run_sleep_script("sleep_consolidate.py", consolidate_args, conv_dir)

    # dry-run must not mutate the store; this snapshot should equal post_rem.
    snap_post_consolidate = count_store_state(db_path)
    results["counts"]["post_consolidate_dry_run"] = snap_post_consolidate

    consolidate_report = parse_kv_report(consolidate_stdout)
    clusters_pre = re.search(
        r"Found (\d+) clusters \((\d+) total memories\)", consolidate_stdout
    )
    archived = leading_int(consolidate_report.get("Archived"))
    merged = leading_int(consolidate_report.get("Merged"))

    results["stages"]["consolidate"] = {
        "args": consolidate_args,
        "reported": consolidate_report,
        "clusters_found": int(clusters_pre.group(1)) if clusters_pre else 0,
        "cluster_total_memories": int(clusters_pre.group(2)) if clusters_pre else 0,
        "archived_proposed": archived,
        "merged_proposed": merged,
        "db_delta_should_be_zero": diff_snapshots(snap_post_rem, snap_post_consolidate),
    }

    # ---- Acceptance gate (§3.2) ----
    nrem_edges_created = nrem_delta.get("edges_total", 0)
    proposed = archived + merged
    gate = {
        "nrem_edges_created": nrem_edges_created,
        "nrem_edges_gt_100": nrem_edges_created > 100,
        "consolidate_archived_proposed": archived,
        "consolidate_merged_proposed": merged,
        "consolidate_proposes_gt_0": proposed > 0,
    }
    gate["pass"] = gate["nrem_edges_gt_100"] and gate["consolidate_proposes_gt_0"]
    results["gate"] = gate
    results["completed_at"] = now_iso()

    write_results(results, results_path)

    verdict = "PASS" if gate["pass"] else "FAIL"
    print("\n" + "=" * 78)
    print(f"ACCEPTANCE GATE: {verdict}")
    print(
        f"  NREM edges created       : {gate['nrem_edges_created']} "
        f"(> 100 required) -> {gate['nrem_edges_gt_100']}"
    )
    print(f"  Consolidate archived     : {gate['consolidate_archived_proposed']}")
    print(f"  Consolidate merged       : {gate['consolidate_merged_proposed']}")
    print(f"  Consolidate proposes > 0 : {gate['consolidate_proposes_gt_0']}")
    print("=" * 78)
    if not gate["pass"]:
        print(
            "\nGate FAILED -- see docs/proposals/sleep-pipeline-optimization.md §3.2 "
            "for the two remediation knobs (finer ingest granularity, then "
            "controlled duplicate injection as a last resort)."
        )


if __name__ == "__main__":
    main()
