#!/usr/bin/env python3
"""H7 summary-coverage judge -- experiments/sleep-bench/judge_summary_coverage.py

The LLM instrument that collapses the H7 bracket. impute_rescore.py brackets the
raw penalty with a provenance UPPER bound (a retrieved REM summary credits ALL
the evidence turns it was generated_from, even ones whose specific fact it
compressed away). This judge replaces provenance-coverage with JUDGED coverage:
for each (query, summary) pair where the summary was generated from >=1 of the
query's evidence turns, ask whether the summary actually CONTAINS that turn's
answer-relevant fact -- "a reader seeing only the summary has the same evidence."

Reuses scripts/locomo_bench/judge_synthetic_coverage.py's proven prompt,
batching (10 questions/call), Q{i}/S{j}: E.. response format, robust parser, and
resumable checkpointing VERBATIM. The only adaptation is the data source: that
judge scores synthetic nodes (source='extraction') via extracted_from edges on
the retrieval_{conv} benchmark layout; this one scores REM summaries
(layer='summary') via generated_from, sourcing queries/evidence from the
sleep-bench sweep JSONL + the rem arm store.

Output: results/summary_coverage.json = {query_id: {summary_id: [covered_evidence_ids]}}
Feed to: impute_rescore.py --judged-coverage results/summary_coverage.json

RUN IN TERMINAL (LLM; batched opus over ~all queries with a covering summary).

Usage:
    uv run experiments/sleep-bench/judge_summary_coverage.py \
        --rem-dir ~/.somnigraph/benchmark/sleepbench-arms/conv0-rem [--model opus] [--resume]
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
BATCH_SIZE = 10
DEFAULT_LOCOMO = Path.home() / ".claude" / "repos" / "locomo" / "data" / "locomo10.json"
DEFAULT_MODEL = "opus"  # Alexis: the load-bearing measurement runs on the gold judge
CATEGORIES = {1: "single-hop", 2: "temporal", 3: "multi-hop",
              4: "open-domain", 5: "adversarial"}

# --- Prompt + parser lifted from judge_synthetic_coverage.py (s/synthetic/summary/) ---

SYSTEM_PROMPT = """\
You are evaluating whether REM summary nodes contain equivalent information to \
original dialogue turns, in the context of answering a specific question.

For each summary node, determine which evidence turns (if any) it covers — \
meaning it contains the specific information from that turn needed to answer \
the question. A summary covers a turn if a reader seeing only the summary \
(not the original turn) would have the same evidence for answering the question.

Be strict: surface topic overlap is not enough. The summary must carry the \
specific facts, details, or statements from the turn that are relevant to the question."""

QUESTION_TEMPLATE = """\
--- Question {q_label} (category: {category}) ---
Question: {question}
Answer: {answer}

Evidence turns:
{evidence_block}

Summary nodes:
{summary_block}
"""

BATCH_FOOTER = """
For each summary in each question, list which evidence turn IDs it covers.
If a summary covers no evidence turns for that question, list "none".

Respond in exactly this format:
Q1/S1: E2, E3
Q1/S2: none
Q2/S1: E1
..."""


def load_store(rem_dir: Path):
    """Return (content_by_id, genfrom_by_summary) from the rem arm."""
    conn = sqlite3.connect(str(rem_dir / "memory.db"))
    try:
        content = {r[0]: (r[1] or "") for r in conn.execute("SELECT id, content FROM memories")}
        layer = {r[0]: r[1] for r in conn.execute("SELECT id, layer FROM memories")}
        genfrom = {}
        for sid, gf in conn.execute(
            "SELECT id, generated_from FROM memories WHERE layer='summary'"
        ):
            if not gf:
                continue
            members = json.loads(gf)
            if members and not any(layer.get(m) == "summary" for m in members):
                genfrom[sid] = set(members)
        return content, genfrom
    finally:
        conn.close()


def load_queries(results_dir: Path) -> dict:
    """query_id -> {question, answer, category, evidence_ids} from the full-summary
    sweep rung (rem_m3 has every query scored; the query set is rung-invariant)."""
    path = results_dir / "rem_m3__rrf.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run sweep_rem_floor.py first")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["query_id"]] = {
                "question": r["query"],
                "answer": str(r.get("gt_answer", "")),
                "category": r.get("category", 0),
                "evidence_ids": r["evidence_ids"],
            }
    return out


def load_queries_from_gt(locomo_data: Path, rem_dir: Path, convs_spec: str) -> dict:
    """query_id -> {question, answer, category, evidence_ids}, sourced directly
    from gt.build_query_set per conv (no Stage-A JSONL needed).

    Mirrors cross_judge_audit.py's sourcing exactly so the eligible-query and
    (query,summary)-pair counts match that audit. Evidence ids resolve against
    rem_dir's dia_map (the slept store's persisted map is exact for ingest-time
    identities -- gt.resolve_dia_map is status-agnostic by design). Imports are
    lazy so backward-compatible (non --from-gt) invocations never touch the
    gt/ingest_baseline modules.
    """
    for p in (str(HERE), str(HERE.parent.parent / "src"), str(HERE.parent.parent / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import gt as gt_mod
    from ingest_baseline import _parse_convs

    out = {}
    for conv in _parse_convs(convs_spec):
        for row in gt_mod.build_query_set(Path(locomo_data), rem_dir, conv):
            out[row["query_id"]] = {
                "question": row["question"],
                "answer": str(row["gt_answer"]),
                "category": row["category"],
                "evidence_ids": row["evidence_ids"],
            }
    return out


def build_question_block(qinfo, content, genfrom, q_label):
    """Build a prompt block for one query. Returns (block, evidence_labels,
    summary_labels) or None if no summary provenance-covers this query's evidence."""
    evidence = qinfo["evidence_ids"]
    if not evidence:
        return None
    evidence_set = set(evidence)

    # Candidate summaries: generated_from intersects this query's evidence.
    candidates = {sid: (gf & evidence_set) for sid, gf in genfrom.items() if gf & evidence_set}
    if not candidates:
        return None

    evidence_labels, evidence_lines = {}, []
    for i, eid in enumerate(evidence, 1):
        label = f"E{i}"
        evidence_labels[label] = eid
        c = content.get(eid, "[missing]")
        evidence_lines.append(f"  {label} ({eid[:8]}): {c[:500]}")

    summary_labels, summary_lines = {}, []
    for i, sid in enumerate(candidates, 1):
        label = f"S{i}"
        summary_labels[label] = sid
        c = content.get(sid, "[missing]")
        summary_lines.append(f"  {label}: {c[:500]}")

    block = QUESTION_TEMPLATE.format(
        q_label=q_label,
        category=CATEGORIES.get(qinfo["category"], "unknown"),
        question=qinfo["question"],
        answer=qinfo["answer"],
        evidence_block="\n".join(evidence_lines),
        summary_block="\n".join(summary_lines),
    )
    return block, evidence_labels, summary_labels


def parse_batch_response(response: str, question_map: dict) -> dict:
    """{qkey: {summary_id: [covered_evidence_ids]}} from Q{i}/S{j}: E.. lines."""
    results = {}
    for line in response.strip().splitlines():
        line = line.strip()
        m = re.match(r"Q(\d+)/S(\d+)\s*:\s*(.*)", line, re.IGNORECASE)
        if not m:
            continue
        q_label, s_label = f"Q{m.group(1)}", f"S{m.group(2)}"
        covered_str = m.group(3).strip().lower()
        if q_label not in question_map:
            continue
        qinfo = question_map[q_label]
        qkey = qinfo["qkey"]
        results.setdefault(qkey, {})
        sid = qinfo["summary_labels"].get(s_label)
        if not sid:
            continue
        if covered_str == "none" or not covered_str:
            results[qkey][sid] = []
            continue
        covered = []
        for tok in re.findall(r"E\d+", covered_str, re.IGNORECASE):
            eid = qinfo["evidence_labels"].get(tok.upper())
            if eid:
                covered.append(eid)
        results[qkey][sid] = covered
    return results


def call_claude(system_prompt: str, user_prompt: str, model: str) -> str:
    """claude -p subprocess, 3 retries with backoff. Matches the proven judge."""
    prompt = (system_prompt + "\n\n" if system_prompt else "") + user_prompt
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    claude_cmd = "claude.cmd" if sys.platform == "win32" else "claude"
    err = "unknown"
    for attempt in range(3):
        try:
            result = subprocess.run(
                [claude_cmd, "-p", "--model", model],
                input=prompt, capture_output=True, timeout=300,
                env=env, encoding="utf-8", errors="replace",
            )
            if result.returncode == 0:
                return result.stdout.strip()
            err = result.stderr.strip()
        except subprocess.TimeoutExpired:
            err = "timeout"
        if attempt < 2:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Claude call failed after 3 attempts: {err}")


def main() -> None:
    ap = argparse.ArgumentParser(description="H7 summary-coverage judge")
    ap.add_argument("--rem-dir", required=True)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="count pairs, no LLM")
    ap.add_argument("--from-gt", action="store_true",
                    help="source queries from gt.build_query_set per conv (train-conv "
                         "coverage judging) instead of the Stage-A sweep JSONL")
    ap.add_argument("--convs", default="1-9",
                    help="conversations for --from-gt sourcing (e.g. '1-9', '0,2,5')")
    ap.add_argument("--locomo-data", default=str(DEFAULT_LOCOMO),
                    help="LoCoMo dataset json (only used with --from-gt)")
    ap.add_argument("--out", default=None,
                    help="output json path (default: <results-dir>/summary_coverage.json)")
    args = ap.parse_args()

    rem = Path(args.rem_dir).expanduser().resolve()
    rdir = Path(args.results_dir).expanduser()
    rdir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out).expanduser() if args.out else rdir / "summary_coverage.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content, genfrom = load_store(rem)
    if args.from_gt:
        queries = load_queries_from_gt(Path(args.locomo_data).expanduser(), rem, args.convs)
    else:
        queries = load_queries(rdir)
    print(f"[judge] {len(genfrom)} leaf summaries, {len(queries)} queries, model={args.model}")

    results = {}
    if args.resume and out_path.exists():
        results = json.loads(out_path.read_text())
        print(f"[judge] resumed {len(results)} prior judgments")

    # Queries with >=1 provenance-covering summary (only those can earn credit).
    pending = []
    n_pairs = 0
    for qkey, qinfo in queries.items():
        if qkey in results:
            continue
        bd = build_question_block(qinfo, content, genfrom, "")
        if bd is None:
            results[qkey] = {}  # no covering summary -> nothing to judge
            continue
        pending.append((qkey, qinfo))
        n_pairs += len(bd[2])

    print(f"[judge] {len(pending)} queries need judging ({n_pairs} (query,summary) pairs)")
    if args.dry_run:
        return

    batches = 0
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        question_map, blocks = {}, []
        for i, (qkey, qinfo) in enumerate(batch, 1):
            q_label = f"Q{i}"
            bd = build_question_block(qinfo, content, genfrom, q_label)
            if bd is None:
                continue
            block, ev_labels, sum_labels = bd
            question_map[q_label] = {"evidence_labels": ev_labels,
                                     "summary_labels": sum_labels, "qkey": qkey}
            blocks.append(block)
        if not blocks:
            continue
        response = call_claude(SYSTEM_PROMPT, "\n".join(blocks) + BATCH_FOOTER, args.model)
        parsed = parse_batch_response(response, question_map)
        for q_label, qinfo in question_map.items():
            qkey = qinfo["qkey"]
            results[qkey] = parsed.get(qkey, {})
        batches += 1
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[judge] {start + len(batch)}/{len(pending)} queries ({batches} batches) -> checkpoint")

    out_path.write_text(json.dumps(results, indent=2))
    covered = sum(1 for cov in results.values() for e in cov.values() if e)
    not_covered = sum(1 for cov in results.values() for e in cov.values() if not e)
    print(f"\n[judge] DONE. {covered} covering (query,summary) judgments, "
          f"{not_covered} non-covering. -> {out_path}")
    print("[judge] Next: uv run experiments/sleep-bench/impute_rescore.py "
          f"--rem-dir {rem} --judged-coverage {out_path}")


if __name__ == "__main__":
    main()
