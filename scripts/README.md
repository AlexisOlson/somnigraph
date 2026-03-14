# Scripts

## Sleep pipeline

Full consolidation cycle: `uv run scripts/sleep.py`

Individual stages (run in order):

- **sleep_nrem.py** — Edge classification: cluster similar memories, merge duplicates, detect contradictions, handle temporal evolution, build graph edges. Modes: `standard` (new memories only) or `deep` (all active).
- **sleep_rem.py** — Higher-order consolidation: taxonomy-driven topic clustering, summary generation (thin ~150-200 token paragraphs optimized for vocabulary bridging), dormancy transitions, health diagnostics, seed curation.
- **sleep_consolidate.py** — Thematic cluster consolidation: detect clusters via taxonomy + weight-filtered edge BFS, LLM-judged keep/archive/merge/rewrite/annotate decisions.
- **sleep.py** — Orchestrator: runs the full 8-stage sequence (Normalize → NREM → Repair → REM → Detect → Review → AutoApply → Probe). NREM/REM failure aborts; theme/repair phases fail softly.
- **sleep_timing.py** — Timing statistics for sleep runs (parses progress log).

## Ground truth

Scripts for building and judging relevance labels used in retrieval tuning.

- **build_ground_truth.py** — Extracts historical queries from the DB, retrieves deep candidate sets (union of vector + keyword at large budget), and optionally calls an LLM judge to grade relevance. Two main modes:
  - `--export-candidates out.json` — Export candidate sets for offline judging (no LLM calls).
  - `--resume` — Run or resume LLM judging directly against the DB.
- **judge_ground_truth.py** — Standalone offline judge. Reads pre-exported candidates JSON and calls `claude -p` to grade relevance. No DB or memory package dependency — runs anywhere with just the claude CLI. Safe to `Ctrl+C` and `--resume`.

Typical workflow:
```bash
# 1. Export candidates from live DB
uv run scripts/build_ground_truth.py --export-candidates candidates.json

# 2. Judge offline (in batches if needed)
python scripts/judge_ground_truth.py --candidates candidates.json --resume --max-queries 200
```

## Research

Experiment scripts used during development. Not polished, but functional for reproducing results documented in `docs/experiments.md`.
