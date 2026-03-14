# Scripts

## Sleep pipeline

Full consolidation cycle: `uv run scripts/sleep.py`

Individual stages (run in order):

- **sleep_nrem.py** — Edge classification: cluster similar memories, merge duplicates, detect contradictions, handle temporal evolution, build graph edges. Modes: `standard` (new memories only) or `deep` (all active).
- **sleep_rem.py** — Higher-order consolidation: taxonomy-driven topic clustering, summary generation (thin ~150-200 token paragraphs optimized for vocabulary bridging), dormancy transitions, health diagnostics, seed curation.
- **sleep_consolidate.py** — Thematic cluster consolidation: detect clusters via taxonomy + weight-filtered edge BFS, LLM-judged keep/archive/merge/rewrite/annotate decisions.
- **sleep.py** — Orchestrator: runs the full 8-stage sequence (Normalize → NREM → Repair → REM → Detect → Review → AutoApply → Probe). NREM/REM failure aborts; theme/repair phases fail softly.
- **sleep_timing.py** — Timing statistics for sleep runs (parses progress log).

## Research

Experiment scripts used during development. Not polished, but functional for reproducing results documented in `docs/experiments.md`.
