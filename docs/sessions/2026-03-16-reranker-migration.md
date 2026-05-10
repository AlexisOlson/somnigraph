# 2026-03-16 — Reranker migration + documentation

Migrated learned reranker from production scripts to repo. New: `src/memory/reranker.py` (live feature extraction + prediction, 18 features), `scripts/train_reranker.py` and `scripts/tune_gt.py` (training infrastructure). Wired reranker into `impl_recall()` with formula fallback. Added lightgbm + numpy to main dependencies.

Documentation: new "The reranker" section in architecture.md, "Reranker methodology" in experiments.md, formula added to "What didn't work" (with honest framing — the formula was successful, the reranker is better and cheaper), roadmap updated (re-tune superseded, three new Tier 1 improvement experiments).

P2 reworded: "real-data tuning" → "reranker deployment + iteration." P2 reorder condition updated accordingly.
