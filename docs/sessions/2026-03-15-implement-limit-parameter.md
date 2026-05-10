# 2026-03-15 — Implement limit parameter + remove cliff detector

Added `limit` param to `recall()` (default 5), removed `apply_quality_floor` and `_fit_log_curve` from scoring.py, removed `CLIFF_Z_THRESHOLD` and `CLIFF_MIN_RESULTS` from constants.

Synced UCB exploration bonus from production → somnigraph (was still using old feedback mean-deviation). Fixed EWMA pseudo-count bug: old code used raw `count` as Beta pseudo-count, but EWMA's effective sample size is `1/(2α - α²)` (~1.85 at α=0.52). This means the exploration bonus was collapsing far too fast — after 10 events the system acted like it had 10 iid samples when it really had ~2 samples of information. UCB_COEFF and EWMA_ALPHA need joint retuning.

Updated README snippet, architecture.md, roadmap.md. Production and somnigraph now match.

No priority reorder — P2 (real-data tuning) is the right next step, now with the ESS correction as an additional reason to retune.

### Surprise

The ESS bug was invisible because the feedback mean-deviation boost (old FEEDBACK_COEFF) was also using the inflated count, so both old and new formulas had the same structural error — it only matters now because UCB uses variance, where the count appears in the denominator and the effect is amplified.
