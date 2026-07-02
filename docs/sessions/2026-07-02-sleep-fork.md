# 2026-07-02 — Sleep counterfactual fork (arc step 4)

The central open question — *does sleep improve retrieval?* — got its first measurement. Four-arm counterfactual fork on copies of the live store (frozen A vs standard-full B, standard-consolidation-only B-noprobe, deep-consolidation-only B2), 1,032 GT queries, paired bootstrap 95% CIs, formula-scored both sides (the fork copies predate the V6 deploy). Run by an unsupervised Opus session against the amended handoff artifact; orchestrator reviewed mid-run and directed the probe-isolation decomposition and the deep arm. Findings ratified and canonized: [`experiments/sleep-fork/findings-sleep-fork.md`](../../experiments/sleep-fork/findings-sleep-fork.md).

**Headline:** at steady state on a regularly-deep-slept store, consolidation is statistically inert on retrieval; the full cycle's only significant delta (+0.0022 NDCG, 0.34% relative) decomposes into two individually-null halves (consolidation ≈ +0.0013, probe feedback ≈ +0.0009). A full `--deep` pass changed ~108 edges and did zero merges/dormancy/archival — the store is saturated. The accumulation-regime re-run (two-week abstinence window, started 2026-07-01) is the registered follow-up.

### Surprises

- **The `--snapshot` trap.** `eval_retrieval.py --snapshot` hardcodes the *live* DB path and ignores `SOMNIGRAPH_DATA_DIR` — running it naively on two store copies would have evaluated the same live DB twice and produced a perfectly convincing fake null. Caught before any arm was interpreted; each arm's `memory.db` was instead placed at the snapshot path and evaluated without `--snapshot`. Any future fork-style experiment must repeat this workaround (documented in the findings' Method section).
- **The first probe strip was incomplete, and the incompleteness was directional.** Stripping only feedback events left two live channels: the 30-day Hebbian co-retrieval window (built from `retrieved` events, not feedback) and edge weights (31 probe-*created* edges plus 359 consolidation edges whose weights the probe had bumped after creation). Both biased the decisive arm toward "consolidation helps." The re-strip covered all three channels and was verified by count-identity against A (`strip_probe.py` / `verify_strip.py`).
- **Net edge counts masked an order of magnitude of churn.** "+137 edges" was actually 902 created / 765 removed. Sleep did substantial rewiring that didn't move retrieval — which makes the null more informative, not less, and means future sleep audits should read gross actions, not net deltas.
- **An orchestrator hypothesis died in review.** After the deep arm's null, "probe feedback is the active ingredient" looked likely; the corrected strip refuted the strong form (both halves individually null). Recorded in the findings because the wrong hypothesis shaped the arm design.
- **Windows footnote:** `sleep.py` prints box-drawing characters; without `PYTHONUTF8=1` the cp1252 console crashes the run.

### Caveats

- Formula scorer on both sides of every arm (fork predates the V6 deploy) — this is sleep-vs-formula. The abstinence re-run against deployed V6 is also the first sleep-vs-reranker measurement.
- Single fork per condition; sleep's LLM steps are nondeterministic. Effect sizes this small could shift within run-to-run noise; the qualitative conclusion (inert consolidation, saturated store) is robust.
- The result answers the *steady-state* regime only. It does not say consolidation is useless — it says a store that is regularly deep-slept has already banked whatever consolidation buys.

### Files touched

- New: `experiments/sleep-fork/` (findings + 4 result JSONs + `analyze_sleep_fork.py`, `strip_probe.py`, `verify_strip.py`, `investigate_edges.py`, run manifest), this session file.
- Edited: `scripts/locomo_bench/eval_retrieval.py` (`eval_production` now emits the ranked list + budget/limit so R@10/MRR are computable — previously NDCG only), `docs/roadmap.md` (open question answered-for-steady-state; experiments #3 and #14 complete; abstinence follow-up registered), `docs/architecture.md` (§ Consolidation quality evaluation first-measurement note; § Feedback loop self-reinforcement probe-arm measurement), `STEWARDSHIP.md` (changelog).

### Reversibility

Store copies (A / B / B2 / B2full / B-noprobe, ~32 GB) remain on `D:\somnigraph-exp\sleepfork-92fdacb-1782966620\` — deletable now that findings + JSONs are committed; the abstinence experiment takes fresh copies. The live store was never written (verified: live `memory_stats()` last-sleep unchanged throughout).

### What's next

The two-week abstinence window (no live sleep; normal usage accumulates unconsolidated memories and window-slice feedback). Fork due ~2026-07-15: identical protocol, old GT + window-slice GT, V6 primary + formula secondary, same probe-isolation treatment. Same clock as the Write Guard shadow-data accumulation — the next arc's two instruments mature together.
