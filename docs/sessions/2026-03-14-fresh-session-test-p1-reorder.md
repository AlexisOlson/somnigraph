# 2026-03-14 — Fresh-session snippet test + P1 reorder

A genuinely fresh Claude session (no prior somnigraph context) used only the snippet for memory guidance. Result: correct tool usage for all workflow steps (startup_load, recall with dual-input, recall_feedback with ratings, session-end reasoning). Advanced params (boost_themes, cutoff_rank) were discovered via tool schema, not snippet — correctly Tier 2 material.

P1 reorder triggered: three internal validation passes (dogfood, simulation, fresh-session) meet the "stable and tested" bar; "external feedback" is structurally blocked by private repo.

Priorities reordered: honest accounting #1 (invariant), real-data tuning #2 (active), documentation #3 (maintenance), migration #4 (terminated).
