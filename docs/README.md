# Documentation map

Somnigraph is a research artifact: **the docs are the product**, the code is proof they aren't theoretical. This is the index to them, grouped by what you're trying to do. Each fact has one canonical home; everything else links to it.

Start at the repo [`README.md`](../README.md) — the front door and the CLAUDE.md snippet — and [`STEWARDSHIP.md`](../STEWARDSHIP.md), which is process, not knowledge: how the project runs and what's prioritized.

## The system — how it works

- [`architecture.md`](architecture.md) — the master narrative: every design decision and why (RRF over single-channel, the decay model, the reranker, the graph, sleep). Its **What didn't work** and **Open problems** sections are the honest-accounting core.

## The evidence — does it work

- [`experiments.md`](experiments.md) — tuning methodology and the studies behind the numbers: metrics, the two-basin discovery, multi-objective optimization, utility calibration, reranker methodology.
- [`benchmarks.md`](benchmarks.md) — the scores next to other systems: LoCoMo retrieval and end-to-end QA (85.1%), the multi-hop vocabulary gap, cross-system reference tables, and the proposed PERMA/AMB work.

## The landscape — what others do

- [`similar-systems.md`](similar-systems.md) — opinionated feature comparison with the systems we learned from, plus practitioner signal from the community.
- [`ideas-considered.md`](ideas-considered.md) — the carsteneu survey ledger: 137 external mechanisms worth knowing, each mapped to a Somnigraph module with an effort estimate (the positive space).
- [`declined.md`](declined.md) — what we examined and chose not to pursue, or tried and removed (the negative space). One index for both external and internal declines.
- [`../research/sources/`](../research/sources/) — the raw corpus: 180+ per-system and per-paper analyses, indexed in [`index.md`](../research/sources/index.md).

## The agenda — what's next

- [`roadmap.md`](roadmap.md) — the forward research agenda: open questions, proposed experiments, infrastructure gaps. Forward-only; settled lessons live in architecture/experiments and are pointed to from here.
- [`proposals/`](proposals/) — design docs for not-yet-built features (e.g. [`proactive-injection.md`](proposals/proactive-injection.md)).

## Front door — the depth gradient

- [`claude-md-guide.md`](claude-md-guide.md) — Tier 2 usage guidance behind the README's ~30-line snippet: token budgets, when to recall, the category taxonomy, feedback intent, common failure modes.

## Reference & archive

- [`migration-notes.md`](migration-notes.md) — how production code was adapted into this repo.
- [`stewardship-history.md`](stewardship-history.md) — retired STEWARDSHIP changelog entries.
- [`sessions/`](sessions/) — per-session retrospectives, one file per session (the institutional record).
