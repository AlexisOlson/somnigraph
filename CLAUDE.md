# Somnigraph

@STEWARDSHIP.md

A research-driven persistent memory system for Claude Code. SQLite + sqlite-vec + FTS5, hybrid retrieval with RRF fusion, biological decay, sleep-based consolidation, retrieval feedback loop.

## Project state

Migration from production is complete (Phases 1-5). The system is live and stable.

See `docs/roadmap.md` for current research agenda and `STEWARDSHIP.md` for session priorities.

@docs/migration-notes.md

## Repo structure

```
docs/           — Narrative documentation (architecture, experiments, similar-systems)
research/       — 68 source analyses (papers, benchmarks, repos)
src/memory/     — 17 server modules
src/memory_server.py — MCP entry point
scripts/        — Sleep pipeline, tuning tools, ground truth
```

## Workflow

1. Read this file and STEWARDSHIP.md for current state and priorities
2. Propose 1-3 things to work on (with reasoning from the priority list)
3. Get approval before starting
4. Work on a feature branch
5. Review diff together before merging
6. Run the retrospective protocol in STEWARDSHIP.md before closing

## Key files

- `docs/architecture.md` — Master narrative of design decisions
- `docs/experiments.md` — Tuning methodology
- `docs/roadmap.md` — Research agenda: what we learned, open questions, proposed experiments
- `docs/similar-systems.md` — Comparison with other systems
- `research/sources/index.md` — Research source catalog
