# Somnigraph

@STEWARDSHIP.md

A research-driven persistent memory system for Claude Code. SQLite + sqlite-vec + FTS5, hybrid retrieval with RRF fusion, biological decay, sleep-based consolidation, retrieval feedback loop.

## Project state

Migrating from production (`~/.claude/servers/`, `~/.claude/scripts/`) into this repo. The system is live and stable — this is packaging, not building.

### Migration status

- [x] Phase 1: Documentation (architecture.md, experiments.md, similar-systems.md)
- [x] Phase 2: Foundation code (db, constants, embeddings, vectors, write, privacy, themes, fts)
- [x] Phase 3: Scoring + tools (scoring, graph, decay, formatting, events, session, stats, tools, memory_server)
- [x] Phase 4: Sleep pipeline (sleep_nrem, sleep_rem, sleep_consolidate, sleep orchestrator)
- [ ] Phase 5: Tuning tools (tune_memory, plot_tuning, probe_recall)
- [ ] Phase 6: Ongoing tending

### Next session work

Phase 5: Migrate tuning tools (tune_memory, plot_tuning, probe_recall).

### Migration notes

- `DATA_DIR` lives in `constants.py` (not `db.py`) to avoid circular imports (db → fts → db)
- Configurable via `SOMNIGRAPH_DATA_DIR` env var, defaults to `~/.somnigraph/`
- Personal data (KNOWN_PHRASES, THEME_VARIANTS, CONTENT_THEME_PHRASES) load from JSON in `DATA_DIR`, with small generic defaults
- API key: `OPENAI_API_KEY` env var, fallback to `DATA_DIR/openai_api_key` file
- `pyproject.toml` license field uses table format for non-SPDX identifier
- Production `sync.py` → somnigraph `events.py` (name clarified; old JSON sync layer was removed in production 2026-03-06)
- Production `memory_server.py` re-exports for sleep script compat stripped — somnigraph server is pure MCP wiring
- Legacy BFS adjacency expansion (`_expand_adjacency_legacy`) preserved in scoring.py for research/tuning tool compatibility
- `consolidate()` archive path uses `DATA_DIR` (production had hardcoded `~/.claude/data/`)
- Sleep scripts: import paths changed from `memory_server` → `memory` package; data paths from hardcoded `~/.claude/data/` → configurable `DATA_DIR`; personal vault paths (core.md, seed.md, fragments, journal, archive) kept as `Path.home()` references
- REM step numbering: `4a/4b/4d/4d2/4d3/4f/4g/4h/4i/4j` → `1-10` (legacy from original multi-phase design where NREM was phases 1-3)
- Orchestrator `memory.sync` import → `memory.events` (matching the production→somnigraph rename)

## Repo structure

```
docs/           — Narrative documentation (architecture, experiments, similar-systems)
research/       — 62 source analyses (papers, benchmarks, repos)
src/memory/     — Server modules (16 files when complete)
src/memory_server.py — MCP entry point
scripts/        — Sleep pipeline + tuning tools (when migrated)
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
- `docs/similar-systems.md` — Comparison with other systems
- `research/sources/index.md` — Research source catalog

## Production reference

The live system that code is migrated from:
- Server: `~/.claude/servers/memory_server.py` + `~/.claude/servers/memory/`
- Scripts: `~/.claude/scripts/sleep*.py`, `tune_memory.py`, `probe_recall.py`, `plot_tuning.py`, `memory_stats.py`
- Data: `~/.claude/data/memory.db`, `taxonomy.json`, `theme_mappings.json`
- Docs: `~/.claude/Personal Vault/01-Projects/Memory Research/`
