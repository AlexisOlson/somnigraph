# Somnigraph

@STEWARDSHIP.md

A research-driven persistent memory system for Claude Code. SQLite + sqlite-vec + FTS5, hybrid retrieval with RRF fusion, biological decay, sleep-based consolidation, retrieval feedback loop.

## Project state

Migration from production is complete (Phases 1-5). The system is live and stable.

### Next session work

GT judging is in progress (~500/1047 queries judged with Sonnet, running in batches). External review complete (4 independent reviewers): PPR contradiction edge traversal bug fixed, narrative corrections applied to architecture.md and experiments.md, roadmap expanded with 8 new experiments and 4 new open questions. See `docs/roadmap.md` § External review findings for the full account. When GT completes, Tier 1 experiments are unblocked — now includes utility calibration study and counterfactual coverage check alongside the original three. The `migrate-gt-scripts` branch is ready for review and merge.

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
- Tuning tools: `SERVERS_DIR` removed; `CONSTANTS_PATH` points to repo's `src/memory/constants.py`; all data paths (`tuning_studies/`, `tuning_cache/`, `tuning_plots/`) use `DATA_DIR`; API key uses `OPENAI_API_KEY` env var with `DATA_DIR/openai_api_key` fallback
- probe_recall.py: imports `impl_recall`/`impl_recall_feedback` from `memory.tools` (not re-exported in `__init__.py`); subprocess workspace uses `DATA_DIR`
- plot_tuning.py: `STUDY_DIR` uses `DATA_DIR / "tuning_studies"` instead of hardcoded path
- `pyproject.toml` adds optional `[tuning]` dependency group for optuna, numpy, scikit-learn, matplotlib
- build_ground_truth.py: `DB_PATH`/`get_db()`/`serialize_f32()` imported from memory package (were local redefinitions); `get_embedding()` replaced by `embed_text()` from `memory.embeddings`; `OUTPUT_PATH` uses `DATA_DIR`
- judge_ground_truth.py: kept standalone (no memory package imports) for portability; default model changed from `claude-opus-4-6` to `claude-sonnet-4-6`

## Repo structure

```
docs/           — Narrative documentation (architecture, experiments, similar-systems)
research/       — 62 source analyses (papers, benchmarks, repos)
src/memory/     — Server modules (16 files when complete)
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

## Production reference

The live system that code is migrated from:
- Server: `~/.claude/servers/memory_server.py` + `~/.claude/servers/memory/`
- Scripts: `~/.claude/scripts/sleep*.py`, `tune_memory.py`, `probe_recall.py`, `plot_tuning.py`, `memory_stats.py`
- Data: `~/.claude/data/memory.db`, `taxonomy.json`, `theme_mappings.json`
- Docs: `~/.claude/Personal Vault/01-Projects/Memory Research/`
