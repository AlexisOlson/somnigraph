# Migration Notes

Reference for how production code (`~/.claude/servers/memory/`) was adapted during migration to this repo. Useful for understanding why certain patterns exist.

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
- train_reranker.py: `SERVERS_DIR`/`SCRIPTS_DIR` → `REPO_ROOT / "src"` and `REPO_ROOT / "scripts"`; `DATA_DIR` from `memory.constants`; `tune_gt` imports unchanged (now from repo's scripts/)
- tune_gt.py: `SERVERS_DIR` → `REPO_ROOT / "src"`; `DATA_DIR` from `memory.constants`; `MEMORY_DB` → `DB_PATH` from `memory.db`; production-only constants (K_FTS, K_VEC, W_THEME, K_THEME, BM25_SUMMARY_WT, BM25_THEMES_WT) kept as local constants (not in somnigraph's constants.py — the reranker makes them irrelevant for live scoring)
- reranker.py: new module for live feature extraction + prediction; `MODEL_PATH` constant added to `constants.py`; model loaded eagerly at server startup; `impl_recall()` branches on model availability
- lightgbm + numpy moved from optional to main dependencies (reranker is the production scoring path)
