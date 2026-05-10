# 2026-03-13 — Phase 6 tending

Phase 6 tending. Fixed stale README Status (still claimed Phases 4-5 unfinished). Fixed pyproject.toml packaging — memory_server.py was excluded from wheel builds (force-include, since hatchling py-modules doesn't resolve src/ layout paths).

### Surprise

The packaging gap was invisible from the uv run usage path and would only surface for pip installs.
