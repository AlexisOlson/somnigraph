# 2026-03-20 — Production unification

Repo's 18-feature reranker replaced with production's 26-feature architecture. sync.py shim added. tools.py rewritten to match production's call flow. 26-feature model retrained (1032q, NDCG=0.7958). Production switched to run from repo via directory junction + settings.json update.

Three restarts needed — settings.json env var passthrough to MCP servers unreliable on Windows, fixed by hardcoding DATA_DIR fallback in memory_server.py. Also hit API key path mismatch (production hardcoded ~/.claude/secrets/, repo reads from DATA_DIR/).

### Surprise

The env var failure was silent — the server created a fresh empty DB at ~/.somnigraph/ and returned 0 memories without error. The plan's assumption that settings.json env would "just work" was wrong; sensible defaults beat configurable paths when the configuration mechanism is unreliable.

Query features experiment effectively complete (8 new features integrated); one remaining P2 experiment (raw-score features).
