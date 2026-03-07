# Somnigraph

A research-driven persistent memory system for [Claude Code](https://docs.anthropic.com/en/docs/claude-code), built on SQLite with hybrid retrieval, biological decay, sleep-based consolidation, and a retrieval feedback loop.

## What this is

This is my personal memory system for Claude Code. I'm sharing it because the ideas and research behind it are interesting — not because I'm offering a polished product. Issues and discussions about the approach are welcome; support requests probably aren't worth your time or mine.

The system gives Claude persistent memory across sessions: the ability to store, recall, forget, and consolidate memories using mechanisms inspired by biological memory systems.

## What makes it different

Most MCP memory servers store and retrieve. Somnigraph also **forgets, sleeps, and learns from feedback**.

- **Hybrid retrieval**: RRF fusion of FTS5 keyword search and vector similarity (sqlite-vec), with [experiments showing when each channel matters](docs/experiments.md)
- **Biological decay**: Per-category exponential decay with configurable half-lives, dormancy detection, and a quality floor
- **Sleep consolidation**: Offline batch processing in two phases — NREM (cluster and merge similar memories) and REM (gap analysis, question generation)
- **Graph edges**: Typed relationships between memories with novelty-scored adjacency expansion during recall
- **Retrieval feedback**: Explicit utility scoring that compounds into confidence, drives Hebbian co-retrieval boosting, and shapes future recall
- **Shadow load tracking**: Memories that surface repeatedly without being useful get suppressed

Every major design choice was tested against live retrieval data. See [docs/experiments.md](docs/experiments.md) for the methodology and results, including several cases where my assumptions were wrong.

## Quick start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), an OpenAI API key (for embeddings).

```bash
git clone https://github.com/alexisolson/somnigraph.git
cd somnigraph
cp .env.example .env
# Edit .env with your OpenAI API key

# Register as a Claude Code MCP server
claude mcp add somnigraph -- uv run src/memory_server.py
```

Add the following to your `CLAUDE.md` to instruct Claude how to use the memory tools effectively:

```markdown
<!-- TODO: CLAUDE.md snippet -->
```

## Architecture

```
src/
├── memory_server.py     # MCP entry point (FastMCP wiring, ~300 lines)
└── memory/
    ├── constants.py     # All tuning parameters, organized by validation tier
    ├── db.py            # SQLite connection, schema, migrations
    ├── scoring.py       # Post-RRF scoring pipeline (feedback, adjacency, Hebbian, shadow, confidence)
    ├── write.py         # Memory creation with deduplication and privacy stripping
    ├── graph.py         # Edge creation, novelty-scored adjacency expansion
    ├── decay.py         # Per-category exponential decay, dormancy, shadow load
    ├── fts.py           # FTS5 indexing and query sanitization
    ├── vectors.py       # sqlite-vec operations, vector math
    ├── embeddings.py    # OpenAI embedding API, token counting, enriched text construction
    ├── themes.py        # Theme normalization and variant mapping
    ├── privacy.py       # Sensitive data stripping before storage
    ├── formatting.py    # Memory display formatting (compact/full/pending)
    ├── stats.py         # Memory statistics and health metrics
    └── tools.py         # Tool implementation bodies (impl_* functions)
```

**Stack**: SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) (vector KNN) + FTS5 (keyword search) + OpenAI `text-embedding-3-small` embeddings.

**11 tools**: `startup_load`, `remember`, `recall`, `recall_feedback`, `link`, `forget`, `reflect`, `review_pending`, `consolidate`, `reembed_all`, `memory_stats`.

## Research

This system was built through iterative research across 14 phases, starting with a survey of academic papers and existing implementations, and ending with empirical validation against live data.

- **[docs/experiments.md](docs/experiments.md)** — Retrospective experiments testing specific hypotheses (vector vs. FTS5, decay models, Hebbian boosting, scoring calibration)
- **[docs/architecture.md](docs/architecture.md)** — Key design decisions with reasoning
- **[docs/similar-systems.md](docs/similar-systems.md)** — Opinionated comparison with other MCP memory servers
- **[research/sources/](research/sources/)** — Raw analysis of papers and implementations surveyed during research

### Key references

- [HippoRAG](https://arxiv.org/abs/2405.14831) (NeurIPS 2024) — Neurobiologically-inspired RAG with Personalized PageRank
- [GraphRAG](https://github.com/microsoft/graphrag) — Microsoft's graph-based RAG with community detection
- [Graphiti](https://github.com/getzep/graphiti) — Zep's temporal knowledge graph (gold-standard contradiction handling)
- "Lost in the Middle" (Liu et al. 2024) — Position bias in long-context retrieval
- Complementary Learning Systems theory — Biological basis for the sleep consolidation approach

## License

Apache 2.0 with [Commons Clause](https://commonsclause.com/). Use it, modify it, learn from it — just don't sell it.
