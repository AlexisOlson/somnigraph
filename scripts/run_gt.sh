#!/usr/bin/env bash
# Thin wrapper for build_ground_truth.py with production env baked in.
# Usage: ./scripts/run_gt.sh [args...]
# Examples:
#   ./scripts/run_gt.sh --max-queries 1
#   ./scripts/run_gt.sh --resume
#   ./scripts/run_gt.sh --dry-run

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

export SOMNIGRAPH_DATA_DIR="$HOME/.claude/data"
export SOMNIGRAPH_EMBEDDING_BACKEND=fastembed

uv run --directory "$REPO_DIR" --extra fastembed python scripts/build_ground_truth.py --resume "$@"
