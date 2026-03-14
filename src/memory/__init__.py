"""Somnigraph — persistent memory with hybrid search, biological decay, and sleep consolidation."""

from memory.constants import DATA_DIR
from memory.db import get_db, DB_PATH, _resolve_id
from memory.embeddings import count_tokens, embed_text, embed_batch, build_enriched_text
from memory.vectors import serialize_f32, deserialize_f32
from memory.write import _insert_memory
from memory.fts import sanitize_fts_query, update_fts, delete_fts
from memory.themes import normalize_themes, add_theme_mapping, normalize_all_themes, split_theme_on_memories
from memory.privacy import _strip_sensitive
from memory.decay import effective_priority
from memory.formatting import format_memory_compact, format_memory_full, format_memory_pending
from memory.events import _row_get, _log_event
from memory.session import detect_session_id, get_session_id
from memory.graph import (
    _create_edge, _find_related_memories, _compute_shadow_load,
    _check_fast_path, _handle_temporal_evolution, _source_confidence_modifier,
)
from memory.scoring import rrf_fuse, apply_hebbian, expand_via_ppr, apply_quality_floor
from memory.stats import compute_stats
from memory.constants import *  # all tuning constants

__all__ = [
    # Foundation
    "get_db", "DB_PATH", "_resolve_id", "DATA_DIR",
    "count_tokens", "embed_text", "embed_batch", "build_enriched_text",
    "serialize_f32", "deserialize_f32",
    "_insert_memory",
    "sanitize_fts_query", "update_fts", "delete_fts",
    "normalize_themes", "add_theme_mapping", "normalize_all_themes", "split_theme_on_memories",
    "_strip_sensitive",
    # Phase 3
    "effective_priority",
    "format_memory_compact", "format_memory_full", "format_memory_pending",
    "_row_get", "_log_event",
    "detect_session_id", "get_session_id",
    "_create_edge", "_find_related_memories", "_compute_shadow_load",
    "_check_fast_path", "_handle_temporal_evolution", "_source_confidence_modifier",
    "rrf_fuse", "apply_hebbian", "expand_via_ppr", "apply_quality_floor",
    "compute_stats",
]
