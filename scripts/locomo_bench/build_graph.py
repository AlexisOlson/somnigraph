# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6", "openai>=2.0.0"]
# ///
"""
Build graph nodes and edges from v6 extraction output.

Reads conv{N}_v6.json and dia_map, inserts:
- Claim memories (retrieval_text as content, source="extraction", category="claim")
- Segment memories (retrieval_text as content, source="extraction", category="segment")
- EXTRACTED_FROM edges: synthetic node → source turn
- ENTITY_COREF edges: turn ↔ turn via shared entity claims

All scriptable — zero LLM cost. Embeddings use the shared embed cache.

Usage:
  uv run scripts/locomo_bench/build_graph.py --conversations 0 1 2
  uv run scripts/locomo_bench/build_graph.py --conversations 0 --dry-run
"""

import argparse
import json
import logging
import sys
import uuid
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from locomo_bench.config import BenchConfig

logger = logging.getLogger(__name__)


def load_extraction(conv_idx: int) -> dict:
    """Load v6 extraction JSON for a conversation."""
    path = REPO_ROOT / "scripts" / "locomo_bench" / "extractions" / f"conv{conv_idx}_v6.json"
    with open(path) as f:
        return json.load(f)


def load_dia_map(run_dir: Path) -> dict[tuple, str]:
    """Load dia_map from a benchmark run directory."""
    path = run_dir / "dia_map.json"
    with open(path) as f:
        raw = json.load(f)
    # Keys are stored as "conv_id:dia_id" strings
    dia_map = {}
    for k, v in raw.items():
        parts = k.split(":", 1)
        conv_id = int(parts[0])
        dia_id = parts[1]
        dia_map[(conv_id, dia_id)] = v
    return dia_map


def resolve_turn_id(dia_id: str, conv_idx: int, dia_map: dict) -> str | None:
    """Resolve extraction turn ID (e.g. 'D3:1') to memory_id via dia_map."""
    return dia_map.get((conv_idx, dia_id))


def build_graph(
    conv_idx: int,
    dia_map: dict[tuple, str],
    embed_cache_path: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Build and insert graph nodes + edges for one conversation.

    Returns stats dict with counts of inserted nodes and edges.
    """
    from memory.db import get_db
    from memory.embeddings import build_enriched_text, embed_batch
    from memory.graph import _create_edge
    from memory.write import _insert_memory

    if embed_cache_path:
        from locomo_bench.ingest import EmbeddingCache
        cache = EmbeddingCache(embed_cache_path)
    else:
        cache = None

    extraction = load_extraction(conv_idx)
    claims = extraction.get("claims", [])
    segments = extraction.get("segments", [])
    entities = extraction.get("entities", [])

    # Build entity alias map: name → canonical name
    entity_aliases = {}
    for ent in entities:
        canonical = ent["name"]
        entity_aliases[canonical.lower()] = canonical
        for alias in ent.get("aliases", []):
            entity_aliases[alias.lower()] = canonical

    stats = {
        "claims_inserted": 0,
        "segments_inserted": 0,
        "extracted_from_edges": 0,
        "entity_coref_edges": 0,
        "skipped_unresolved": 0,
    }

    # --- Prepare synthetic nodes ---
    synthetic_nodes = []  # (mem_id, content, summary, category, themes_list, source_turn_ids)

    # Claims → synthetic memories
    for claim in claims:
        retrieval_text = claim.get("retrieval_text", "")
        if not retrieval_text:
            continue

        subject = claim.get("subject", "")
        relation = claim.get("relation", "")
        attributed_to = claim.get("attributed_to", "")
        status = claim.get("status", "")
        polarity = claim.get("polarity", "")

        # Build themes from claim metadata
        themes = []
        if subject:
            themes.append(subject)
        if attributed_to and attributed_to != subject:
            themes.append(attributed_to)
        if relation:
            themes.append(relation)
        if status:
            themes.append(status)

        # Add session tags from time_scope
        time_scope = claim.get("time_scope", {})
        for sess in time_scope.get("sessions", []):
            themes.append(f"session_{sess}")

        # Resolve evidence turns
        source_turn_ids = []
        for eid in claim.get("evidence_turn_ids", []):
            mem_id = resolve_turn_id(eid, conv_idx, dia_map)
            if mem_id:
                source_turn_ids.append(mem_id)
            else:
                stats["skipped_unresolved"] += 1

        if not source_turn_ids:
            continue

        summary = f"{subject} {relation} {claim.get('object', '')}"[:80]
        mem_id = str(uuid.uuid4())
        synthetic_nodes.append((
            mem_id, retrieval_text, summary, "claim", themes, source_turn_ids
        ))

    # Segments → synthetic memories
    for segment in segments:
        retrieval_text = segment.get("retrieval_text", "")
        if not retrieval_text:
            continue

        # Build themes from segment metadata
        themes = list(segment.get("categories", []))
        themes.extend(segment.get("bridging_terms", []))
        sess_num = segment.get("session_number")
        if sess_num is not None:
            themes.append(f"session_{sess_num}")

        # Resolve constituent turns
        source_turn_ids = []
        for tid in segment.get("turns", []):
            mem_id = resolve_turn_id(tid, conv_idx, dia_map)
            if mem_id:
                source_turn_ids.append(mem_id)
            else:
                stats["skipped_unresolved"] += 1

        if not source_turn_ids:
            continue

        summary = segment.get("topic", retrieval_text[:80])[:80]
        mem_id = str(uuid.uuid4())
        synthetic_nodes.append((
            mem_id, retrieval_text, summary, "segment", themes, source_turn_ids
        ))

    if dry_run:
        claim_count = sum(1 for n in synthetic_nodes if n[3] == "claim")
        seg_count = sum(1 for n in synthetic_nodes if n[3] == "segment")
        logger.info("DRY RUN conv %d: %d claims, %d segments, %d unresolved turn refs",
                     conv_idx, claim_count, seg_count, stats["skipped_unresolved"])

        # Count claim coref edges
        coref_pairs = _build_claim_coref_pairs(claims, conv_idx, dia_map)
        logger.info("DRY RUN conv %d: would create %d claim coref edges", conv_idx, len(coref_pairs))
        return stats

    # --- Embed synthetic nodes ---
    enriched_texts = []
    for mem_id, content, summary, category, themes, _ in synthetic_nodes:
        enriched = build_enriched_text(content, category, themes, summary)
        enriched_texts.append(enriched)

    if cache:
        missing = cache.missing_indices(enriched_texts)
        if missing:
            logger.info("  Embedding %d new synthetic texts (%d cached)...",
                        len(missing), len(enriched_texts) - len(missing))
            miss_texts = [enriched_texts[i] for i in missing]
            miss_embeddings = embed_batch(miss_texts)
            for idx, emb in zip(missing, miss_embeddings):
                cache.put(enriched_texts[idx], emb)
            cache.save()
        embeddings = [cache.get(t) for t in enriched_texts]
    else:
        logger.info("  Embedding %d synthetic texts...", len(enriched_texts))
        embeddings = embed_batch(enriched_texts)

    # --- Insert into DB ---
    db = get_db()

    try:
        # Insert synthetic memories
        for (mem_id, content, summary, category, themes, source_turn_ids), embedding in zip(
            synthetic_nodes, embeddings
        ):
            _insert_memory(
                db, mem_id, content,
                summary=summary,
                category=category,
                themes_json=json.dumps(themes),
                priority=3,  # lower than turns (5)
                source="extraction",
                status="active",
                embedding=embedding,
            )

            if category == "claim":
                stats["claims_inserted"] += 1
            else:
                stats["segments_inserted"] += 1

            # EXTRACTED_FROM edges: synthetic → source turns
            for turn_id in source_turn_ids:
                edge_id = _create_edge(
                    db, mem_id, turn_id,
                    linking_context=f"extracted_from:{category}",
                    flags=["derivation"],
                    created_by="extraction",
                )
                if edge_id:
                    stats["extracted_from_edges"] += 1

        # --- Claim co-reference edges: turn ↔ turn ---
        coref_pairs = _build_claim_coref_pairs(claims, conv_idx, dia_map)
        for turn_a, turn_b, context in coref_pairs:
            edge_id = _create_edge(
                db, turn_a, turn_b,
                linking_context=context,
                created_by="extraction",
            )
            if edge_id:
                stats["entity_coref_edges"] += 1

        db.commit()
    finally:
        db.close()

    logger.info("Conv %d: %d claims, %d segments, %d extracted_from edges, %d entity_coref edges",
                conv_idx, stats["claims_inserted"], stats["segments_inserted"],
                stats["extracted_from_edges"], stats["entity_coref_edges"])
    return stats


def _build_claim_coref_pairs(
    claims: list[dict],
    conv_idx: int,
    dia_map: dict[tuple, str],
) -> set[tuple[str, str, str]]:
    """Build co-reference edges between turns that share a claim.

    Rather than connecting all turns mentioning an entity (O(n^2) per entity),
    connects turns within the same claim's evidence set. This is more precise:
    "these turns are about the same specific fact about this entity."

    Returns set of (turn_id_a, turn_id_b, context_str) tuples.
    """
    pairs = set()
    for claim in claims:
        subject = claim.get("subject", "")
        relation = claim.get("relation", "")
        if not subject:
            continue
        # Resolve evidence turns for this claim
        turn_ids = []
        for eid in claim.get("evidence_turn_ids", []):
            mem_id = resolve_turn_id(eid, conv_idx, dia_map)
            if mem_id:
                turn_ids.append(mem_id)
        # Pairwise edges within this claim's evidence
        context = f"claim_coref:{subject}:{relation}"
        for i in range(len(turn_ids)):
            for j in range(i + 1, len(turn_ids)):
                a, b = sorted([turn_ids[i], turn_ids[j]])
                pairs.add((a, b, context))
    return pairs


def count_existing_synthetic(run_dir: Path) -> int:
    """Check if synthetic nodes already exist in the DB."""
    from memory.db import get_db
    db = get_db()
    try:
        count = db.execute(
            "SELECT COUNT(*) FROM memories WHERE source = 'extraction'"
        ).fetchone()[0]
        return count
    finally:
        db.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Build graph from v6 extractions")
    parser.add_argument("--conversations", type=int, nargs="+", default=list(range(10)),
                        help="Conversation indices to process (default: all 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without inserting")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if synthetic nodes already exist")
    parser.add_argument("--enrich-suffix", default="",
                        help="DB directory suffix (e.g. '_enriched')")
    args = parser.parse_args()

    bench_config = BenchConfig()

    for conv_idx in args.conversations:
        run_dir = bench_config.base_dir / f"retrieval_{conv_idx}{args.enrich_suffix}"

        if not (run_dir / "dia_map.json").exists():
            logger.warning("Conv %d: no dia_map.json in %s — run eval_retrieval first", conv_idx, run_dir)
            continue

        # Point DB at this conversation's isolated DB
        import memory.db as db_mod
        db_path = run_dir / "memory.db"
        if not db_path.exists():
            logger.warning("Conv %d: no memory.db in %s", conv_idx, run_dir)
            continue

        # Set DB path for this conversation
        from locomo_bench.run import setup_isolated_db
        setup_isolated_db(run_dir)

        if not args.dry_run and not args.force:
            existing = count_existing_synthetic(run_dir)
            if existing > 0:
                logger.info("Conv %d: %d synthetic nodes already exist (use --force to rebuild)",
                            conv_idx, existing)
                continue

        if not args.dry_run and args.force:
            # Remove existing synthetic nodes and their edges
            from memory.db import get_db
            db = get_db()
            try:
                synthetic_ids = [r[0] for r in db.execute(
                    "SELECT id FROM memories WHERE source = 'extraction'"
                ).fetchall()]
                if synthetic_ids:
                    placeholders = ",".join("?" * len(synthetic_ids))
                    db.execute(f"DELETE FROM memory_edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                               synthetic_ids + synthetic_ids)
                    db.execute(f"DELETE FROM memory_fts WHERE rowid IN (SELECT rowid FROM memory_rowid_map WHERE memory_id IN ({placeholders}))",
                               synthetic_ids)
                    db.execute(f"DELETE FROM memory_vec WHERE rowid IN (SELECT rowid FROM memory_rowid_map WHERE memory_id IN ({placeholders}))",
                               synthetic_ids)
                    db.execute(f"DELETE FROM memory_rowid_map WHERE memory_id IN ({placeholders})",
                               synthetic_ids)
                    db.execute(f"DELETE FROM memories WHERE id IN ({placeholders})",
                               synthetic_ids)
                    # Also remove entity_coref edges (between turns, created by extraction)
                    db.execute("DELETE FROM memory_edges WHERE created_by = 'extraction'")
                    db.commit()
                    logger.info("Conv %d: removed %d existing synthetic nodes", conv_idx, len(synthetic_ids))
            finally:
                db.close()

        dia_map = load_dia_map(run_dir)
        stats = build_graph(
            conv_idx, dia_map,
            embed_cache_path=bench_config.embed_cache,
            dry_run=args.dry_run,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
