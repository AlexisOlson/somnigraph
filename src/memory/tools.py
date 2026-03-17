"""MCP tool implementations — business logic for all memory tools."""

import json
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from memory.constants import (
    DATA_DIR,
    DEDUP_THRESHOLD, PENDING_STALE_DAYS, CATEGORY_DECAY_RATES,
    DEFAULT_DECAY_RATE, RRF_K,
    MAX_EDGES_PER_CYCLE, MAX_PRIORITY_BOOST_PER_CYCLE,
    CO_UTILITY_THRESHOLD, BRIDGE_THEME_THRESHOLD, BRIDGE_TERM_MIN_LEN,
    DISAPPOINTED_RECALL_MAX_UTIL, DISAPPOINTED_RETRIEVAL_SCORE,
    CONF_USEFUL_THRESHOLD, CONF_USELESS_THRESHOLD, CONF_GROWTH_RATE,
    CONF_DECAY_RATE, CONF_DURABILITY_NUDGE, CONF_DEFAULT, CONF_MIN_DELTA,
    DECAY_DURABILITY_SCALE, EDGE_WEIGHT_POS_STEP, EDGE_WEIGHT_NEG_STEP,
    THEME_REFINE_THRESHOLD, MAX_THEMES, MAX_NEW_TERMS,
)
from memory.db import get_db, _resolve_id
from memory.events import _row_get, _log_event
from memory.decay import effective_priority
from memory.embeddings import count_tokens, embed_text, embed_batch, build_enriched_text
from memory.formatting import format_memory_compact, format_memory_full, format_memory_pending
from memory.fts import sanitize_fts_query, _themes_for_fts
from memory.themes import normalize_themes
from memory.vectors import serialize_f32
from memory.privacy import _strip_sensitive
from memory.write import _insert_memory
from memory.graph import _create_edge
from memory.session import detect_session_id, get_session_id
from memory.scoring import (
    rrf_fuse, apply_hebbian, expand_via_ppr,
)
from memory.reranker import get_model as get_reranker_model, extract_live_features, rerank
from memory.stats import compute_stats


def impl_startup_load(budget: int = 3000) -> str:
    # Detect and cache the current session ID from JSONL transcript
    detect_session_id()

    db = get_db()

    # Get all active memories (projected — only columns needed for scoring + formatting)
    rows = db.execute("""
        SELECT id, content, summary, category, themes, base_priority,
               last_accessed, decay_rate, flags
        FROM memories WHERE status = 'active'
    """).fetchall()

    if not rows:
        pending_count = db.execute(
            "SELECT count(*) FROM memories WHERE status = 'pending'"
        ).fetchone()[0]
        if pending_count:
            return f"No active memories yet. {pending_count} memories awaiting review — use review_pending() to confirm or discard."
        return "No memories stored yet. Use remember() to start building memory."

    # Sort by effective priority
    scored = []
    for row in rows:
        mem_flags = json.loads(_row_get(row, "flags") or "[]")
        ep = effective_priority(
            row["base_priority"], row["last_accessed"],
            decay_rate=_row_get(row, "decay_rate"), category=row["category"],
            flags=mem_flags,
        )
        scored.append((ep, row))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Separate questions from regular memories, classify correction-adjacent
    CORRECTION_THEMES = {"calibration", "correction", "gotcha"}
    questions = []
    regular_correction = []
    regular_other = []
    for ep, row in scored:
        themes_list = json.loads(row["themes"]) if row["themes"] else []
        if "question" in themes_list:
            questions.append((ep, row))
        elif CORRECTION_THEMES & set(themes_list):
            regular_correction.append((ep, row))
        else:
            regular_other.append((ep, row))

    # Pack within budget — questions first, then regular memories with diversity floor
    lines = []
    loaded_ids = []
    tokens_used = 0
    header = "## Memory Briefing"

    # Include questions first (guaranteed slots)
    for ep, row in questions:
        line = format_memory_compact(row)
        line_tokens = count_tokens(line)
        if tokens_used + line_tokens > budget - 100:
            break
        lines.append(line)
        loaded_ids.append(row["id"])
        tokens_used += line_tokens

    question_ids = {row["id"] for _, row in questions}
    loaded_question_count = len(question_ids & set(loaded_ids))
    unloaded_question_count = len(question_ids - set(loaded_ids))

    # Fill remaining budget with regular memories, enforcing diversity floor:
    # correction-adjacent memories capped at 40% of loaded regular memories
    CORRECTION_CAP = 0.4
    correction_count = 0
    other_count = 0
    # Interleave: pull from both lists by effective priority
    corr_idx = 0
    other_idx = 0
    deferred_corrections = []  # corrections skipped due to cap, may fill remaining space
    while corr_idx < len(regular_correction) or other_idx < len(regular_other):
        # Pick the higher-priority candidate from each list
        corr_candidate = regular_correction[corr_idx] if corr_idx < len(regular_correction) else None
        other_candidate = regular_other[other_idx] if other_idx < len(regular_other) else None

        # Decide which to try next (higher effective priority wins)
        if corr_candidate and other_candidate:
            if corr_candidate[0] >= other_candidate[0]:
                candidate = corr_candidate
                is_correction = True
            else:
                candidate = other_candidate
                is_correction = False
        elif corr_candidate:
            candidate = corr_candidate
            is_correction = True
        elif other_candidate:
            candidate = other_candidate
            is_correction = False
        else:
            break

        ep, row = candidate
        line = format_memory_compact(row)
        line_tokens = count_tokens(line)

        if tokens_used + line_tokens > budget - 100:
            # Advance past this candidate either way
            if is_correction:
                corr_idx += 1
            else:
                other_idx += 1
            break

        total_regular = correction_count + other_count
        if is_correction:
            # Check if adding this correction would exceed the cap
            if total_regular > 0 and (correction_count + 1) / (total_regular + 1) > CORRECTION_CAP:
                # Defer this correction — try the other list instead
                deferred_corrections.append((ep, row, line, line_tokens))
                corr_idx += 1
                continue
            correction_count += 1
            corr_idx += 1
        else:
            other_count += 1
            other_idx += 1

        lines.append(line)
        loaded_ids.append(row["id"])
        tokens_used += line_tokens

    # If budget remains, fill with deferred corrections (soft floor, not hard block)
    for ep, row, line, line_tokens in deferred_corrections:
        if tokens_used + line_tokens > budget - 100:
            break
        lines.append(line)
        loaded_ids.append(row["id"])
        tokens_used += line_tokens
    total_questions = len(question_ids)

    # Count totals
    total_active = len(rows)
    pending_count = db.execute(
        "SELECT count(*) FROM memories WHERE status = 'pending'"
    ).fetchone()[0]

    footer_parts = [f"— {len(lines)} of {total_active} memories loaded"]
    if pending_count:
        footer_parts.append(
            f"{pending_count} pending review — use review_pending() to confirm or discard"
        )

    result = f"{header} ({len(lines)} of {total_active})\n\n"
    result += "\n".join(lines)
    result += f"\n\n{'. '.join(footer_parts)}."
    result += "\nUse recall('topic') for deeper retrieval."

    if total_questions:
        result += f"\n\n**Open questions**: {total_questions} active"
        if unloaded_question_count:
            result += f" ({unloaded_question_count} not shown — recall('question') for full list)"
        result += "\nWhen a question gets answered: store the answer as a memory, then forget() the question."

    result += ("\n\nFeedback: Once you have session context, pick ~5 memories with real signal "
               "(e.g., stale, surprisingly useful, irrelevant to how you actually work now) and grade them. "
               "Skip if nothing stands out.\n"
               "recall_feedback(feedback, query) — scores: {id: utility} (0.0-1.0) or "
               "{id: [utility, durability]} where durability is -1.0..1.0")

    # Update last_accessed but NOT access_count — startup loads shouldn't
    # inflate access stats. Only recall() and reflect() count.
    now = datetime.now(timezone.utc).isoformat()
    for mid in loaded_ids:
        db.execute(
            "UPDATE memories SET last_accessed = ?, startup_count = startup_count + 1 WHERE id = ?",
            (now, mid),
        )
    db.commit()
    db.close()

    return result


def impl_remember(
    content: str,
    category: str = "semantic",
    priority: int = 5,
    themes: str = "[]",
    summary: str = "",
    source: str = "session",
    status: str = "active",
    decay_rate: float = -1,
    confidence: float = -1,
    flags: str = "[]",
) -> str:
    db = get_db()
    try:
        # Privacy stripping — redact secrets before storage
        content = _strip_sensitive(content)
        if summary:
            summary = _strip_sensitive(summary)

        # Validate category
        valid_categories = {"episodic", "semantic", "procedural", "reflection", "meta"}
        if category not in valid_categories:
            return f"Invalid category '{category}'. Must be one of: {', '.join(sorted(valid_categories))}"

        # Validate status
        if status not in ("active", "pending"):
            return f"Invalid status '{status}'. Must be 'active' or 'pending'."

        # Clamp priority
        priority = max(1, min(10, priority))

        # Parse themes
        try:
            themes_list = json.loads(themes) if isinstance(themes, str) else themes
            if not isinstance(themes_list, list):
                themes_list = []
        except json.JSONDecodeError:
            themes_list = []
        themes_list = normalize_themes(themes_list, content=content)
        themes_json = json.dumps(themes_list)

        # Resolve decay_rate sentinel: -1 means use category default (NULL in DB)
        resolved_decay_rate = None if decay_rate < 0 else decay_rate

        # Resolve confidence sentinel: -1 means auto-resolve from source/status
        if confidence < 0:
            if status == "pending":
                resolved_confidence = 0.3
            elif source == "correction":
                resolved_confidence = 0.6
            elif source == "reflect":
                resolved_confidence = 0.55
            else:
                resolved_confidence = 0.5
        else:
            resolved_confidence = max(0.1, min(0.95, confidence))

        # Parse flags
        try:
            flags_list = json.loads(flags) if isinstance(flags, str) else flags
            if not isinstance(flags_list, list):
                flags_list = []
        except json.JSONDecodeError:
            flags_list = []
        resolved_flags_json = json.dumps(flags_list)

        # Generate enriched embedding (content + metadata for richer vector representation)
        enriched = build_enriched_text(content, category, themes_list, summary)
        embedding = embed_text(enriched)

        # Dedup check: cosine distance < threshold against same-category active+pending
        existing_ids = db.execute(
            "SELECT memory_id FROM memory_rowid_map WHERE memory_id IN "
            "(SELECT id FROM memories WHERE category = ? AND status IN ('active', 'pending'))",
            (category,),
        ).fetchall()

        if existing_ids:
            rowids = [
                db.execute(
                    "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (r["memory_id"],)
                ).fetchone()["rowid"]
                for r in existing_ids
            ]

            # Check against each existing memory for near-duplicates
            dupes = db.execute(
                """
                SELECT rowid, distance FROM memory_vec
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (serialize_f32(embedding), min(len(rowids), 20)),
            ).fetchall()

            for dupe in dupes:
                if dupe["distance"] < DEDUP_THRESHOLD:
                    # Find the memory this belongs to
                    dup_map = db.execute(
                        "SELECT memory_id FROM memory_rowid_map WHERE rowid = ?",
                        (dupe["rowid"],),
                    ).fetchone()
                    if dup_map:
                        dup_mem = db.execute(
                            "SELECT * FROM memories WHERE id = ? AND category = ? AND status IN ('active', 'pending')",
                            (dup_map["memory_id"], category),
                        ).fetchone()
                        if dup_mem:
                            if dup_mem["base_priority"] < priority:
                                # Supersede the lower-priority existing one
                                new_id = str(uuid.uuid4())
                                _insert_memory(
                                    db, new_id, content, summary, category,
                                    themes_json, priority, source, status, embedding,
                                    decay_rate=resolved_decay_rate,
                                    confidence=resolved_confidence,
                                    flags_json=resolved_flags_json,
                                    session_id=get_session_id(),
                                )
                                db.execute(
                                    "UPDATE memories SET superseded_by = ?, status = 'deleted' WHERE id = ?",
                                    (new_id, dup_mem["id"]),
                                )
                                _log_event(db, new_id, "created", context={"source": source, "superseded": dup_mem["id"]})
                                _log_event(db, dup_mem["id"], "superseded", context={"superseded_by": new_id})
                                db.commit()
                                return (
                                    f"Stored (superseded existing similar memory {dup_mem['id'][:8]}...).\n"
                                    f"ID: {new_id}\nStatus: {status}"
                                )
                            else:
                                _log_event(db, dup_mem["id"], "dedup_rejected", context={
                                    "new_summary": summary or content[:80],
                                    "distance": round(dupe["distance"], 4),
                                    "new_priority": priority,
                                    "existing_priority": dup_mem["base_priority"],
                                })
                                db.commit()
                                dup_summary = dup_mem["summary"] or dup_mem["content"][:80]
                                return (
                                    f"Similar memory already exists (distance={dupe['distance']:.3f}):\n"
                                    f"  [{dup_mem['category']}] {dup_summary}\n"
                                    f"  ID: {dup_mem['id']}\n"
                                    f"Use forget() first to replace, or adjust content to differentiate."
                                )

        # No duplicate — insert
        new_id = str(uuid.uuid4())
        _insert_memory(
            db, new_id, content, summary, category,
            themes_json, priority, source, status, embedding,
            decay_rate=resolved_decay_rate,
            confidence=resolved_confidence,
            flags_json=resolved_flags_json,
            session_id=get_session_id(),
        )
        _log_event(db, new_id, "created", context={"source": source, "category": category})
        db.commit()

        return f"Stored.\nID: {new_id}\nCategory: {category} | Priority: {priority} | Status: {status}"
    finally:
        db.close()


def _compute_ppr_for_reranker(
    db: sqlite3.Connection,
    all_ids: set[str],
    vec_ranked: dict[str, int],
    fts_ranked: dict[str, int],
    exclude_set: set[str],
) -> dict[str, float]:
    """Compute raw PPR scores for reranker features.

    Uses the same graph walk as expand_via_ppr but returns raw scores
    without mutating any score dict. Seeds are top candidates by RRF.
    """
    from memory.scoring import personalized_pagerank
    from memory.constants import (
        RRF_K, RRF_VEC_WEIGHT, ADJACENCY_SEED_COUNT,
        PPR_DAMPING, PPR_MIN_SCORE,
    )

    if not all_ids:
        return {}

    # Compute basic RRF scores for seed selection
    base_scores = {}
    for mid in all_ids:
        score = 0.0
        if mid in vec_ranked:
            score += RRF_VEC_WEIGHT / (RRF_K + vec_ranked[mid] + 1)
        if mid in fts_ranked:
            score += (1.0 - RRF_VEC_WEIGHT) / (RRF_K + fts_ranked[mid] + 1)
        base_scores[mid] = score

    if not base_scores:
        return {}

    seed_ids = sorted(base_scores, key=base_scores.get, reverse=True)[:ADJACENCY_SEED_COUNT]
    seed_set = set(seed_ids)
    seed_weights = {sid: base_scores[sid] for sid in seed_ids}

    # Build 2-hop subgraph (same as expand_via_ppr but no contradiction filter needed
    # since the reranker model learns which signals matter)
    ph = ",".join("?" * len(seed_ids))
    hop1_rows = db.execute(f"""
        SELECT source_id, target_id, weight, flags
        FROM memory_edges
        WHERE source_id IN ({ph}) OR target_id IN ({ph})
    """, seed_ids + seed_ids).fetchall()

    hop1_neighbors = set()
    hop1_clean = []
    for edge in hop1_rows:
        flags = edge["flags"] or ""
        if "contradiction" in flags:
            continue
        hop1_clean.append(edge)
        hop1_neighbors.add(edge["source_id"])
        hop1_neighbors.add(edge["target_id"])
    hop1_neighbors -= seed_set

    hop2_rows = []
    if hop1_neighbors:
        n_list = list(hop1_neighbors)
        n_ph = ",".join("?" * len(n_list))
        hop2_rows = db.execute(f"""
            SELECT source_id, target_id, weight, flags
            FROM memory_edges
            WHERE source_id IN ({n_ph}) OR target_id IN ({n_ph})
        """, n_list + n_list).fetchall()

    hop2_clean = [e for e in hop2_rows if "contradiction" not in (e["flags"] or "")]
    adj: dict[str, list[tuple[str, float]]] = {}
    for edge in hop1_clean + hop2_clean:
        src, tgt = edge["source_id"], edge["target_id"]
        w = edge["weight"] if edge["weight"] is not None else 1.0
        adj.setdefault(src, []).append((tgt, w))
        adj.setdefault(tgt, []).append((src, w))

    if not adj:
        return {}

    raw_ppr = personalized_pagerank(adj, seed_weights, damping=PPR_DAMPING)
    # Return non-seed scores above minimum threshold
    return {mid: ps for mid, ps in raw_ppr.items()
            if mid not in seed_set and mid not in exclude_set and ps > PPR_MIN_SCORE}


def impl_recall(
    query: str,
    context: str = "",
    budget: int = 5000,
    category: str = "",
    limit: int = 5,
    exclude_ids: str = "[]",
    since: str = "",
    before: str = "",
    boost_themes: str = "[]",
    min_priority: int = 0,
    internal: bool = False,
) -> str:
    db = get_db()
    try:
        # Parse filter parameters
        try:
            exclude_set = set(json.loads(exclude_ids)) if exclude_ids else set()
        except (json.JSONDecodeError, TypeError):
            exclude_set = set()

        try:
            boost_themes_list = json.loads(boost_themes) if boost_themes else []
        except (json.JSONDecodeError, TypeError):
            boost_themes_list = []

        # Parse since/before into ISO date strings
        since_date = ""
        if since:
            rel_match = re.match(r"^(\d+)d$", since.strip())
            if rel_match:
                days = int(rel_match.group(1))
                since_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            else:
                since_date = since  # assume ISO date

        before_date = before

        # Generate query embedding — use context for vector search if provided
        vector_input = context.strip() if context and context.strip() else query
        query_embedding = embed_text(vector_input)

        # --- Vector search ---
        vec_k = max(limit * 5, 200)
        vec_results = db.execute(
            """
            SELECT rowid, distance FROM memory_vec
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (serialize_f32(query_embedding), vec_k),
        ).fetchall()

        # Map rowids to memory_ids (keep raw distances for reranker)
        vec_ranked = {}  # memory_id -> rank (0-based)
        vec_distances = {}  # memory_id -> raw cosine distance
        for rank, row in enumerate(vec_results):
            mapped = db.execute(
                "SELECT memory_id FROM memory_rowid_map WHERE rowid = ?",
                (row["rowid"],),
            ).fetchone()
            if mapped:
                vec_ranked[mapped["memory_id"]] = rank
                vec_distances[mapped["memory_id"]] = row["distance"]

        # --- Keyword search (keep raw BM25 scores for reranker) ---
        fts_ranked = {}  # memory_id -> rank (0-based)
        fts_bm25_scores = {}  # memory_id -> raw BM25 score
        fts_query = sanitize_fts_query(query)
        try:
            fts_results = db.execute(
                """
                SELECT rowid, bm25(memory_fts, 5.0, 3.0) as rank FROM memory_fts
                WHERE memory_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, max(limit * 2, 150)),
            ).fetchall()

            for rank, row in enumerate(fts_results):
                mapped = db.execute(
                    "SELECT memory_id FROM memory_rowid_map WHERE rowid = ?",
                    (row["rowid"],),
                ).fetchone()
                if mapped:
                    fts_ranked[mapped["memory_id"]] = rank
                    fts_bm25_scores[mapped["memory_id"]] = row["rank"]
        except sqlite3.OperationalError:
            # FTS query syntax error — fall back to vector-only
            pass

        # --- Candidate pool ---
        all_ids = set(vec_ranked.keys()) | set(fts_ranked.keys())

        # Filter out non-active memories early
        if all_ids:
            active_ph = ",".join("?" * len(all_ids))
            active_rows = db.execute(
                f"SELECT id FROM memories WHERE id IN ({active_ph}) AND status = 'active'",
                list(all_ids),
            ).fetchall()
            all_ids = {r["id"] for r in active_rows}

        # --- Scoring: reranker or formula ---
        reranker_model = get_reranker_model()
        expansion_new = expansion_boosted = per_seed_cap_hits = 0
        total_cap_hit = False

        if reranker_model is not None:
            # Reranker path: compute raw PPR scores for features
            ppr_scores = _compute_ppr_for_reranker(db, all_ids, vec_ranked, fts_ranked, exclude_set)

            # Get themes_map for feature extraction
            themes_map = {}
            if all_ids:
                t_ph = ",".join("?" * len(all_ids))
                theme_rows = db.execute(f"""
                    SELECT id, themes FROM memories WHERE id IN ({t_ph})
                """, list(all_ids)).fetchall()
                themes_map = {r["id"]: r["themes"] for r in theme_rows}

            # Extract features and predict
            features, candidate_ids = extract_live_features(
                db, vec_ranked, fts_ranked, fts_bm25_scores, vec_distances,
                all_ids, ppr_scores, {}, themes_map, query,
            )
            if len(candidate_ids) > 0:
                rrf_scores = rerank(reranker_model, features, candidate_ids)
            else:
                rrf_scores = {}
            feedback_map = {}
            sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
            top_score = rrf_scores[sorted_ids[0]] if sorted_ids else 0.0
        else:
            # Formula path (existing)
            rrf_scores, feedback_map, themes_map = rrf_fuse(
                db, vec_ranked, fts_ranked, all_ids, boost_themes_list)
            apply_hebbian(db, rrf_scores)
            expansion_new, expansion_boosted, per_seed_cap_hits, total_cap_hit = expand_via_ppr(
                db, rrf_scores, query_embedding, query, exclude_set)
            sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
            top_score = rrf_scores[sorted_ids[0]] if sorted_ids else 0.0

        # Fetch memories and apply filters (post-RRF), capped by limit
        results = []
        for mid in sorted_ids:
            if mid in exclude_set:
                continue
            mem = db.execute(
                "SELECT * FROM memories WHERE id = ? AND status = 'active'", (mid,)
            ).fetchone()
            if mem is None:
                continue
            if category and mem["category"] != category:
                continue
            if min_priority and mem["base_priority"] < min_priority:
                continue
            if since_date and mem["created_at"] < since_date:
                continue
            if before_date and mem["created_at"] >= before_date:
                continue
            results.append(mem)
            if len(results) >= limit:
                break

        if not results:
            return f"No memories found for query: {query}"

        # Format within budget
        header = f'## Recall: "{query}" ({len(results)} results, limit={limit})'
        output_lines = [header + "\n"]
        tokens_used = count_tokens(output_lines[0])
        returned_ids = []

        for i, mem in enumerate(results):
            if i < 3:
                formatted = format_memory_full(mem)
            else:
                formatted = format_memory_compact(mem) + f"  (ID: {mem['id'][:8]})"
            line_tokens = count_tokens(formatted)
            if tokens_used + line_tokens > budget - 50:
                output_lines.append(f"... {len(results) - i} more results truncated by budget")
                break
            output_lines.append(formatted)
            returned_ids.append(mem["id"])
            tokens_used += line_tokens

        # Update access tracking (recall phase)
        now = datetime.now(timezone.utc).isoformat()
        for mid in returned_ids:
            db.execute(
                "UPDATE memories SET last_accessed = ?, recall_count = recall_count + 1, access_count = access_count + 1 WHERE id = ?",
                (now, mid),
            )

        # Check for contradictions in result set
        result_ids = [mem["id"] for mem in results]
        contradiction_notes = []
        if len(result_ids) >= 2:
            r_ph = ",".join("?" * len(result_ids))
            contra_edges = db.execute(f"""
                SELECT source_id, target_id, linking_context FROM memory_edges
                WHERE flags LIKE '%contradiction%'
                AND source_id IN ({r_ph}) AND target_id IN ({r_ph})
            """, result_ids + result_ids).fetchall()
            for ce in contra_edges:
                ctx = ce["linking_context"] or ""
                contradiction_notes.append(
                    f"Contradiction between {ce['source_id'][:8]} and {ce['target_id'][:8]}"
                    + (f": {ctx}" if ctx else "")
                )

        if contradiction_notes:
            output_lines.append("\n**Contradictions detected in results:**")
            for cn in contradiction_notes:
                output_lines.append(f"  - {cn}")

        if not internal:
            # Log retrieval events (with expansion metadata)
            expansion_meta = {
                "edge_expanded": expansion_new,
                "edge_boosted": expansion_boosted,
                "novelty_scored": True,
                "per_seed_cap_hits": per_seed_cap_hits,
                "total_cap_hit": total_cap_hit,
            }
            expanded_ids = {mid for mid in returned_ids if mid not in all_ids}
            for mid in returned_ids:
                per_mem_meta = {**expansion_meta}
                if mid in expanded_ids:
                    per_mem_meta["from_expansion"] = True
                _log_event(db, mid, "retrieved",
                           query=query,
                           co_memory_ids=[x for x in returned_ids if x != mid],
                           similarity_score=rrf_scores.get(mid),
                           context=per_mem_meta)

            # Log recall_meta summary
            returned_ids_set = set(returned_ids)
            kept_scores = [(mid[:8], round(rrf_scores[mid], 5)) for mid in returned_ids]
            beyond_limit = [(mid[:8], round(rrf_scores[mid], 5))
                            for mid in sorted_ids if mid not in returned_ids_set]
            recall_meta = {
                "top_score": round(top_score, 5),
                "limit": limit,
                "kept": kept_scores,
            }
            if context and context.strip() and context.strip() != query:
                recall_meta["vector_input"] = context.strip()
            if beyond_limit:
                recall_meta["beyond_limit"] = beyond_limit
            _log_event(db, "_recall", "recall_meta", query=query, context=recall_meta)

        db.commit()
    finally:
        db.close()

    if not internal:
        # Append compact feedback template
        sids = [mid[:8] for mid in returned_ids]
        all_ids = " ".join(sids)
        output_lines.append(
            f"\n---\nrecall_feedback IDs: {all_ids}\n"
            '  Format: {id: utility} (0.0-1.0) or {id: [utility, durability]} where durability is -1.0..1.0'
        )

    return "\n".join(output_lines)


def impl_recall_feedback(
    feedback: str = "{}",
    query: str = "",
    reason: str = "",
    cutoff_rank: int = -1,
) -> str:
    try:
        fb = json.loads(feedback) if isinstance(feedback, str) else feedback
    except (json.JSONDecodeError, TypeError):
        return "Invalid feedback JSON. Expected {id: score} or {id: [utility, durability]}."

    if not isinstance(fb, dict):
        return "Invalid feedback format. Expected JSON object."

    if not fb:
        return "No memory feedback provided."

    db = get_db()
    recorded = 0
    decay_adjusted = 0
    resolved_feedback = []  # (resolved_mid, utility) pairs for co-utility edges

    for mid, value in fb.items():
        # Parse value: float or [utility, durability]
        durability = 0.0
        if isinstance(value, (int, float)):
            utility = float(value)
        elif isinstance(value, list) and len(value) >= 1:
            utility = float(value[0])
            if len(value) >= 2:
                durability = float(value[1])
        else:
            continue

        # Clamp
        utility = max(0.0, min(1.0, utility))
        durability = max(-1.0, min(1.0, durability))

        # Resolve short ID prefix
        resolved, _err = _resolve_id(db, mid, status_filter="active")
        if resolved:
            mid = resolved

        # Verify memory exists and is active
        mem = db.execute(
            "SELECT id, decay_rate, category, confidence, themes, content, summary, base_priority, last_accessed, flags FROM memories WHERE id = ? AND status = 'active'", (mid,)
        ).fetchone()
        if not mem:
            continue

        # Log feedback event
        event_context = {"utility": round(utility, 4)}
        if abs(durability) > 0.01:
            event_context["durability"] = round(durability, 4)
        if reason:
            event_context["reason"] = reason

        # Signal enrichment for parameter tuning
        event_context["confidence_before"] = round(mem["confidence"] if mem["confidence"] is not None else 0.5, 4)

        _last_accessed = mem["last_accessed"] or ""
        if _last_accessed:
            try:
                _la = datetime.fromisoformat(_last_accessed)
                if _la.tzinfo is None:
                    _la = _la.replace(tzinfo=timezone.utc)
                age_days = (datetime.now(timezone.utc) - _la).total_seconds() / 86400
                event_context["age_days"] = round(age_days, 1)
            except (ValueError, TypeError):
                pass

        _flags = json.loads(mem["flags"] or "[]") if mem["flags"] else []
        eff_pri = effective_priority(
            mem["base_priority"], mem["last_accessed"],
            mem["decay_rate"], mem["category"], _flags
        )
        event_context["effective_priority"] = round(eff_pri, 3)

        _log_event(db, mid, "feedback", query=query or None, context=event_context)
        recorded += 1
        resolved_feedback.append((mid, utility))

        # Adjust decay_rate if durability signal provided
        if abs(durability) > 0.01:
            current_rate = mem["decay_rate"]
            if current_rate is None:
                current_rate = CATEGORY_DECAY_RATES.get(mem["category"], DEFAULT_DECAY_RATE)

            if current_rate != 0:
                if durability > 0:
                    new_rate = max(0.0, current_rate * (1.0 - DECAY_DURABILITY_SCALE * durability))
                else:
                    new_rate = min(2.0, current_rate * (1.0 + DECAY_DURABILITY_SCALE * abs(durability)))
                db.execute("UPDATE memories SET decay_rate = ? WHERE id = ?", (new_rate, mid))
                decay_adjusted += 1

        # Confidence adjustment — continuous function of utility
        fetched_conf = mem["confidence"] if mem["confidence"] is not None else CONF_DEFAULT
        conf_delta = 0.0

        if utility >= CONF_USEFUL_THRESHOLD:
            conf_delta = CONF_GROWTH_RATE * utility * (1.0 - fetched_conf)
        elif utility < CONF_USELESS_THRESHOLD:
            conf_delta = fetched_conf * CONF_DECAY_RATE * -1

        # Durability nudges confidence continuously
        if abs(durability) > 0.01:
            if durability > 0:
                conf_delta += durability * CONF_DURABILITY_NUDGE * (1.0 - (fetched_conf + conf_delta))
            else:
                conf_delta += durability * CONF_DURABILITY_NUDGE

        if abs(conf_delta) > CONF_MIN_DELTA:
            db.execute(
                "UPDATE memories SET confidence = "
                "MAX(0.1, MIN(0.95, COALESCE(confidence, 0.5) + ?)) WHERE id = ?",
                (round(conf_delta, 4), mid),
            )

        # Theme refinement: close FTS gaps for useful memories
        if utility >= THEME_REFINE_THRESHOLD and query:
            _STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                           "at", "to", "for", "of", "with", "and", "or", "not", "how",
                           "what", "when", "where", "why", "who", "which", "this", "that",
                           "my", "your", "do", "does", "did", "have", "has", "had", "be",
                           "been", "being", "will", "would", "could", "should", "can",
                           "about", "from", "it", "its", "i", "me", "we", "they"}
            current_themes = json.loads(mem["themes"] or "[]")
            current_themes_lower = {t.lower() for t in current_themes}
            content_lower = (mem["content"] or "").lower()
            query_terms = [t.lower().strip("?.,!") for t in query.split()
                           if len(t) > 2 and t.lower().strip("?.,!") not in _STOP_WORDS]
            # Content-match terms
            content_terms = [t for t in query_terms
                             if t not in current_themes_lower and t in content_lower]
            # Bridge terms: query vocabulary not in content or summary (higher bar)
            bridge_terms = []
            if utility >= BRIDGE_THEME_THRESHOLD:
                summary_lower = (mem["summary"] or "").lower()
                bridge_terms = [t for t in query_terms
                                if t not in current_themes_lower
                                and t not in content_lower
                                and t not in summary_lower
                                and len(t) >= BRIDGE_TERM_MIN_LEN]
            new_terms = content_terms + bridge_terms
            if new_terms and len(current_themes) + len(new_terms) > MAX_THEMES:
                _log_event(db, mid, "theme_cap_hit", context={
                    "current_count": len(current_themes),
                    "blocked_terms": new_terms[:5],
                    "utility": utility,
                })
            if new_terms and len(current_themes) + len(new_terms) <= MAX_THEMES:
                updated = normalize_themes(current_themes + new_terms[:MAX_NEW_TERMS], content=mem["content"] or "")
                db.execute("UPDATE memories SET themes = ? WHERE id = ?",
                           (json.dumps(updated), mid))
                # Update FTS index
                fts_rowid = db.execute(
                    "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (mid,)
                ).fetchone()
                if fts_rowid:
                    db.execute("DELETE FROM memory_fts WHERE rowid = ?", (fts_rowid["rowid"],))
                    fts_themes = _themes_for_fts(json.dumps(updated))
                    db.execute(
                        "INSERT INTO memory_fts (rowid, summary, themes) VALUES (?, ?, ?)",
                        (fts_rowid["rowid"], mem["summary"] or "", fts_themes),
                    )

        # Edge weight propagation: reinforce/weaken connected edges
        connected_edges = db.execute(
            "SELECT rowid, weight FROM memory_edges WHERE source_id = ? OR target_id = ?",
            (mid, mid),
        ).fetchall()
        for ce in connected_edges:
            current_w = ce["weight"] if ce["weight"] is not None else 0.5
            if utility >= 0.5:
                new_w = min(1.0, current_w + EDGE_WEIGHT_POS_STEP)
            else:
                new_w = max(0.0, current_w - EDGE_WEIGHT_NEG_STEP)
            if new_w != current_w:
                db.execute("UPDATE memory_edges SET weight = ? WHERE rowid = ?", (new_w, ce["rowid"]))
                _log_event(db, mid, "edge_weight_change", context={
                    "edge_rowid": ce["rowid"],
                    "weight_before": round(current_w, 4),
                    "weight_after": round(new_w, 4),
                    "utility": round(utility, 4),
                })

    # Co-utility edges: link memories that were useful together
    edges_created = 0
    high_utility_ids = [mid for mid, u in resolved_feedback if u >= CO_UTILITY_THRESHOLD]
    if len(high_utility_ids) >= 2 and query:
        co_ids = high_utility_ids[:5]  # cap to avoid quadratic explosion
        try:
            query_emb = embed_text(query)
        except Exception:
            query_emb = None
        for i, src in enumerate(co_ids):
            for tgt in co_ids[i + 1:]:
                edge_id = _create_edge(
                    db, src, tgt,
                    linking_context=query,
                    linking_embedding=query_emb,
                    created_by="co_retrieval",
                )
                if edge_id:
                    edges_created += 1

    # Disappointed-recall detection: all memories scored low utility
    if resolved_feedback and query:
        max_util = max(u for _, u in resolved_feedback)
        if max_util <= DISAPPOINTED_RECALL_MAX_UTIL:
            meta_row = db.execute(
                """SELECT context FROM memory_events
                   WHERE memory_id = '_recall' AND event_type = 'recall_meta'
                   AND query = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (query,)
            ).fetchone()

            top_score = 0.0
            total_candidates = 0
            quality_dropped = 0
            if meta_row:
                meta = json.loads(meta_row[0])
                top_score = meta.get("top_score", 0.0)
                total_candidates = len(meta.get("kept", [])) + len(meta.get("dropped", []))
                quality_dropped = len(meta.get("dropped", []))

            if top_score >= DISAPPOINTED_RETRIEVAL_SCORE:
                miss_type = "retrieval_failure"
            else:
                miss_type = "coverage_gap"

            miss_event = {
                "miss_type": miss_type,
                "max_utility": round(max_util, 4),
                "memories_rated": len(resolved_feedback),
                "top_score": round(top_score, 5),
                "total_candidates": total_candidates,
                "quality_dropped": quality_dropped,
            }
            _log_event(db, "_recall", "recall_miss", query=query, context=miss_event)

    # Log cutoff annotation if provided
    cutoff_msg = ""
    if cutoff_rank >= 0 and query:
        row = db.execute(
            """SELECT context FROM memory_events
               WHERE memory_id = '_recall' AND event_type = 'recall_meta'
               AND query = ?
               ORDER BY created_at DESC LIMIT 1""",
            (query,)
        ).fetchone()
        if row:
            meta = json.loads(row[0])
            kept = meta.get("kept", [])
            total_returned = len(kept)
            top_score = meta.get("top_score", 0)

            last_useful_score = (
                kept[cutoff_rank - 1][1]
                if cutoff_rank > 0 and cutoff_rank <= len(kept)
                else None
            )
            first_noise_score = (
                kept[cutoff_rank][1]
                if cutoff_rank < len(kept)
                else None
            )

            cutoff_event: dict = {
                "cutoff_rank": cutoff_rank,
                "total_returned": total_returned,
                "top_score": top_score,
            }
            if last_useful_score is not None:
                cutoff_event["last_useful_score"] = round(last_useful_score, 5)
                cutoff_event["cutoff_ratio"] = (
                    round(last_useful_score / top_score, 5) if top_score > 0 else 0
                )
            if first_noise_score is not None:
                cutoff_event["first_noise_score"] = round(first_noise_score, 5)
            if last_useful_score is not None and first_noise_score is not None:
                gap = last_useful_score - first_noise_score
                cutoff_event["gap"] = round(gap, 5)
                cutoff_event["gap_ratio"] = (
                    round(gap / top_score, 5) if top_score > 0 else 0
                )

            _log_event(db, "_recall", "recall_cutoff", query=query, context=cutoff_event)
            cutoff_msg = f" Cutoff logged at rank {cutoff_rank}/{total_returned}."

    db.commit()
    db.close()

    parts = [f"Feedback recorded for {recorded} memories."]
    if decay_adjusted:
        parts.append(f"Decay rate adjusted for {decay_adjusted}.")
    if edges_created:
        parts.append(f"{edges_created} co-utility edge(s) created.")
    if cutoff_msg:
        parts.append(cutoff_msg.strip())
    return " ".join(parts)


def impl_link(
    source_id: str,
    target_id: str,
    linking_context: str,
    flags: str = "[]",
) -> str:
    db = get_db()
    try:
        # Resolve IDs
        resolved_source, err = _resolve_id(db, source_id, status_filter="active")
        if not resolved_source:
            return f"Source memory not found: {err}"
        resolved_target, err = _resolve_id(db, target_id, status_filter="active")
        if not resolved_target:
            return f"Target memory not found: {err}"

        if resolved_source == resolved_target:
            return "Cannot link a memory to itself."

        # Parse flags
        try:
            parsed_flags = json.loads(flags) if isinstance(flags, str) else flags
        except json.JSONDecodeError:
            parsed_flags = []

        # Embed linking_context for novelty-scored adjacency expansion
        linking_emb = None
        if linking_context:
            try:
                linking_emb = embed_text(linking_context)
            except Exception:
                pass  # Edge still useful without embedding

        edge_id = _create_edge(
            db,
            source_id=resolved_source,
            target_id=resolved_target,
            linking_context=linking_context,
            linking_embedding=linking_emb,
            flags=parsed_flags,
            created_by="session",
        )

        db.commit()

        if not edge_id:
            return "Edge already exists between these memories (duplicate)."

        # Format response
        src_short = resolved_source[:8]
        tgt_short = resolved_target[:8]
        flag_str = f" [{', '.join(parsed_flags)}]" if parsed_flags else ""
        return f"Edge created: {src_short} -> {tgt_short}{flag_str}\nContext: {linking_context}"
    finally:
        db.close()


def impl_forget(memory_id: str) -> str:
    db = get_db()
    resolved, err = _resolve_id(db, memory_id)
    if err:
        db.close()
        return err
    memory_id = resolved
    mem = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not mem:
        db.close()
        return f"Memory not found: {memory_id}"
    if mem["status"] == "deleted":
        db.close()
        return f"Memory already deleted: {memory_id}"

    db.execute(
        "UPDATE memories SET status = 'deleted' WHERE id = ?", (memory_id,)
    )

    # Clean up vector, FTS, and rowid mapping for deleted memory
    vec_map = db.execute(
        "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    if vec_map:
        db.execute("DELETE FROM memory_vec WHERE rowid = ?", (vec_map["rowid"],))
        db.execute("DELETE FROM memory_fts WHERE rowid = ?", (vec_map["rowid"],))
        db.execute("DELETE FROM memory_rowid_map WHERE rowid = ?", (vec_map["rowid"],))

    _log_event(db, memory_id, "updated", context={"action": "deleted"})
    db.commit()
    db.close()

    summary = mem["summary"] or mem["content"][:60]
    return f"Deleted: [{mem['category']}] {summary}\nID: {memory_id}"


def impl_reflect(memory_id: str) -> str:
    db = get_db()
    resolved, err = _resolve_id(db, memory_id)
    if err:
        db.close()
        return err
    memory_id = resolved
    mem = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not mem:
        db.close()
        return f"Memory not found: {memory_id}"

    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        "UPDATE memories SET last_accessed = ?, reflect_count = reflect_count + 1 WHERE id = ?",
        (now, memory_id),
    )
    db.commit()
    db.close()

    summary = mem["summary"] or mem["content"][:60]
    return f"Reheated: [{mem['category']}] {summary}"


def impl_review_pending(
    action: str = "list",
    memory_id: str = "",
    edit_content: str = "",
) -> str:
    db = get_db()
    try:
        if action == "list":
            rows = db.execute(
                "SELECT * FROM memories WHERE status = 'pending' ORDER BY created_at DESC"
            ).fetchall()
            if not rows:
                return "No pending memories."
            lines = [f"## Pending Memories ({len(rows)})\n"]
            for i, row in enumerate(rows, 1):
                lines.append(format_memory_pending(row, i))
            return "\n\n".join(lines)

        elif action == "confirm":
            if not memory_id:
                return "memory_id required for confirm action."
            resolved, err = _resolve_id(db, memory_id, status_filter="pending")
            if err:
                return err
            memory_id = resolved
            mem = db.execute(
                "SELECT * FROM memories WHERE id = ? AND status = 'pending'",
                (memory_id,),
            ).fetchone()
            if not mem:
                return f"Pending memory not found: {memory_id}"

            updates = ["status = 'active'"]
            params = []
            if edit_content:
                updates.append("content = ?")
                params.append(edit_content)
                # Re-embed if content changed
                themes_list = json.loads(mem["themes"]) if mem["themes"] else []
                enriched = build_enriched_text(edit_content, mem["category"], themes_list, mem["summary"] or "")
                embedding = embed_text(enriched)
                vec_rowid = db.execute(
                    "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()["rowid"]
                db.execute(
                    "UPDATE memory_vec SET embedding = ? WHERE rowid = ?",
                    (serialize_f32(embedding), vec_rowid),
                )
                updates.append("token_count = ?")
                params.append(count_tokens(edit_content))

            params.append(memory_id)
            db.execute(
                f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params
            )
            db.commit()
            summary = mem["summary"] or mem["content"][:60]
            return f"Confirmed: [{mem['category']}] {summary}"

        elif action == "confirm_all":
            pending = db.execute(
                "SELECT id FROM memories WHERE status = 'pending'"
            ).fetchall()
            db.execute(
                "UPDATE memories SET status = 'active' WHERE status = 'pending'"
            )
            db.commit()
            return f"Confirmed {len(pending)} pending memories."

        elif action == "discard":
            if not memory_id:
                return "memory_id required for discard action."
            resolved, err = _resolve_id(db, memory_id, status_filter="pending")
            if err:
                return err
            memory_id = resolved
            mem = db.execute(
                "SELECT * FROM memories WHERE id = ? AND status = 'pending'",
                (memory_id,),
            ).fetchone()
            if not mem:
                return f"Pending memory not found: {memory_id}"

            db.execute(
                "UPDATE memories SET status = 'deleted' WHERE id = ?", (memory_id,)
            )
            db.commit()
            summary = mem["summary"] or mem["content"][:60]
            return f"Discarded: [{mem['category']}] {summary}"

        elif action == "discard_all":
            pending = db.execute(
                "SELECT id FROM memories WHERE status = 'pending'"
            ).fetchall()
            db.execute(
                "UPDATE memories SET status = 'deleted' WHERE status = 'pending'"
            )
            db.commit()
            return f"Discarded {len(pending)} pending memories."

        else:
            return f"Unknown action '{action}'. Use: list, confirm, confirm_all, discard, discard_all."
    finally:
        db.close()


def impl_consolidate(category: str = "") -> str:
    db = get_db()
    try:
        report = []

        # 1. Prune orphaned embedding rows (deleted/missing memories)
        orphans = db.execute("""
            SELECT mrm.rowid
            FROM memory_rowid_map mrm
            LEFT JOIN memories m ON mrm.memory_id = m.id
            WHERE m.id IS NULL OR m.status = 'deleted'
        """).fetchall()
        if orphans:
            for o in orphans:
                db.execute("DELETE FROM memory_vec WHERE rowid = ?", (o["rowid"],))
                db.execute("DELETE FROM memory_fts WHERE rowid = ?", (o["rowid"],))
                db.execute("DELETE FROM memory_rowid_map WHERE rowid = ?", (o["rowid"],))
            report.append(f"Pruned {len(orphans)} orphaned embedding row(s).")

        # 2. Archive old retrieval events (keep 90 days)
        event_cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        old_events = db.execute(
            "SELECT * FROM memory_events WHERE event_type = 'retrieved' AND created_at < ?",
            (event_cutoff,)
        ).fetchall()
        if old_events:
            archive_path = DATA_DIR / "memory_events_archive.db"
            archive = sqlite3.connect(str(archive_path))
            archive.execute("""
                CREATE TABLE IF NOT EXISTS memory_events (
                    id INTEGER PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    query TEXT,
                    session_id TEXT,
                    co_memory_ids TEXT DEFAULT '[]',
                    similarity_score REAL,
                    context TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
            """)
            for ev in old_events:
                archive.execute(
                    "INSERT OR IGNORE INTO memory_events VALUES (?,?,?,?,?,?,?,?,?)",
                    (ev["id"], ev["memory_id"], ev["event_type"], ev["query"],
                     ev["session_id"], ev["co_memory_ids"], ev["similarity_score"],
                     ev["context"], ev["created_at"]),
                )
            archive.commit()
            archive.close()
            db.execute(
                "DELETE FROM memory_events WHERE event_type = 'retrieved' AND created_at < ?",
                (event_cutoff,)
            )
            report.append(f"Archived {len(old_events)} retrieval events older than 90 days.")

        # 3. Clean up stale pending memories
        cutoff = datetime.now(timezone.utc)
        stale_pending = db.execute(
            "SELECT * FROM memories WHERE status = 'pending'"
        ).fetchall()
        stale_count = 0
        for mem in stale_pending:
            created = datetime.fromisoformat(mem["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            days_old = (cutoff - created).total_seconds() / 86400
            if days_old > PENDING_STALE_DAYS:
                db.execute(
                    "UPDATE memories SET status = 'deleted' WHERE id = ?",
                    (mem["id"],),
                )
                stale_count += 1
        if stale_count:
            report.append(f"Auto-discarded {stale_count} stale pending memories (>{PENDING_STALE_DAYS} days old).")

        # 4. Find near-duplicate clusters among active memories
        cat_filter = "AND category = ?" if category else ""
        cat_params = (category,) if category else ()

        active = db.execute(
            f"SELECT id, content, category, base_priority, summary FROM memories WHERE status = 'active' {cat_filter}",
            cat_params,
        ).fetchall()

        if len(active) < 2:
            db.commit()
            if report:
                return "\n".join(report) + "\nNo duplicates found among active memories."
            return "No duplicates found. Memory store is clean."

        # For each memory, check against others for high similarity
        merged_count = 0
        processed = set()

        for mem in active:
            if mem["id"] in processed:
                continue

            vec_rowid = db.execute(
                "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?",
                (mem["id"],),
            ).fetchone()
            if not vec_rowid:
                continue

            nearby = db.execute(
                """
                SELECT rowid, distance FROM memory_vec
                WHERE embedding MATCH (SELECT embedding FROM memory_vec WHERE rowid = ?)
                AND k = 10
                ORDER BY distance
                """,
                (vec_rowid["rowid"],),
            ).fetchall()

            for near in nearby:
                dist = near["distance"]
                if dist is None or dist > 0.15 or dist == 0.0:
                    continue
                near_map = db.execute(
                    "SELECT memory_id FROM memory_rowid_map WHERE rowid = ?",
                    (near["rowid"],),
                ).fetchone()
                if not near_map or near_map["memory_id"] in processed:
                    continue

                near_mem = db.execute(
                    "SELECT * FROM memories WHERE id = ? AND status = 'active'",
                    (near_map["memory_id"],),
                ).fetchone()
                if not near_mem:
                    continue
                if category and near_mem["category"] != mem["category"]:
                    continue

                # Found a near-duplicate. Keep higher priority, supersede the other.
                if near_mem["base_priority"] >= mem["base_priority"]:
                    winner_id = near_mem["id"]
                    loser_id = mem["id"]
                else:
                    winner_id = mem["id"]
                    loser_id = near_mem["id"]

                db.execute(
                    "UPDATE memories SET superseded_by = ?, status = 'deleted' WHERE id = ?",
                    (winner_id, loser_id),
                )
                processed.add(loser_id)
                merged_count += 1
                loser_summary = near_mem["summary"] or near_mem["content"][:60]
                report.append(f"  Merged: '{loser_summary}' -> kept {winner_id[:8]}...")

        if merged_count:
            report.insert(
                len(report) - merged_count,
                f"Found and merged {merged_count} near-duplicate(s):",
            )

        db.commit()

        if not report:
            return "Memory store is clean. No duplicates or stale pending found."
        return "\n".join(report)
    finally:
        db.close()


def impl_reembed_all() -> str:
    import time
    start = time.time()
    db = get_db()
    try:
        rows = db.execute(
            "SELECT * FROM memories WHERE status IN ('active', 'pending')"
        ).fetchall()

        if not rows:
            return "No active or pending memories to re-embed."

        count = 0
        for row in rows:
            themes_list = json.loads(row["themes"]) if row["themes"] else []
            enriched = build_enriched_text(
                row["content"], row["category"], themes_list, row["summary"] or ""
            )
            embedding = embed_text(enriched)

            vec_rowid = db.execute(
                "SELECT rowid FROM memory_rowid_map WHERE memory_id = ?",
                (row["id"],),
            ).fetchone()
            if not vec_rowid:
                continue

            db.execute(
                "UPDATE memory_vec SET embedding = ? WHERE rowid = ?",
                (serialize_f32(embedding), vec_rowid["rowid"]),
            )
            count += 1

        db.commit()
        elapsed = time.time() - start
        return f"Re-embedded {count} memories in {elapsed:.2f}s."
    finally:
        db.close()


def impl_memory_stats() -> str:
    db = get_db()
    result = compute_stats(db)
    db.close()
    return result
