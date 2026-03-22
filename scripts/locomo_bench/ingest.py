"""LoCoMo dataset loading and ingestion into Somnigraph."""

import hashlib
import json
import logging
import pickle
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_locomo(path: Path) -> list[dict]:
    """Load LoCoMo JSON and return list of 10 conversations."""
    with open(path) as f:
        data = json.load(f)
    logger.info("Loaded %d conversations from LoCoMo", len(data))
    return data


def parse_session_datetime(date_str: str) -> str:
    """Parse LoCoMo date strings like '1:56 pm on 8 May, 2023' to ISO format."""
    date_str = date_str.strip()
    formats = [
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %B %d, %Y",
        "%I:%M%p on %d %B, %Y",
        "%I:%M%p on %B %d, %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.isoformat()
        except ValueError:
            continue
    return "2023-01-01T00:00:00"


def extract_turns(conversations: list[dict]) -> list[dict]:
    """Extract all turns from all conversations as flat list."""
    turns = []
    for conv_idx, conv in enumerate(conversations):
        c = conv["conversation"]
        for sess_num in range(1, 50):
            sess_key = f"session_{sess_num}"
            date_key = f"session_{sess_num}_date_time"
            if sess_key not in c:
                break
            session_date = parse_session_datetime(c.get(date_key, ""))
            for turn in c[sess_key]:
                turns.append({
                    "id": str(uuid.uuid4()),
                    "conv_id": conv_idx,
                    "session": sess_num,
                    "dia_id": turn["dia_id"],
                    "speaker": turn["speaker"],
                    "text": turn["text"],
                    "created_at": session_date,
                })
    return turns


def extract_questions(conversations: list[dict]) -> list[dict]:
    """Extract all QA pairs with evidence pointers."""
    questions = []
    for conv_idx, conv in enumerate(conversations):
        for qa_idx, qa in enumerate(conv["qa"]):
            evidence = qa.get("evidence", [])
            answer = qa.get("answer", qa.get("adversarial_answer", ""))
            questions.append({
                "conv_id": conv_idx,
                "qa_idx": qa_idx,
                "question": qa["question"],
                "answer": answer,
                "evidence": evidence,
                "category": qa.get("category"),
            })
    return questions


# ---------------------------------------------------------------------------
# Embedding cache (reuses existing cache from prior bench_locomo.py runs)
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """Disk-backed embedding cache keyed by text hash."""

    def __init__(self, path: Path):
        self.path = path
        self.cache: dict[str, list[float]] = {}
        if path.exists():
            with open(path, "rb") as f:
                self.cache = pickle.load(f)
            logger.info("Loaded %d cached embeddings from %s",
                        len(self.cache), path.name)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str) -> list[float] | None:
        return self.cache.get(self._key(text))

    def put(self, text: str, embedding: list[float]):
        self.cache[self._key(text)] = embedding

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.cache, f)

    def missing_indices(self, texts: list[str]) -> list[int]:
        """Return indices of texts not in cache."""
        return [i for i, t in enumerate(texts) if self.get(t) is None]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest_conversation(
    conv_turns: list[dict],
    embed_cache_path: Path | None = None,
    enrich: bool = False,
) -> dict[tuple, str]:
    """Ingest turns for one conversation directly into the DB.

    Bypasses impl_remember to avoid:
    1. Dedup rejection (conversation turns are semantically similar)
    2. Wrong timestamps (impl_remember stamps now, not the LoCoMo date)

    Uses batch embedding with disk cache for efficiency.
    If enrich=True, uses GPT-4.1-mini to extract topics/facts/entities
    and stores them in themes and summary for better retrieval.

    Returns mapping of (conv_id, dia_id) -> memory_id for evidence matching.
    """
    from memory.db import get_db
    from memory.embeddings import build_enriched_text, embed_batch
    from memory.write import _insert_memory

    # Load or create embedding cache
    cache = None
    if embed_cache_path:
        cache = EmbeddingCache(embed_cache_path)

    # Optionally enrich with LLM-extracted metadata
    if enrich:
        logger.info("  Enriching %d turns with LLM metadata...", len(conv_turns))
        enrichments = _batch_enrich(conv_turns)
    else:
        enrichments = [None] * len(conv_turns)

    # Build all content and enriched texts
    contents = []
    enriched_texts = []
    themes_lists = []
    summaries = []
    for turn, enrichment in zip(conv_turns, enrichments):
        content = f"[{turn['speaker']}] {turn['text']}"
        themes_list = [turn["speaker"], f"session_{turn['session']}"]

        if enrichment:
            # Add extracted topics to themes for FTS
            topics = enrichment.get("topics", [])
            entities = enrichment.get("entities", [])
            themes_list.extend(topics)
            themes_list.extend(entities)
            # Use extracted facts as summary for better vector search
            facts = enrichment.get("facts", [])
            summary = "; ".join(facts) if facts else content[:80]
        else:
            summary = content[:80]

        enriched = build_enriched_text(content, "episodic", themes_list, summary)
        contents.append(content)
        enriched_texts.append(enriched)
        themes_lists.append(themes_list)
        summaries.append(summary)

    # Batch embed (using cache where available)
    if cache:
        missing = cache.missing_indices(enriched_texts)
        if missing:
            logger.info("  Embedding %d new texts (%d cached)...",
                        len(missing), len(enriched_texts) - len(missing))
            miss_texts = [enriched_texts[i] for i in missing]
            miss_embeddings = embed_batch(miss_texts)
            for idx, emb in zip(missing, miss_embeddings):
                cache.put(enriched_texts[idx], emb)
            cache.save()
        else:
            logger.info("  All %d embeddings cached", len(enriched_texts))
        embeddings = [cache.get(t) for t in enriched_texts]
    else:
        logger.info("  Batch embedding %d texts...", len(enriched_texts))
        embeddings = embed_batch(enriched_texts)

    # Insert all into DB
    db = get_db()
    dia_map = {}

    try:
        for i, (turn, content, embedding, themes_list, summary) in enumerate(
            zip(conv_turns, contents, embeddings, themes_lists, summaries)
        ):
            themes_json = json.dumps(themes_list)
            mem_id = str(uuid.uuid4())

            _insert_memory(
                db, mem_id, content,
                summary=summary,
                category="episodic",
                themes_json=themes_json,
                priority=5,
                source="benchmark",
                status="active",
                embedding=embedding,
            )

            # Fix created_at to use the LoCoMo session date
            db.execute(
                "UPDATE memories SET created_at = ?, last_accessed = ? WHERE id = ?",
                (turn["created_at"], turn["created_at"], mem_id),
            )

            dia_map[(turn["conv_id"], turn["dia_id"])] = mem_id

        db.commit()
    finally:
        db.close()

    logger.info("  Ingested %d turns, %d mapped", len(conv_turns), len(dia_map))
    return dia_map


# ---------------------------------------------------------------------------
# LLM enrichment
# ---------------------------------------------------------------------------

ENRICH_PROMPT = """Extract searchable metadata from this conversation turn. Return JSON with:
- "topics": list of 3-8 key topics/concepts (nouns, entities, activities, implicit themes)
- "facts": list of 1-3 factual statements that could answer future questions
- "entities": list of named entities (people, places, books, organizations)

Be thorough — include implicit topics. If someone mentions "my son got in an accident" extract topics like ["son", "children", "family", "accident", "emergency"].

Turn: {text}

JSON:"""


def _batch_enrich(
    conv_turns: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    max_workers: int = 10,
) -> list[dict]:
    """Extract topics/facts/entities from turns using claude -p.

    Returns list of dicts with keys: topics, facts, entities.
    """
    import os
    import subprocess
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    empty = {"topics": [], "facts": [], "entities": []}
    claude_cmd = "claude.cmd" if sys.platform == "win32" else "claude"
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    def enrich_one(idx: int) -> tuple[int, dict]:
        turn = conv_turns[idx]
        text = f"[{turn['speaker']}] {turn['text']}"
        prompt = ENRICH_PROMPT.format(text=text)

        for attempt in range(3):
            try:
                result = subprocess.run(
                    [claude_cmd, "-p", "--model", model],
                    input=prompt,
                    capture_output=True,
                    timeout=60,
                    env=env,
                    encoding="utf-8",
                    errors="replace",
                )
                if result.returncode == 0:
                    output = result.stdout.strip()
                    # Strip markdown code fences if present
                    if "```" in output:
                        output = output.split("```")[1]
                        if output.startswith("json"):
                            output = output[4:]
                        output = output.strip()
                    return idx, json.loads(output)
            except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
                if attempt < 2:
                    time.sleep(1)
                continue

        logger.warning("  Enrichment failed for turn %d", idx)
        return idx, empty

    results = [empty] * len(conv_turns)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(enrich_one, i): i
                   for i in range(len(conv_turns))}
        for future in as_completed(futures):
            idx, parsed = future.result()
            results[idx] = parsed
            completed += 1
            if completed % 100 == 0:
                logger.info("  Enriched %d/%d turns...",
                            completed, len(conv_turns))

    return results
