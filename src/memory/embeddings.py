"""Embedding backend — OpenAI or fastembed, selected by SOMNIGRAPH_EMBEDDING_BACKEND."""

import logging
import os
import time

from memory.constants import EMBEDDING_BACKEND, EMBEDDING_MODEL

logger = logging.getLogger("claude-memory")

__all__ = ["count_tokens", "embed_text", "embed_batch", "build_enriched_text"]

# ---------------------------------------------------------------------------
# Tokenizer (shared across backends)
# ---------------------------------------------------------------------------

_tokenizer = None


def get_tokenizer():
    """Lazy-load tiktoken tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        import tiktoken
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text))


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

_openai_client = None


def _get_openai_client():
    """Lazy-load OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            from memory.constants import DATA_DIR
            key_file = DATA_DIR / "openai_api_key"
            if key_file.exists():
                api_key = key_file.read_text().strip()
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _retry(func, max_retries=3, base_delay=1.0):
    """Retry with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Embedding attempt {attempt + 1} failed ({e}), retrying in {delay}s")
            time.sleep(delay)


def _openai_embed_text(text: str) -> list[float]:
    client = _get_openai_client()
    def _call():
        response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response.data[0].embedding
    return _retry(_call)


def _openai_embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = _get_openai_client()
    def _call():
        response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]
    return _retry(_call)


# ---------------------------------------------------------------------------
# Fastembed backend
# ---------------------------------------------------------------------------

_fastembed_client = None


def _get_fastembed_client():
    """Lazy-load fastembed client."""
    global _fastembed_client
    if _fastembed_client is None:
        from fastembed import TextEmbedding
        cache_dir = os.path.join(os.path.expanduser("~"), ".claude", "data", "fastembed_cache")
        _fastembed_client = TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=cache_dir)
    return _fastembed_client


def _fastembed_embed_text(text: str) -> list[float]:
    client = _get_fastembed_client()
    results = list(client.embed([text]))
    return results[0].tolist()


def _fastembed_embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = _get_fastembed_client()
    return [r.tolist() for r in client.embed(texts)]


# ---------------------------------------------------------------------------
# Public API — dispatches to configured backend
# ---------------------------------------------------------------------------

if EMBEDDING_BACKEND == "fastembed":
    embed_text = _fastembed_embed_text
    embed_batch = _fastembed_embed_batch
else:
    embed_text = _openai_embed_text
    embed_batch = _openai_embed_batch


def build_enriched_text(content: str, category: str, themes_list: list, summary: str) -> str:
    """Concatenate content with metadata for richer embedding."""
    parts = [content]
    parts.append(f"Category: {category}")
    if themes_list:
        parts.append(f"Themes: {', '.join(themes_list)}")
    if summary:
        parts.append(f"Summary: {summary}")
    return "\n".join(parts)
