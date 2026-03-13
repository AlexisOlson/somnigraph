"""Decay model — time-based priority reduction with per-category rates."""

import math
from datetime import datetime, timezone

from memory.constants import CATEGORY_DECAY_RATES, DEFAULT_DECAY_RATE, DECAY_FLOOR


def effective_priority(base: int, last_accessed: str, decay_rate: float = None,
                       category: str = None, flags: list = None) -> float:
    """Compute ranking priority with per-memory exponential decay and floor."""
    # Retention immunity — pinned memories don't decay
    if flags and ("pinned" in flags or "keep" in flags):
        return float(base)

    if decay_rate is None:
        decay_rate = CATEGORY_DECAY_RATES.get(category, DEFAULT_DECAY_RATE)
    if decay_rate == 0:
        return float(base)

    now = datetime.now(timezone.utc)
    last = datetime.fromisoformat(last_accessed)
    # Ensure timezone-aware comparison
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    days_since = (now - last).total_seconds() / 86400

    decayed = base * math.exp(-decay_rate * days_since)
    floor = base * DECAY_FLOOR

    return max(decayed, floor)
