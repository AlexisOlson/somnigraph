"""Benchmark configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchConfig:
    """Configuration for a LoCoMo benchmark run."""

    # Models
    model_reader: str = "gpt-4.1"  # CORE uses gpt-4.1-2025-04-14
    model_judge: str = "gpt-4.1"   # CORE uses gpt-4.1-2025-04-14

    # Prompt mode: "core" for faithful replication, "somnigraph" for our optimized prompts
    prompt_mode: str = "core"

    # Retrieval
    recall_limit: int = 20
    recall_budget: int = 5000
    reader_top_k: int = 10  # how many retrieved memories to show the reader

    # Categories
    skip_adversarial: bool = False  # CORE skips cat 5; we report both

    # Ablations
    run_sleep: bool = False  # NREM between ingest and eval
    use_feedback_loop: bool = False  # two-pass recall with feedback
    use_reranker: bool = True  # False = formula fallback

    # Paths
    locomo_data: Path = Path.home() / ".claude" / "repos" / "locomo" / "data" / "locomo10.json"
    base_dir: Path = Path.home() / ".somnigraph" / "benchmark"
    embed_cache: Path = Path.home() / ".claude" / "data" / "bench_locomo_embeddings.pkl"

    # Scope
    conversations: list[int] = field(default_factory=list)  # empty = all 10

    # Resume
    resume: bool = False

    # Skip judging (generate answers only, batch judge later)
    no_judge: bool = False


CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}
