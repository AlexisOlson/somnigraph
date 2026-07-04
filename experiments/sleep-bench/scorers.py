"""Scorer axis for the Stage A retrieval rig -- formula / v6 / l5b.

Independent of eval_retrieval.py's `apply_scoring_config`/`CONFIGS`: this
switches `memory.reranker`'s own `_cache` dict (the structure reranker.py
itself defines -- see `src/memory/reranker.py`) directly, rather than
importing the eval harness's config-patching machinery.

Every switch is followed by a hard, loud assertion via `reranker.
scorer_status()` (the exact anti-recurrence check added after the
2026-04-07 silent-fallback outage) -- this rig will not silently score a
"v6" run on the formula fallback.
"""

from pathlib import Path

import memory.constants as constants
import memory.reranker as reranker

SCORER_NAMES = ("formula", "v6", "l5b")

# Known real V6 model locations, .txt ONLY. A stale reranker_model.pkl must
# never shadow the deployed .txt -- that shadowing (LightGBM's .txt-native
# loader rejecting the pickle and silently falling back to formula) is
# exactly the 2026-04-07 outage shape and the eval_retrieval.py bug this
# proposal's §3.4 called out. A store-local copy (proposal §3.4: "copy the
# deployed reranker_model.txt into each fork's tuning_studies/") wins
# automatically via `constants.DATA_DIR`'s own default MODEL_PATH; these are
# the fallback for stores that don't carry their own copy.
V6_FALLBACK_LOCATIONS = [
    Path.home() / ".somnigraph" / "tuning_studies" / "reranker_model.txt",
    Path.home() / ".claude" / "data" / "tuning_studies" / "reranker_model.txt",
]

# L5b (LoCoMo Config K, 15-feature graph-augmented reranker) artifact.
# See docs/benchmarks.md "Feature selection" section for provenance.
L5B_MODEL_PATH = Path.home() / ".somnigraph" / "benchmark" / "locomo_reranker_model.pkl"


def _reset_cache() -> None:
    reranker._cache = {
        "model": None, "feature_names": None, "memory_meta": None,
        "meta_mem_count": -1, "loaded": False, "failed": False,
    }


def apply_scorer(name: str) -> str:
    """Configure the reranker axis for `name`. Returns `scorer_status()`
    AFTER applying and (for v6) verifying the load -- callers should log
    this string so a run's provenance is legible without re-deriving it.

    Raises RuntimeError if a scorer's own self-check fails (formula must
    disable the model; v6 must load a real .txt, not fall back).
    Raises NotImplementedError for l5b (see module note below) -- this is a
    loud, explicit stub, not a silent skip.
    """
    if name == "formula":
        _reset_cache()
        reranker._cache["failed"] = True   # force formula fallback
        reranker._cache["loaded"] = True
        status = reranker.scorer_status()
        if not status.startswith("formula fallback"):
            raise RuntimeError(
                f"formula scorer did not disable the reranker: {status!r}"
            )
        return status

    if name == "v6":
        _reset_cache()
        model_path = constants.MODEL_PATH
        if not model_path.exists():
            for candidate in V6_FALLBACK_LOCATIONS:
                if candidate.exists():
                    model_path = candidate
                    break
        if model_path.suffix != ".txt":
            raise RuntimeError(
                f"v6 scorer refuses a non-.txt MODEL_PATH ({model_path}) -- "
                "this is the exact 2026-04-07 outage shape (a stale .pkl "
                "shadowing the deployed .txt and silently degrading to "
                "formula). Deploy reranker_model.txt, don't point here."
            )
        constants.MODEL_PATH = model_path
        reranker.MODEL_PATH = model_path
        features_path = model_path.parent / "reranker_features.json"
        constants.MODEL_FEATURES_PATH = features_path
        reranker.MODEL_FEATURES_PATH = features_path

        status = reranker.scorer_status()
        if not status.startswith("reranker:"):
            raise RuntimeError(
                "V6 FAILED TO LOAD -- silently fell back to formula "
                f"(anti-recurrence assert tripped): {status!r}. "
                f"Checked MODEL_PATH={model_path}"
            )
        return status

    if name == "l5b":
        found = L5B_MODEL_PATH.exists()
        raise NotImplementedError(
            "l5b is a REGISTERED-BUT-STUBBED scorer, not wired. "
            f"Artifact {'FOUND' if found else 'NOT FOUND'} at {L5B_MODEL_PATH}. "
            "Its feature extraction (_locomo_rerank_inner in "
            "scripts/locomo_bench/eval_retrieval.py, ~700 lines) is tightly "
            "coupled to LoCoMo-bench-only state: a per-conversation entity/"
            "graph cache, synthetic-node resolution keyed to v6-extraction's "
            "build_graph.py, and eval_retrieval.py's own monkey-patch of "
            "memory.reranker.rerank. Porting that here would mean depending "
            "on exactly the eval-harness machinery this rig is designed to "
            "be independent of. Per docs/proposals/sleep-pipeline-"
            "optimization.md §3.4, L5b is explicitly 'context only' and is "
            "never used as a sleep-ablation scorer anyway (its graph "
            "features read the synthetic-extraction graph, not sleep's "
            "memory_edges) -- wiring it is not on this rig's critical path. "
            "Not blocking: formula and v6 are the two scorers Phase 1-3 "
            "actually need."
        )

    raise ValueError(f"unknown scorer: {name!r} (known: {SCORER_NAMES})")
