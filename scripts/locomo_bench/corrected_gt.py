"""Build and apply corrected LoCoMo ground truth from the locomo-audit.

The locomo-audit repo (github.com/dial481/locomo-audit) identified 156 issues
in LoCoMo-10's ground truth, of which 99 are score-corrupting (golden answer
is wrong). This module:

1. Builds a corrected copy of locomo10.json using the audit's errors.json
2. Provides a lookup for re-judging existing results against corrected GT

Only score-corrupting errors are patched (HALLUCINATION, TEMPORAL_ERROR,
ATTRIBUTION_ERROR, AMBIGUOUS, INCOMPLETE). WRONG_CITATION errors don't
affect answer scoring and are left as-is.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Error types that change the golden answer (score-corrupting)
SCORE_CORRUPTING = {
    "HALLUCINATION", "TEMPORAL_ERROR", "ATTRIBUTION_ERROR",
    "AMBIGUOUS", "INCOMPLETE",
}

AUDIT_ERRORS_PATH = Path(__file__).resolve().parent / "locomo_audit_errors.json"


def load_corrections(errors_path: Path = AUDIT_ERRORS_PATH) -> dict[tuple[int, int], dict]:
    """Load score-corrupting corrections as {(conv_idx, qa_idx): error_entry}.

    Question ID format in audit: locomo_{conv}_qa{idx} where idx matches
    the 0-based index into data[conv]['qa'].
    """
    with open(errors_path) as f:
        errors = json.load(f)

    corrections = {}
    for entry in errors:
        if entry["error_type"] not in SCORE_CORRUPTING:
            continue
        if not entry.get("correct_answer"):
            continue

        qid = entry["question_id"]
        # Parse locomo_{conv}_qa{idx}
        parts = qid.split("_")
        conv_idx = int(parts[1])
        qa_idx = int(parts[2][2:])  # strip "qa" prefix

        corrections[(conv_idx, qa_idx)] = entry

    logger.info("Loaded %d score-corrupting corrections", len(corrections))
    return corrections


def build_corrected_dataset(
    original_path: Path,
    output_path: Path,
    errors_path: Path = AUDIT_ERRORS_PATH,
) -> dict:
    """Create a corrected locomo10.json with patched golden answers.

    Returns stats about what was changed.
    """
    with open(original_path) as f:
        data = json.load(f)

    corrections = load_corrections(errors_path)

    stats = {"patched": 0, "by_type": {}, "by_conv": {}}
    for (conv_idx, qa_idx), entry in corrections.items():
        if conv_idx >= len(data):
            logger.warning("Conv %d out of range, skipping %s",
                           conv_idx, entry["question_id"])
            continue
        qa_list = data[conv_idx]["qa"]
        if qa_idx >= len(qa_list):
            logger.warning("QA %d out of range in conv %d, skipping %s",
                           qa_idx, conv_idx, entry["question_id"])
            continue

        old_answer = qa_list[qa_idx]["answer"]
        new_answer = entry["correct_answer"]

        if old_answer != new_answer:
            qa_list[qa_idx]["answer"] = new_answer
            stats["patched"] += 1

            etype = entry["error_type"]
            stats["by_type"][etype] = stats["by_type"].get(etype, 0) + 1
            stats["by_conv"][conv_idx] = stats["by_conv"].get(conv_idx, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Wrote corrected dataset: %d answers patched -> %s",
                stats["patched"], output_path)
    return stats


def get_corrected_answer(
    corrections: dict[tuple[int, int], dict],
    conv_idx: int,
    qa_idx: int,
) -> str | None:
    """Look up a corrected answer for a specific question. Returns None if no correction."""
    entry = corrections.get((conv_idx, qa_idx))
    if entry:
        return entry["correct_answer"]
    return None


def get_error_type(
    corrections: dict[tuple[int, int], dict],
    conv_idx: int,
    qa_idx: int,
) -> str | None:
    """Look up the error type for a specific question. Returns None if no error."""
    entry = corrections.get((conv_idx, qa_idx))
    if entry:
        return entry["error_type"]
    return None
