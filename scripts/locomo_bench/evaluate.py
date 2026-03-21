"""Retrieval + answer generation + judging."""

import json
import logging
import os
import re
import sqlite3
import string
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .config import CATEGORY_NAMES, BenchConfig
from .prompts import (
    ADVERSARIAL_PHRASES,
    format_judge_prompt,
    format_reader_input,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def recall_memories(
    question: str,
    config: BenchConfig,
) -> tuple[list[str], str]:
    """Call impl_recall and return (memory_ids, formatted_context).

    Returns the ranked memory IDs and a reader-friendly context string
    built from the top-k results.
    """
    from memory.db import get_db
    from memory.tools import impl_recall

    # Call impl_recall with internal=True to suppress event logging
    result = impl_recall(
        query=question,
        context=question,
        budget=config.recall_budget,
        limit=config.recall_limit,
        internal=True,
    )

    # Parse memory IDs from the formatted output
    memory_ids = _parse_recall_ids(result)

    # Build reader context from DB
    db = get_db()
    try:
        context = _build_reader_context(db, memory_ids, config.reader_top_k)
    finally:
        db.close()

    return memory_ids, context


def _parse_recall_ids(recall_output: str) -> list[str]:
    """Extract memory IDs from impl_recall output."""
    # The feedback line has all IDs
    for line in recall_output.split("\n"):
        if line.startswith("recall_feedback IDs:"):
            short_ids = line.split(":")[1].strip().split()
            return short_ids

    # Fallback: parse individual ID lines
    ids = []
    for line in recall_output.split("\n"):
        match = re.search(r"ID:\s+([a-f0-9]{8})", line)
        if match:
            ids.append(match.group(1))
    return ids


def _build_reader_context(
    db: sqlite3.Connection,
    memory_ids: list[str],
    top_k: int,
) -> str:
    """Build reader context string from memory IDs."""
    lines = []
    for short_id in memory_ids[:top_k]:
        row = db.execute(
            "SELECT content, created_at FROM memories WHERE id LIKE ? AND status = 'active'",
            (short_id + "%",),
        ).fetchone()
        if not row:
            continue
        date_str = row["created_at"][:10] if row["created_at"] else "unknown date"
        content = row["content"]
        # Parse speaker from content (format: "[Speaker] text")
        speaker_match = re.match(r"\[([^\]]+)\]\s*(.*)", content)
        if speaker_match:
            speaker = speaker_match.group(1)
            text = speaker_match.group(2)
        else:
            speaker = "?"
            text = content
        lines.append(f'({date_str}) {speaker} said, "{text}"')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer generation (reader)
# ---------------------------------------------------------------------------


def generate_answer(
    question: str,
    context: str,
    category: int,
    ground_truth: str,
    config: BenchConfig,
    seed: int = 42,
) -> str:
    """Call reader model to generate an answer."""
    system_prompt, user_prompt = format_reader_input(
        question, context, category, ground_truth,
        mode=config.prompt_mode, seed=seed,
    )

    response = _call_llm(
        system_prompt, user_prompt, config.model_reader,
    )

    # CORE mode: parse JSON from <output> tags
    if config.prompt_mode == "core":
        return _parse_core_answer(response)
    return response.strip()


def _parse_core_answer(response: str) -> str:
    """Parse CORE-format answer from <output>{"answer": "..."}</output>."""
    # Try <output> tags first
    match = re.search(r"<output>([\s\S]*?)</output>", response)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
            return parsed.get("answer", match.group(1).strip())
        except json.JSONDecodeError:
            return match.group(1).strip()

    # Try raw JSON
    try:
        parsed = json.loads(response.strip())
        return parsed.get("answer", response.strip())
    except json.JSONDecodeError:
        pass

    return response.strip()


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def judge_answer(
    question: str,
    gold: str,
    generated: str,
    category: int,
    config: BenchConfig,
) -> dict:
    """Judge a generated answer against ground truth.

    Returns {"label": "CORRECT"|"WRONG", "reasoning": str, "method": str}.
    """
    # Cat 5: mechanical abstention check
    if category == 5:
        result = _score_adversarial(generated)
        result["method"] = "mechanical"
        return result

    # Check if model abstained on an answerable question
    if any(p in generated.lower() for p in ADVERSARIAL_PHRASES):
        return {
            "label": "WRONG",
            "reasoning": "mechanical: abstained on answerable question",
            "method": "mechanical",
        }

    prompt = format_judge_prompt(question, gold, generated, mode=config.prompt_mode)

    try:
        if config.prompt_mode == "core":
            # CORE sends judge as single user message (no system)
            output = _call_llm(None, prompt, config.model_judge)
        else:
            output = _call_llm(prompt, "", config.model_judge)

        result = _parse_judge_output(output)
        result["method"] = "llm"

        # Compute match ratio (CORE does this for all questions)
        result["match_ratio"] = _match_ratio(gold, generated)
        return result

    except Exception as e:
        logger.warning("Judge LLM failed, using heuristic: %s", e)
        ratio = _match_ratio(gold, generated)
        return {
            "label": "CORRECT" if ratio > 0.3 else "WRONG",
            "reasoning": f"heuristic_fallback: match_ratio={ratio:.2f}",
            "method": "heuristic_fallback",
            "match_ratio": ratio,
        }


def _score_adversarial(generated: str) -> dict:
    """Cat 5: mechanical abstention check."""
    pred_lower = generated.lower()
    abstained = any(phrase in pred_lower for phrase in ADVERSARIAL_PHRASES)
    return {
        "label": "CORRECT" if abstained else "WRONG",
        "reasoning": ("mechanical: abstention detected"
                      if abstained else "mechanical: specific answer given"),
    }


def _parse_judge_output(output: str) -> dict:
    """Parse judge model output into label + reasoning."""
    try:
        text = output.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)
        label = parsed.get("label", "").upper()
        reasoning = parsed.get("reasoning", text)
        if label not in ("CORRECT", "WRONG"):
            if "correct" in output.lower() and "wrong" not in output.lower():
                label = "CORRECT"
            elif "wrong" in output.lower():
                label = "WRONG"
            else:
                label = "UNKNOWN"
        return {"label": label, "reasoning": reasoning}
    except json.JSONDecodeError:
        # CORE's fallback: look for CORRECT/WRONG in text
        if "CORRECT" in output and "WRONG" not in output:
            return {"label": "CORRECT", "reasoning": output[:200]}
        elif "WRONG" in output and "CORRECT" not in output:
            return {"label": "WRONG", "reasoning": output[:200]}
        # If both appear, try to find the final one
        elif "WRONG" in output:
            return {"label": "WRONG", "reasoning": output[:200]}
        else:
            return {"label": "UNKNOWN", "reasoning": output[:200]}


def _match_ratio(gold: str, generated: str) -> float:
    """CORE's word overlap metric: fraction of 3+ letter gold words found in generated."""
    gen_lower = generated.lower()
    gold_lower = str(gold).lower()
    gold_words = [w for w in gold_lower.split() if len(w) > 2]
    if not gold_words:
        return 0.0
    matching = sum(1 for w in gold_words if w in gen_lower)
    return matching / len(gold_words)


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 matching LoCoMo paper."""
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize(text: str) -> str:
    """Lowercase, remove punctuation/articles, collapse whitespace."""
    text = str(text).lower()
    text = text.replace(",", "")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# LLM calling
# ---------------------------------------------------------------------------

_openai_client = None


def _get_openai_client():
    """Lazy-init OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


def _call_llm(
    system_prompt: str | None,
    user_prompt: str,
    model: str,
) -> str:
    """Call an LLM. Routes to OpenAI API for gpt-* models, claude -p for claude-* models."""
    if model.startswith("gpt-"):
        return _call_openai(system_prompt, user_prompt, model)
    return _call_claude(system_prompt, user_prompt, model)


def _call_openai(
    system_prompt: str | None,
    user_prompt: str,
    model: str,
) -> str:
    """Call OpenAI API directly."""
    client = _get_openai_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"OpenAI API failed after 3 attempts: {e}") from e


def _call_claude(
    system_prompt: str | None,
    user_prompt: str,
    model: str,
) -> str:
    """Call Claude via claude -p subprocess."""
    prompt = ""
    if system_prompt:
        prompt += system_prompt + "\n\n"
    prompt += user_prompt

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    claude_cmd = "claude.cmd" if sys.platform == "win32" else "claude"

    for attempt in range(3):
        try:
            result = subprocess.run(
                [claude_cmd, "-p", "--model", model],
                input=prompt,
                capture_output=True,
                timeout=120,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0:
                return result.stdout.strip()
            err = result.stderr.strip()
        except subprocess.TimeoutExpired:
            err = "timeout"

        if attempt < 2:
            time.sleep(2 ** attempt)

    raise RuntimeError(f"claude -p failed after 3 attempts: {err}")
