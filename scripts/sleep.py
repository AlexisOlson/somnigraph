# /// script
# requires-python = ">=3.11"
# dependencies = ["sqlite-vec>=0.1.6", "openai>=2.0.0", "tiktoken>=0.7.0", "mcp[cli]>=1.2.0"]
# ///
"""
Full sleep cycle: deep NREM then REM.

Usage: uv run scripts/sleep.py
"""

import ast
import collections
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from memory.constants import DATA_DIR

SCRIPTS = Path(__file__).parent

# Ensure claude CLI is on PATH (npm global bin may not be in PowerShell PATH)
_NPM_BIN = Path.home() / "AppData" / "Roaming" / "npm"
if _NPM_BIN.exists() and str(_NPM_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_NPM_BIN) + os.pathsep + os.environ.get("PATH", "")


def run_phase(name, cmd):
    """Run a phase, streaming output. Returns elapsed seconds."""
    print(f"{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(flush=True)

    start = time.time()
    result = subprocess.run(cmd, cwd=str(SCRIPTS), env=os.environ)
    elapsed = int(time.time() - start)

    print(f"\n  {name} complete ({elapsed}s)\n", flush=True)

    if result.returncode != 0:
        print(f"  ERROR: {name} exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    return elapsed


from memory import DB_PATH
_SUBPROCESS_CWD = DATA_DIR / "subprocess_workspace"
EMPTY_MCP_CONFIG = json.dumps({"mcpServers": {}})
THEME_REVIEW_LOG = DATA_DIR / "theme_review.json"


def run_theme_normalization():
    """Quick pre-sleep pass: normalize variant themes across all memories."""
    print(f"{'─' * 60}")
    print(f"  Pre-sleep: Theme normalization")
    print(f"{'─' * 60}")

    from memory import normalize_all_themes
    updated = normalize_all_themes()
    if updated:
        print(f"  Normalized themes on {updated} memories")
    else:
        print(f"  All themes already canonical")
    print()
    return updated


def detect_theme_variants():
    """Detect potential theme variants. Prints candidates for human review.

    High-confidence: casing, underscore/hyphen, singular/plural.
    Medium-confidence: prefix where short form is established (>=5 uses)
    and long form is rare (<=2 uses) — suggests the compound was unnecessary.
    """
    print(f"{'─' * 60}")
    print(f"  Post-sleep: Theme variant detection")
    print(f"{'─' * 60}")

    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT themes FROM memories WHERE status IN ('active', 'pending')"
    ).fetchall()
    db.close()

    counter = collections.Counter()
    for r in rows:
        themes = json.loads(r["themes"]) if r["themes"] else []
        for t in themes:
            counter[t] += 1

    themes = sorted(counter.keys())
    candidates = []

    for i, a in enumerate(themes):
        for b in themes[i + 1:]:
            reason = _variant_reason(a, b, counter[a], counter[b])
            if reason:
                candidates.append((a, counter[a], b, counter[b], reason))

    if candidates:
        # Group by variant type, then sort each group by combined frequency
        by_type = {}
        for c in candidates:
            by_type.setdefault(c[4], []).append(c)

        # Order types: mechanical first (casing, separator, plural, verb), then prefix
        type_order = ["casing", "separator", "separator (no-sep vs hyphenated)",
                       "singular/plural", "verb form (-ing)", "prefix (rare compound)"]
        ordered_types = [t for t in type_order if t in by_type]
        ordered_types += [t for t in by_type if t not in ordered_types]

        print(f"  Found {len(candidates)} potential variant pair(s):\n")
        for vtype in ordered_types:
            pairs = sorted(by_type[vtype], key=lambda x: -(x[1] + x[3]))
            print(f"    [{vtype}] ({len(pairs)} pairs)")
            for a, ca, b, cb, reason in pairs:
                print(f"      \"{a}\" ({ca}) <-> \"{b}\" ({cb})")
            print()
    else:
        print("  No potential variants detected\n")

    return candidates


def _normalize_separators(s: str) -> str:
    """Collapse all separator variants (spaces, dots, underscores) to hyphens."""
    return s.lower().replace(" ", "-").replace("_", "-").replace(".", "-")


def _variant_reason(a: str, b: str, count_a: int, count_b: int) -> str | None:
    """Return a reason string if a and b look like theme variants, else None.

    Conservative: only flags high-confidence matches to keep output actionable.
    """
    al, bl = a.lower(), b.lower()

    # Exact match after lowercasing (casing variant) — always flag
    if al == bl and a != b:
        return "casing"

    # Separator normalization: spaces, underscores, dots, hyphens all equivalent
    an, bn = _normalize_separators(a), _normalize_separators(b)
    if an == bn and a != b:
        return "separator"

    # No-separator vs hyphenated (e.g., "virtualtrebuchet" vs "virtual-trebuchet")
    if an != bn and an.replace("-", "") == bn.replace("-", ""):
        return "separator (no-sep vs hyphenated)"

    # Singular/plural (trailing 's' difference) — always flag
    if al != bl and len(al) >= 4 and len(bl) >= 4:
        if al + "s" == bl or bl + "s" == al:
            return "singular/plural"

    # Verb form variants: -ing suffix (e.g., "overclaim" vs "overclaiming")
    if al != bl and len(al) >= 5 and len(bl) >= 5:
        if al + "ing" == bl or bl + "ing" == al:
            return "verb form (-ing)"

    # Prefix match — only flag when the shorter form is established (>=5 uses)
    # and the longer form is rare (<=2 uses), suggesting unnecessary compound
    if al != bl:
        if len(al) <= len(bl):
            shorter, longer, cs, cl = al, bl, count_a, count_b
        else:
            shorter, longer, cs, cl = bl, al, count_b, count_a
        if longer.startswith(shorter + "-") and len(shorter) >= 4:
            if cs >= 5 and cl <= 2:
                return "prefix (rare compound)"

    return None


def _load_theme_variants_from_source() -> dict | None:
    """Extract THEME_VARIANTS dict from themes.py without importing it."""
    server_path = Path(__file__).resolve().parent.parent / "src" / "memory" / "themes.py"
    if not server_path.exists():
        return None
    try:
        source = server_path.read_text(encoding="utf-8")
        # Find the dict between "THEME_VARIANTS = {" and the closing "}"
        start = source.find("THEME_VARIANTS = {")
        if start < 0:
            return None
        # Find matching closing brace
        brace_start = source.index("{", start)
        depth = 0
        for i in range(brace_start, len(source)):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    dict_str = source[brace_start:i + 1]
                    return ast.literal_eval(dict_str)
        return None
    except Exception:
        return None


def _ensure_subprocess_workspace():
    """Create isolated workspace with git boundary to block CLAUDE.md loading."""
    git_dir = _SUBPROCESS_CWD / ".git"
    head_file = git_dir / "HEAD"
    if not head_file.exists():
        git_dir.mkdir(parents=True, exist_ok=True)
        head_file.write_text("ref: refs/heads/main\n")


def _call_claude_json(prompt: str, model: str = "sonnet", timeout: int = 300, label: str = "LLM") -> dict | None:
    """Call claude -p and parse JSON response. Returns parsed dict or None."""
    _ensure_subprocess_workspace()
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)

    kwargs = {}
    if sys.platform == "win32":
        CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = CREATE_NO_WINDOW

    claude_bin = shutil.which("claude") or "claude"
    try:
        result = subprocess.run(
            [
                claude_bin, "-p",
                "--model", model,
                "--no-session-persistence",
                "--strict-mcp-config",
                "--mcp-config", EMPTY_MCP_CONFIG,
                "--system-prompt", (
                    "You are a JSON-only output tool. Respond with the requested "
                    "JSON. No markdown fences, no commentary, no tool calls."
                ),
                "--tools", "",
                "--disable-slash-commands",
                "--setting-sources", "local",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            cwd=str(_SUBPROCESS_CWD),
            env=env,
            **kwargs,
        )
    except subprocess.TimeoutExpired:
        print(f"  WARNING: {label} timed out ({timeout}s)", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"  ERROR: 'claude' CLI not found on PATH", file=sys.stderr)
        return None

    if result.returncode != 0:
        stderr_preview = (result.stderr or "")[:200]
        print(f"  WARNING: {label} returned {result.returncode}: {stderr_preview}", file=sys.stderr)
        return None

    raw = result.stdout.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        print(f"  WARNING: {label} response not valid JSON: {raw[:200]}", file=sys.stderr)
        return None


def review_themes_with_llm(detected_variants: list | None = None):
    """Run mechanical and nuanced theme reviews in parallel.

    Mechanical review: uses detected variant pairs (fast, Haiku).
    Nuanced review: uses full theme list (slower, Sonnet).
    Results merged into one suggestion set.
    """
    print(f"{'─' * 60}")
    print(f"  Post-sleep: LLM theme review")
    print(f"{'─' * 60}")

    # Gather theme frequencies
    counter = _get_theme_counter()

    # Build variant block grouped by type for mechanical review
    variant_block = ""
    if detected_variants:
        by_type = {}
        for c in detected_variants:
            by_type.setdefault(c[4], []).append(c)
        type_order = ["casing", "separator", "separator (no-sep vs hyphenated)",
                       "singular/plural", "verb form (-ing)"]
        lines = []
        for vtype in type_order:
            if vtype not in by_type:
                continue
            lines.append(f"\n  [{vtype}]")
            for a, ca, b, cb, _ in sorted(by_type[vtype], key=lambda x: -(x[1] + x[3])):
                lines.append(f"    \"{a}\" ({ca}) <-> \"{b}\" ({cb})")
        variant_block = "\n".join(lines)

    # Build theme list for nuanced review — only themes with 3+ uses (skip mechanical territory)
    # Also exclude themes already flagged as mechanical variants
    mechanical_themes = set()
    if detected_variants:
        for a, ca, b, cb, reason in detected_variants:
            if reason != "prefix (rare compound)":
                mechanical_themes.add(a)
                mechanical_themes.add(b)
    theme_lines = [f"  {count:3d}  {theme}" for theme, count in counter.most_common()
                   if count >= 3 and theme not in mechanical_themes]
    theme_block = "\n".join(theme_lines)

    existing_mappings = _load_theme_variants_from_source()
    if existing_mappings:
        mappings_block = "\n".join(f"  {k} -> {v}" for k, v in sorted(existing_mappings.items()))
    else:
        mappings_block = "  (none loaded)"

    # --- Mechanical prompt (fast, focused on detected variants) ---
    mechanical_prompt = f"""You are normalizing theme tags in a personal memory system. Below are detected variant pairs grouped by type. For each pair, decide the canonical form and suggest a merge.

## Detected variant pairs:
{variant_block}

## Existing mappings (already handled, skip these):
{mappings_block}

## Rules:
- For casing: prefer lowercase
- For separators: prefer hyphenated (e.g., "eval-retrieval" not "eval_retrieval")
- For singular/plural: prefer singular
- For verb forms: prefer the more common form (higher count)
- Skip any pair where both forms are genuinely distinct concepts
- Up to 50 suggestions

## Output format:
Return ONLY a JSON object:
{{
  "merge": [{{"from": "variant", "to": "canonical", "reason": "type"}}],
  "split": [],
  "drop": []
}}""" if variant_block else None

    # --- Nuanced prompt (deeper analysis of full theme list) ---
    nuanced_prompt = f"""You are reviewing theme tags in a personal memory system for deeper quality issues. Mechanical variants (casing, plurals, separators) are handled separately — focus on semantic and structural problems.

## Current themes (count, tag):
{theme_block}

## Existing mappings (already handled):
{mappings_block}

## What to look for (up to 10 suggestions):

1. **Semantic duplicates**: Different words for the same concept. Only flag if clearly the same.
2. **Overly specific singletons**: Tags used once that are just a longer version of an established tag, where the memory probably already has the base tag.
3. **Tags that should be split**: A compound tag better served as two separate established tags (e.g., "skyrim-modding-gotchas" -> ["skyrim-modding", "gotcha"]).
4. **Prefer split over merge**: When a compound tag's components are both established tags (5+ uses each), suggest a split.

## What NOT to flag:
- Casing, separator, plural, or verb-form variants (handled separately)
- Tags that are genuinely distinct even if one is a prefix
- Tags already in the existing mappings

## Output format:
Return ONLY a JSON object:
{{
  "merge": [{{"from": "variant-tag", "to": "canonical-tag", "reason": "brief explanation"}}],
  "split": [{{"tag": "compound-tag", "into": ["tag1", "tag2"], "reason": "brief explanation"}}],
  "drop": [{{"tag": "useless-tag", "reason": "brief explanation"}}]
}}"""

    # Launch both reviews in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed

    futures = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        if mechanical_prompt:
            futures["mechanical"] = pool.submit(
                _call_claude_json, mechanical_prompt, "haiku", 120, "mechanical review"
            )
        futures["nuanced"] = pool.submit(
            _call_claude_json, nuanced_prompt, "sonnet", 300, "nuanced review"
        )

        results = {}
        for name, future in futures.items():
            results[name] = future.result()

    # Merge results
    merged = {"merge": [], "split": [], "drop": []}
    for name in ["mechanical", "nuanced"]:
        r = results.get(name)
        if r:
            merged["merge"].extend(r.get("merge", []))
            merged["split"].extend(r.get("split", []))
            merged["drop"].extend(r.get("drop", []))
            print(f"  {name.capitalize()}: {sum(len(r.get(k, [])) for k in ('merge', 'split', 'drop'))} suggestions")
        else:
            print(f"  {name.capitalize()}: failed")

    # Deduplicate merges (both might suggest the same thing)
    seen = set()
    unique_merges = []
    for m in merged["merge"]:
        key = (m.get("from", ""), m.get("to", ""))
        if key not in seen:
            seen.add(key)
            unique_merges.append(m)
    merged["merge"] = unique_merges

    suggestions = merged
    merges = suggestions.get("merge", [])
    splits = suggestions.get("split", [])
    drops = suggestions.get("drop", [])
    total = len(merges) + len(splits) + len(drops)

    if total == 0:
        print("  LLM found no issues")
    else:
        print(f"  LLM suggestions ({total} total):\n")
        if merges:
            print("  Merge:")
            for m in merges:
                print(f"    \"{m['from']}\" -> \"{m['to']}\" ({m.get('reason', '')})")
        if splits:
            print("  Split:")
            for s in splits:
                print(f"    \"{s['tag']}\" -> {s['into']} ({s.get('reason', '')})")
        if drops:
            print("  Drop:")
            for d in drops:
                print(f"    \"{d['tag']}\" ({d.get('reason', '')})")

    # Save to file for next session to review
    review_data = {
        "timestamp": datetime.now().isoformat(),
        "suggestions": suggestions,
        "theme_count": len(counter),
        "memory_count": len(rows),
    }
    THEME_REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)
    THEME_REVIEW_LOG.write_text(json.dumps(review_data, indent=2), encoding="utf-8")
    print(f"\n  Saved to {THEME_REVIEW_LOG}")
    print()

    return suggestions


def _get_theme_counter() -> collections.Counter:
    """Get theme frequency counter from the database."""
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT themes FROM memories WHERE status IN ('active', 'pending')"
    ).fetchall()
    db.close()

    counter = collections.Counter()
    for r in rows:
        themes = json.loads(r["themes"]) if r["themes"] else []
        for t in themes:
            counter[t] += 1
    return counter


def apply_theme_suggestions(suggestions: dict, dry_run: bool = False):
    """Auto-apply high-confidence theme suggestions, queue the rest.

    Thresholds:
    - Merge auto-apply: heuristic-detected variant (casing/separator/plural/verb-form),
      OR target has 15x+ source count AND source <= 2,
      OR LLM-confirmed merge where source <= 5 uses
    - Split auto-apply: ALL resulting tags already exist with 5+ uses
    - Merge → split conversion: source is compound of established tags (5+ each)
    - Drops: never auto-apply
    """
    print(f"{'─' * 60}")
    print(f"  Post-sleep: Auto-applying theme suggestions")
    print(f"{'─' * 60}")

    from memory import (
        add_theme_mapping,
        normalize_all_themes,
        split_theme_on_memories,
    )

    counter = _get_theme_counter()
    merges = suggestions.get("merge", [])
    splits = suggestions.get("split", [])
    drops = suggestions.get("drop", [])

    applied = []
    queued_merges = []
    queued_splits = []
    queued_drops = list(drops)  # drops always queued

    # --- Process merges ---
    for m in merges:
        src, tgt = m["from"], m["to"]
        src_count = counter.get(src, 0)
        tgt_count = counter.get(tgt, 0)
        reason = m.get("reason", "")

        # Check if this should be converted to a split instead
        # Source contains hyphens and components are each established themes
        parts = src.split("-")
        if len(parts) >= 2 and all(counter.get(p, 0) >= 5 for p in parts):
            # All parts are established themes — convert merge to split
            if dry_run:
                applied.append(f'  SPLIT (converted): "{src}" -> {parts}')
            else:
                updated = split_theme_on_memories(src, parts)
                applied.append(
                    f'  SPLIT (converted): "{src}" -> {parts} ({updated} memories)'
                )
            continue

        # Check if heuristic-detected (mechanical variant)
        is_heuristic = _variant_reason(src, tgt, src_count, tgt_count) is not None

        # Check ratio threshold: target 15x+ source AND source <= 2
        ratio_ok = tgt_count >= 15 * max(src_count, 1) and src_count <= 2

        # LLM-confirmed low-frequency: source <=5 uses, trust the LLM's judgment
        # The LLM review IS the quality gate for these — no need for a second filter
        llm_low_freq = src_count <= 5

        if is_heuristic or ratio_ok or llm_low_freq:
            ratio_str = f"ratio {tgt_count}:{max(src_count, 1)}"
            label = "heuristic" if is_heuristic else ratio_str if ratio_ok else f"llm-confirmed, {src_count} uses"
            if dry_run:
                applied.append(f'  MERGE: "{src}" -> "{tgt}" ({label})')
            else:
                add_theme_mapping(src, tgt)
                applied.append(f'  MERGE: "{src}" -> "{tgt}" ({label})')
        else:
            queued_merges.append(m)

    # --- Process splits ---
    for s in splits:
        tag = s["tag"]
        into = s["into"]

        # Auto-apply if ALL target tags have 5+ uses
        if all(counter.get(t, 0) >= 5 for t in into):
            if dry_run:
                applied.append(f'  SPLIT: "{tag}" -> {into}')
            else:
                updated = split_theme_on_memories(tag, into)
                applied.append(f'  SPLIT: "{tag}" -> {into} ({updated} memories)')
        else:
            queued_splits.append(s)

    # --- Run normalize_all_themes if we added any merge mappings ---
    if not dry_run and any(line.startswith("  MERGE:") for line in applied):
        norm_count = normalize_all_themes()
        if norm_count:
            print(f"  (normalized {norm_count} memories after merges)")

    # --- Print summary ---
    if applied:
        print(f"\n  Applied {len(applied)} suggestion(s):\n")
        for line in applied:
            print(f"  {line}")
    else:
        print("\n  No suggestions met auto-apply thresholds")

    total_queued = len(queued_merges) + len(queued_splits) + len(queued_drops)
    if total_queued:
        print(f"\n  Queued {total_queued} for review:")
        for m in queued_merges:
            print(f'    MERGE: "{m["from"]}" -> "{m["to"]}" ({m.get("reason", "")})')
        for s in queued_splits:
            print(f'    SPLIT: "{s["tag"]}" -> {s["into"]} ({s.get("reason", "")})')
        for d in queued_drops:
            print(f'    DROP: "{d["tag"]}" ({d.get("reason", "")})')

    # --- Rewrite theme_review.json with only queued items ---
    if not dry_run:
        remaining = {
            "merge": queued_merges,
            "split": queued_splits,
            "drop": queued_drops,
        }
        if total_queued:
            review_data = {
                "timestamp": datetime.now().isoformat(),
                "suggestions": remaining,
                "note": "Auto-applied suggestions removed; these need manual review",
            }
            THEME_REVIEW_LOG.write_text(json.dumps(review_data, indent=2), encoding="utf-8")
        else:
            # All applied — remove the file
            if THEME_REVIEW_LOG.exists():
                THEME_REVIEW_LOG.unlink()
                print(f"\n  Cleared {THEME_REVIEW_LOG} (all suggestions applied)")

    print()


def repair_retrieval_failures():
    """Diagnose retrieval failures via LLM and apply theme corrections.

    Consumes recall_miss events with miss_type='retrieval_failure',
    sends them to Sonnet for diagnosis, and applies theme repairs.
    Returns count of repairs applied.
    """
    print(f"{'─' * 60}")
    print(f"  Retrieval failure repair")
    print(f"{'─' * 60}")

    from memory.fts import update_fts
    from memory.themes import normalize_themes
    from memory.events import _log_event
    from memory.db import _resolve_id
    from memory.constants import MAX_THEMES

    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row

    # 1. Gather unprocessed retrieval failures
    all_misses = db.execute(
        "SELECT id, query, context, created_at FROM memory_events "
        "WHERE event_type = 'recall_miss' ORDER BY created_at ASC"
    ).fetchall()

    processed_rows = db.execute(
        "SELECT context FROM memory_events WHERE event_type = 'recall_miss_repaired'"
    ).fetchall()
    processed_ids = set()
    for r in processed_rows:
        ctx = json.loads(r["context"]) if r["context"] else {}
        if "miss_event_id" in ctx:
            processed_ids.add(ctx["miss_event_id"])

    unprocessed = []
    for m in all_misses:
        if m["id"] in processed_ids:
            continue
        ctx = json.loads(m["context"]) if m["context"] else {}
        miss_type = ctx.get("miss_type", "")

        # Skip coverage gaps — nothing to repair
        if miss_type == "coverage_gap":
            _log_event(db, "_recall", "recall_miss_repaired",
                       query=m["query"],
                       context={"miss_event_id": m["id"], "skipped": "coverage_gap"})
            db.commit()
            continue

        if miss_type == "retrieval_failure":
            unprocessed.append(m)

    if not unprocessed:
        print("  No retrieval failures to repair\n")
        db.close()
        return 0

    # Cap at 10 cases per cycle (oldest first)
    batch = unprocessed[:10]
    print(f"  Processing {len(batch)} retrieval failure(s) (of {len(unprocessed)} pending)")

    # 2. Find returned memories for each miss and build cases
    cases = []
    for miss in batch:
        retrieved = db.execute(
            "SELECT DISTINCT memory_id, similarity_score FROM memory_events "
            "WHERE event_type = 'retrieved' AND query = ? "
            "AND created_at < ? AND created_at > datetime(?, '-1 hour') "
            "ORDER BY similarity_score DESC",
            (miss["query"], miss["created_at"], miss["created_at"])
        ).fetchall()

        if not retrieved:
            # No retrieved events found — mark processed and skip
            _log_event(db, "_recall", "recall_miss_repaired",
                       query=miss["query"],
                       context={"miss_event_id": miss["id"], "skipped": "no_retrieved_events"})
            db.commit()
            continue

        memory_ids = [r["memory_id"] for r in retrieved]
        placeholders = ",".join("?" * len(memory_ids))
        memories = db.execute(
            f"SELECT id, content, summary, themes FROM memories "
            f"WHERE id IN ({placeholders}) AND status = 'active'",
            memory_ids
        ).fetchall()

        if not memories:
            _log_event(db, "_recall", "recall_miss_repaired",
                       query=miss["query"],
                       context={"miss_event_id": miss["id"], "skipped": "memories_deleted"})
            db.commit()
            continue

        cases.append({
            "miss": miss,
            "memories": memories,
        })

    if not cases:
        print("  All failures resolved (no retrieved events or memories deleted)\n")
        db.close()
        return 0

    # 3. Build LLM prompt
    case_blocks = []
    for i, case in enumerate(cases, 1):
        mem_lines = []
        for mem in case["memories"]:
            themes = json.loads(mem["themes"]) if mem["themes"] else []
            prefix = mem["id"][:8]
            mem_lines.append(
                f'  [{prefix}] themes: {json.dumps(themes)} | summary: "{mem["summary"] or ""}"'
            )
        case_blocks.append(
            f'### Case {i}\nQuery: "{case["miss"]["query"]}"\n'
            f'Returned memories (all rated useless):\n' + "\n".join(mem_lines)
        )

    prompt = (
        "You are diagnosing retrieval failures in a personal memory system. Each case shows\n"
        "a query where the system returned memories that the user rated as completely useless\n"
        "(utility <= 0.2). Your job: figure out WHY these memories surfaced and suggest theme\n"
        "modifications to prevent it.\n\n"
        "Possible repairs:\n"
        "1. Remove themes that are too broad and cause false matches. Only remove if genuinely\n"
        "   misleading — not just because it didn't help for one query.\n"
        "2. Add themes that make the memory more precisely findable for its actual topic.\n\n"
        "Cases:\n\n" + "\n\n".join(case_blocks) + "\n\n"
        "Output format:\n"
        '{\n'
        '  "repairs": [\n'
        '    {\n'
        '      "case": 1,\n'
        '      "memory": "abc12345",\n'
        '      "remove_themes": ["overly-broad-theme"],\n'
        '      "add_themes": ["more-specific-theme"],\n'
        '      "reason": "Brief explanation"\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
        "Rules:\n"
        "- Be conservative. Most memories don't need changes.\n"
        "- Never remove ALL themes from a memory.\n"
        "- Don't add more than 2 themes per memory.\n"
        "- Don't remove a theme that accurately describes the memory's content.\n"
        "- Empty repairs array is fine if nothing needs fixing."
    )

    # 4. Call haiku subprocess
    _ensure_subprocess_workspace()
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)

    kwargs = {}
    if sys.platform == "win32":
        CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = CREATE_NO_WINDOW

    claude_bin = shutil.which("claude") or "claude"
    try:
        result = subprocess.run(
            [
                claude_bin, "-p",
                "--model", "sonnet",
                "--no-session-persistence",
                "--strict-mcp-config",
                "--mcp-config", EMPTY_MCP_CONFIG,
                "--system-prompt", (
                    "You are a JSON-only output tool. Respond with the requested "
                    "JSON. No markdown fences, no commentary, no tool calls."
                ),
                "--tools", "",
                "--disable-slash-commands",
                "--setting-sources", "local",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
            cwd=str(_SUBPROCESS_CWD),
            env=env,
            **kwargs,
        )
    except subprocess.TimeoutExpired:
        print("  WARNING: LLM retrieval repair timed out", file=sys.stderr)
        print()
        db.close()
        return 0
    except FileNotFoundError:
        print("  ERROR: 'claude' CLI not found on PATH", file=sys.stderr)
        print()
        db.close()
        return 0

    if result.returncode != 0:
        stderr_preview = (result.stderr or "")[:200]
        print(f"  WARNING: claude -p returned {result.returncode}: {stderr_preview}", file=sys.stderr)
        print()
        db.close()
        return 0

    # Parse JSON from response
    raw = result.stdout.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines)

    try:
        response = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                response = json.loads(raw[start:end])
            except json.JSONDecodeError:
                print(f"  WARNING: could not parse LLM response: {raw[:300]}", file=sys.stderr)
                print()
                db.close()
                return 0
        else:
            print(f"  WARNING: no JSON found in LLM response: {raw[:300]}", file=sys.stderr)
            print()
            db.close()
            return 0

    # 5. Apply repairs
    repairs = response.get("repairs", [])
    repair_count = 0

    # Index repairs by case number
    repairs_by_case = {}
    for r in repairs:
        case_num = r.get("case", 0)
        if case_num not in repairs_by_case:
            repairs_by_case[case_num] = []
        repairs_by_case[case_num].append(r)

    for i, case in enumerate(cases, 1):
        case_repairs = repairs_by_case.get(i, [])
        applied_for_case = []

        for repair in case_repairs:
            mem_prefix = repair.get("memory", "")
            remove_themes = repair.get("remove_themes", [])
            add_themes = repair.get("add_themes", [])
            reason = repair.get("reason", "")

            if not mem_prefix:
                continue

            full_id, err = _resolve_id(db, mem_prefix)
            if err or not full_id:
                print(f"    Skip repair for {mem_prefix}: {err}")
                continue

            mem_row = db.execute(
                "SELECT themes FROM memories WHERE id = ? AND status = 'active'",
                (full_id,)
            ).fetchone()
            if not mem_row:
                continue

            current_themes = json.loads(mem_row["themes"]) if mem_row["themes"] else []

            # Remove themes (case-insensitive)
            remove_lower = {t.lower() for t in remove_themes}
            new_themes = [t for t in current_themes if t.lower() not in remove_lower]

            # Validate at least 1 theme remains
            if not new_themes and current_themes:
                new_themes = current_themes  # don't strip all themes

            # Add themes (cap at MAX_THEMES)
            for t in add_themes[:2]:  # max 2 additions
                if len(new_themes) >= MAX_THEMES:
                    break
                if t not in new_themes:
                    new_themes.append(t)

            # Normalize
            new_themes = normalize_themes(new_themes)

            if new_themes != current_themes:
                db.execute(
                    "UPDATE memories SET themes = ? WHERE id = ?",
                    (json.dumps(new_themes), full_id)
                )
                update_fts(db, full_id)

                actual_removed = [t for t in current_themes if t not in new_themes]
                actual_added = [t for t in new_themes if t not in current_themes]
                applied_for_case.append({
                    "memory": mem_prefix,
                    "removed": actual_removed,
                    "added": actual_added,
                    "reason": reason,
                })
                repair_count += 1
                print(f"    Repaired {mem_prefix}: -{actual_removed} +{actual_added}")

        # Mark this case as processed
        _log_event(db, "_recall", "recall_miss_repaired",
                   query=case["miss"]["query"],
                   context={
                       "miss_event_id": case["miss"]["id"],
                       "repairs_applied": len(applied_for_case),
                       "repairs": applied_for_case,
                   })

    db.commit()
    db.close()

    if repair_count:
        print(f"\n  Applied {repair_count} theme repair(s)")
    else:
        print("  LLM found no repairs needed")
    print()
    return repair_count


def main():
    print(f"{'═' * 60}")
    print(f"  SLEEP CYCLE (deep NREM + REM)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 60}\n")

    total_start = time.time()

    run_theme_normalization()

    nrem_time = run_phase(
        "Phase 1: NREM — edge classification + dedup",
        [sys.executable, str(SCRIPTS / "sleep_nrem.py"), "deep"],
    )

    repair_retrieval_failures()

    rem_time = run_phase(
        "Phase 2: REM — summaries, gestalt, dormancy, health",
        [sys.executable, str(SCRIPTS / "sleep_rem.py")],
    )

    variants = detect_theme_variants()
    suggestions = review_themes_with_llm(variants)
    if suggestions:
        apply_theme_suggestions(suggestions)

    # Phase 3: Retrieval probe — generate feedback signal for underserved memories
    sys.path.insert(0, str(SCRIPTS))
    from probe_recall import run_probe
    print(f"{'─' * 60}")
    print(f"  Phase 3: Retrieval probe")
    print(f"{'─' * 60}")
    print(flush=True)
    probe_start = time.time()
    run_probe()
    probe_time = int(time.time() - probe_start)
    print(f"\n  Probe complete ({probe_time}s)\n", flush=True)

    total = int(time.time() - total_start)
    mins, secs = divmod(total, 60)

    print(f"{'═' * 60}")
    print(f"  DONE — {mins}m {secs}s total (NREM: {nrem_time}s, REM: {rem_time}s, Probe: {probe_time}s)")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
