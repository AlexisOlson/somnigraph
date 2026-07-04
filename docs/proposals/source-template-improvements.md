# Source-Analysis Template Improvements

*Proposal, 2026-07-03. Survey of current practice in `research/sources/` against `research/source-analysis-prompt.md`, with ranked changes, a refreshed context block, and a version-log mechanism. Nothing in the live template or any analysis is modified by this document.*

Scope: read the template, the ~12 most recently added source analyses (via `git log --diff-filter=A`), `research/sources/arhuman-mnemos.md` (just produced, deliberately extension-heavy), plus `STEWARDSHIP.md`, `docs/similar-systems.md`, and `docs/declined.md`. Every claim below cites a file and line I actually opened.

---

## 1. Findings

### 1.1 Analysts consistently ADD a name-collision / disambiguation block (not in the template)

Three of the recent files lead with an explicit "this is a different project from X" block before any analysis:

- `arhuman-mnemos.md` lines 5-11 — bolded "**Name collision, read first**", distinguishing arhuman/mnemos from the existing `ori-mnemos.md`.
- `agentmemory-rohitg00.md` lines 5-7 — "**Name-collision note.** This is **NOT** the system in `agentmemory.md`... `agentmemory.md` is left untouched."
- `engram-gentleman.md` line 9 — "This is a DIFFERENT project from Harshitk-cp/engram (`engram.md`) — do not conflate."

This is now formal corpus infrastructure: `docs/declined.md` Part 3 (lines 184-189) maintains a "Name collisions" registry precisely because collisions recur (engram, agentmemory, Memvid/memv, Memory Palace/MemPalace, claude-mem family, EverOS). The template says nothing about it, so whether a collision gets flagged depends on the individual agent noticing.

### 1.2 Repo-provenance / maturity signals appear ad hoc; the rigorous git-log version is new

Maturity signals show up across the recent files but inconsistently and from inconsistent sources:

- `arhuman-mnemos.md` lines 18-22 — the fullest version, pulled from git: "**Repo age verified from `git log`: initial commit 2026-06-29, most recent commit 2026-07-01, 17 commits total, latest tag `v0.2.0`.** This is a brand-new project — about five days old." It also *declines to guess* star count at that age (line 21-22).
- Star counts elsewhere are taken from README badges with varying skepticism: `clawmem.md` line 3 ("177★"), `arcrift.md` line 9 ("~150 stars"), `engram-gentleman.md` line 117 ("3.8k-star... popularity is a UX/DX signal, not a quality one"), and `claude-mem.md` line 57 ("~79.3k GitHub stars... implausibly high for the mechanism depth — treat as unverified").

The vaporware-vs-real signal (commit count, first-commit date, latest tag) is genuinely useful — mnemos being "five days old, 17 commits" reframes its whole assessment — but only the deliberately-extended mnemos file computed it from `git log`. Everyone else relied on whatever the README badged. This is the clearest "codify the extension" candidate.

### 1.3 Verify-absence-by-grep is common practice, unstated in the template

When an analysis claims a feature is *absent*, the strong versions prove it by grepping rather than inferring from silence:

- `arhuman-mnemos.md` lines 97-101 — "confirmed by grep across `internal/` for `dedup`, `duplicate`, `confidence`, `decay`, `contradiction`, `feedback`, `priority` — the only hits are an unrelated..."; restated in the Summary lines 391-393 ("confirmed by grepping the entire codebase for those concepts and finding nothing").
- `claude-mem.md` line 18 — "has zero readers/writers in `src` (grep for `observation_feedback`/`signal_type` returns only the schema). It is an unrealized stub"; line 37 "(grep confirms)".
- `acontext.md` line 40 — "zero `get_embedding` call sites in the Python core outside the `embeddings/` module".

This is exactly the honest-accounting discipline STEWARDSHIP Priority 1 asks for, and it separates a load-bearing "absent" from a hand-waved one — but it is left to agent temperament.

### 1.4 The Connections section is grounded inconsistently — a verified live failure mode

The template's Connections instruction (lines 123-125) says only "Cross-reference specific source analyses by filename." It does **not** say *only cite systems you actually read*. The strongest recent file added that rule itself, defensively:

- `arhuman-mnemos.md` lines 308-310 — "*(Every system named here was cross-read against its own source-analysis file before being cited, not inferred from index summaries. Systems I did not read are not named.)*" — and reiterates per-item "each verified by reading the linked file" (line 322).

That an agent felt the need to *state* this is the tell. Citing convergence from `index.md` one-liners an agent never opened is a real, repeated error (confirmed live this session). The corpus has ~185 one-line index summaries that make it trivially easy to name-drop a "convergence" that was never checked. One sentence in the template closes it.

### 1.5 A one-word verdict is written by nearly every analysis but the template never asks for it

The Summary Assessments routinely end with a verdict that `docs/declined.md` and `index.md` then depend on:

- `acontext.md` line 110 "Verdict for us: **MAYBE**"; `agentmemory-rohitg00.md` line 126 "Verdict: **MAYBE**, leaning SKIP"; `arcrift.md` line 112 "Verdict: **MAYBE**"; `claude-mem.md` line 110 "Verdict: MAYBE".
- `docs/declined.md` lines 25-86 is an entire index keyed on DIVE/MAYBE/SKIP verdicts with a "Nug" (idea count) column, whose semantics are defined at line 23 ("DIVE = holds an adopt-tier idea; MAYBE = one or more consider-tier angles; SKIP = surveyed, nothing to adopt").

The verdict vocabulary is load-bearing corpus infrastructure that the template leaves implicit, so it comes out as free-form prose ("MAYBE", "MAYBE leaning SKIP", "Verdict for us: MAYBE") that must be hand-normalized into the table.

### 1.6 The 200-300 line target matches almost no repo-only analysis

Actual lengths (authoritative `cat -n` counts from reading each file):

| File | Lines | vs 200-300 target |
|------|-------|-------------------|
| `acontext.md` | 110 | under |
| `claude-mem.md` | 110 | under |
| `arcrift.md` | 112 | under |
| `ai-memory.md` | 117 | under |
| `clawmem.md` | 118 | under |
| `agentmemory-rohitg00.md` | 125 | under |
| `engram-gentleman.md` | 126 | under |
| `arhuman-mnemos.md` | 409 | far over |

Repo-only analyses (no Paper Overview, no PERMA table — both correctly skipped) land at **110-126 lines**, roughly half the stated floor. The one file that came out long (mnemos, 409) is the one that applied the un-templated extensions (collision block, git-age header, grep-absence proofs, per-item Connections verification) — useful content, but 2x over the ceiling. The target as written (200-300, calibrated for paper+repo analyses) is wrong in both directions for the repo-only case that now dominates the corpus.

### 1.7 Honest-caveat flagging is already well-served by the Assessment column

The brief asked whether analysts flag "project honestly caveats its own numbers vs overclaims." They do, heavily — but the existing "Key Claims & Evidence" Assessment column already carries it (`arhuman-mnemos.md` line 175 credits mnemos for labeling its own table "smoke signals, not benchmarks"; `agentmemory-rohitg00.md` line 57 "The repo states this plainly"). No new section needed; this is a strength of the current template, not a gap.

### 1.8 No template section is dead weight; the conditional guards work

PERMA Benchmark Results and Paper Overview are skipped in every recent repo analysis, but both are correctly `<!-- Skip if... -->` guarded and fire when a paper/PERMA source appears. The six Architecture subsections are consistently filled and high-value. I found no section to cut. The template's problem is *missing* discipline and *stale numbers*, not bloat — which is the right problem to have.

### 1.9 Embedded "Somnigraph Context for Agents" block is materially stale

Every stale figure (enumerated in §3). Headline: the block says "26-feature reranker... NDCG=0.7958" (template lines 144, 156, 166) but STEWARDSHIP Priority 3 documents a **31-feature V6 model deployed live 2026-07-02** (NDCG@5k 0.8997, +8.42pp), trained on 4,316 queries not 1,032. "90+ source analyses" (line 176) is now 184 (CLAUDE.md) / 186 files on disk. "14 profiled systems" (line 175) matches no current doc — grep for "profiled systems" across `docs/` returns nothing. An agent pasting this block into a subprompt is feeding the analysis wrong facts about the very system it is comparing against.

---

## 2. Ranked proposed changes

Ordered by value-per-line. Additions marked **[+]**, scope adjustments **[~]**. No trims proposed (see §1.8).

### #1 [+] Connections grounding rule (highest value, one line, prevents a live error)

**Why**: §1.4 — citing unread "convergence" from index one-liners is a verified, repeated failure; the best file already self-imposed the fix. One sentence makes it universal and cheap to enforce in review.

Before (template lines 123-125):
```markdown
## Connections

{How does this system relate to other systems we've analyzed? Cross-reference specific source analyses by filename (e.g., "see memos.md", "convergent with memv's supersession pattern"). Note where multiple systems independently arrive at the same solution — that's strong evidence.}
```

After:
```markdown
## Connections

{How does this system relate to other systems we've analyzed? Cross-reference specific source analyses by filename (e.g., "see memos.md", "convergent with memv's supersession pattern"). Note where multiple systems independently arrive at the same solution — that's strong evidence.

**Only cite a system you actually opened and read.** Do not infer convergence from `index.md` one-line summaries — a filename you name here is a claim you read that file. If a connection is plausible but unverified, say so explicitly or leave it out.}
```

### #2 [+] Structured verdict line closing the Summary (standardizes the declined.md/index integration)

**Why**: §1.5 — the DIVE/MAYBE/SKIP verdict is already written free-form and is load-bearing for `docs/declined.md` and `index.md`. Codifying it removes the hand-normalization step and forces the effort/idea-count the index tracks.

Before (template line 129-131, end of Output Structure):
```markdown
## Summary Assessment

{2-3 paragraphs. What is this system's core contribution? What's the single most important thing for Somnigraph to take from it? What's overhyped or missing? Be honest.}
```

After:
```markdown
## Summary Assessment

{2-3 paragraphs. What is this system's core contribution? What's the single most important thing for Somnigraph to take from it? What's overhyped or missing? Be honest.}

**Verdict**: {DIVE | MAYBE | SKIP} — {one clause}. (DIVE = holds an adopt-tier idea; MAYBE = one or more consider-tier angles; SKIP = surveyed, nothing to adopt. This line feeds the `docs/declined.md` / `index.md` verdict tables — keep the keyword exact.)
```

### #3 [+] Repo-provenance header as the first line of Architecture (vaporware-vs-real signal)

**Why**: §1.2 — commit count / first-commit date / latest tag reframed mnemos's whole assessment ("five days old, 17 commits"), and star counts are already collected but inconsistently and credulously. Codify the git-log source and the required skepticism, in one optional line, so it isn't left to whoever remembers to run `git log`.

Before (template lines 45-49):
```markdown
## Architecture

<!-- For repos: describe what the code actually does, not what the README claims.
     For papers: describe the proposed system.
     For both: reconcile any gaps between paper claims and code reality. -->
```

After:
```markdown
## Architecture

<!-- For repos: describe what the code actually does, not what the README claims.
     For papers: describe the proposed system.
     For both: reconcile any gaps between paper claims and code reality. -->

<!-- Repos only: open with a one-line provenance signal pulled from `git log`
     (NOT the README) — e.g. "First commit 2026-06-29, 17 commits, latest tag v0.2.0
     — ~5 days old." This is the vaporware-vs-real filter. Treat README star counts
     as an unverified popularity signal, not a quality signal; omit rather than guess
     when a repo is too young to assess. -->
```

### #4 [+] Verify-absence-by-grep one-liner in the Architecture guidance

**Why**: §1.3 — "feature X is absent" is a much stronger claim when grep-confirmed, and the honest-accounting ethos (STEWARDSHIP P1) wants the difference between "absent" and "undocumented" made explicit. One clause, folded into the existing Architecture comment above (or the Write Path field), no new section.

Add to the Architecture `<!-- ... -->` block (extends #3's edit):
```markdown
<!-- When you assert a feature is ABSENT (no dedup, no decay, no feedback, no
     reranker), confirm it by grepping the codebase for the concept rather than
     inferring from the README's silence, and say you did. "Grep for `decay`
     returns only telemetry-backoff" is load-bearing; "no decay" alone is not. -->
```

### #5 [+] Conditional name-collision block (add to Output Structure + checklist)

**Why**: §1.1 — collisions recur often enough that `docs/declined.md` maintains a registry; three recent files hand-rolled a lead block. Make it a guarded, standard element so it is consistent and so `declined.md` Part 3 stays in sync.

Add near the top of the Output Structure, right after the byline (template line 26):
```markdown
<!-- If this system's name collides with another entry in research/sources/
     (check index.md before starting), open with a bolded collision note:
     which project this file covers, which file covers the other, and an explicit
     "the other file is left untouched." Cross-check docs/declined.md Part 3. -->
```

Add to "Checklist Before Launching" (after template line 184):
```markdown
- [ ] Name checked against index.md for collisions; collision note added if needed
```

### #6 [~] Replace the length target with a signal-density discipline

**Why**: §1.6 shows repo-only analyses cluster at 110-126 lines while the richest (arhuman-mnemos ~400) run much longer. But a fixed target is the wrong fix — it taxes exactly the sources with the most to learn from. Per Alexis (2026-07-03): "I'm fine with a bit longer if it holds signal. Forcing an exact length punishes ones with the most to learn from." So the correction is to drop the target entirely and make length track signal: no-filler as the floor, self-repetition-is-padding as the only ceiling test.

Before (template line 17):
```markdown
Write a single markdown file following the structure below. Target 200-300 lines. Be specific and evidence-based — cite file paths, function names, line numbers, or paper sections. Do not pad with filler.
```

After:
```markdown
Write a single markdown file following the structure below. There is no fixed length target — length should track signal. The floor is "every line earns its place": a thin or early system yields a short analysis (do not pad to look thorough), and a rich system with many transferable mechanisms earns a long one (do not truncate the parts most worth learning from to hit a number). For orientation only, repo-only analyses have typically run ~110-180 lines and paper+repo deep-dives ~200-400 — descriptions of past practice, not targets. Be specific and evidence-based — cite file paths, function names, line numbers, or paper sections. The real length discipline is not a line count: if you catch yourself repeating a point across the Relevance / Worth Stealing / Not Useful sections, that is padding — cut it.
```

### Not proposed (considered, rejected)

- **Merging "What Somnigraph does better" with "Not Useful For Us"**, and "…does that Somnigraph doesn't" with "Worth Stealing." They overlap in practice, but each pairing serves a distinct read (capability gap vs. adopt decision; strength vs. reject decision). Merging risks losing the adopt/reject framing that makes the files actionable. Leave as-is.
- **A dedicated "honest caveats" subsection** — already served by the Assessment column (§1.7). Adding it would be bloat.

---

## 3. Refreshed "Somnigraph Context for Agents" block (ready to paste)

Every figure below re-verified against `STEWARDSHIP.md` and `CLAUDE.md` on 2026-07-03. Changes from the current template are called out in comments; strip the `<!-- CHANGED -->` markers when pasting.

```markdown
## Somnigraph Context for Agents

Include a condensed version of this section in each agent prompt so the agent knows what Somnigraph is and can write informed comparisons.

*Context last verified against STEWARDSHIP.md / CLAUDE.md: 2026-07-03. If today is much later, re-verify the reranker version, GT size, and source count before trusting them (see the template Changelog).*

### Somnigraph in brief

Somnigraph is a persistent memory system for Claude Code. SQLite + sqlite-vec + FTS5, single-user, MCP-based.

<!-- CHANGED: 26-feature/1032q/0.7958 -> 31-feature V6 / 4316q / 0.8997; note the restore -->
**Retrieval**: Hybrid BM25 + vector search with RRF fusion (k=14, Bayesian-optimized). 31-feature LightGBM reranker (V6) trained on 4,316 real-data queries (NDCG@5k=0.8997, +8.42pp over the hand-tuned formula fallback on the same GT). Deployed and confirmed live 2026-07-02, ending a silent formula-fallback outage that ran 2026-04-07 → 2026-07-02. PPR-based graph expansion via edges detected during sleep.

**Feedback loop**: Explicit per-query utility ratings (0-1 float + durability). EWMA aggregation, UCB exploration bonus. Per-query Spearman r=0.70 with ground truth. Hebbian co-retrieval PMI strengthens edges between memories frequently retrieved together.

**Graph**: Typed edges (supports, contradicts, evolves, revision, derivation) detected during NREM sleep. PPR expansion, novelty-scored adjacency, betweenness centrality as reranker feature.

**Sleep pipeline**: Three-phase offline consolidation — NREM (pairwise classification, edge creation, merge/archive), REM (gap analysis, question generation, taxonomy), orchestrator. LLM-mediated, not rule-based.

**Decay**: Per-category exponential decay with configurable half-lives, floors, and reheat on access.

**Schema**: Memories have category (episodic/semantic/procedural/reflection/meta), priority 1-10, themes array, summary, valid_from/valid_until, decay_rate. Layers: detail/summary/gestalt.

<!-- CHANGED: 26-feature -> 31-feature V6 -->
**Benchmarks**: 85.1% on LoCoMo QA (Opus judge). 31-feature V6 reranker. Multi-hop failure analysis shows 88% vocabulary gap as the retrieval ceiling.

**Key architectural differentiators**: (1) Learned reranker, (2) Explicit feedback loop with measured GT correlation, (3) Offline LLM-mediated consolidation (sleep), (4) Graph-conditioned retrieval via PPR.

<!-- CHANGED: refine write-path claim — secret/PII redaction now runs on the write path -->
**What Somnigraph lacks**: Real-time graph construction (graph builds during sleep, not at write time), entity resolution, multi-user support, write-path *quality/salience* gating (secret/PII redaction *does* already run on the write path via `privacy.py` — the gap is the near-duplicate/salience gate, an active Priority 2), preference-state tracking, cross-domain synthesis evaluation.

### Key files

| Module | Purpose |
|--------|---------|
| `src/memory/reranker.py` | 31-feature (V6) LightGBM reranker, cached meta-computation |   <!-- CHANGED: 26 -> 31 -->
| `src/memory/scoring.py` | PPR expansion, novelty-scored adjacency, formula fallback |
| `src/memory/tools.py` | MCP tool implementations (remember, recall, forget, etc.) |
| `src/memory/fts.py` | FTS5 BM25 search with theme/summary boosting |
| `src/memory/embeddings.py` | Enriched embedding (content + category + themes + summary) |
| `src/memory/db.py` | SQLite schema, edge table, memory CRUD |
| `scripts/sleep_nrem.py` | Pairwise classification, edge creation, merge/archive |
| `scripts/sleep_rem.py` | Gap analysis, question generation, taxonomy |
| `docs/architecture.md` | Master narrative of design decisions |
| `docs/similar-systems.md` | Comparative analysis + the "write-path is the lever" framing |
| `docs/declined.md` | Honest-accounting negative space: surveyed-and-declined systems, collisions |   <!-- ADDED -->
| `docs/ideas-considered.md` | Adopt/consider candidates mapped to Somnigraph modules |   <!-- ADDED -->
| `research/sources/` | 184+ source analyses (this is where output goes) |   <!-- CHANGED: 90+ -> 184+ -->
```

Stale-figure ledger (what changed and why):

| Location | Was | Now | Source |
|----------|-----|-----|--------|
| Retrieval para | "26-feature LightGBM reranker trained on 1032 real-data queries (NDCG=0.7958, +6.17pp)" | "31-feature (V6), 4,316 queries, NDCG@5k=0.8997, +8.42pp" | STEWARDSHIP P3 (V6 deployed 2026-07-02) |
| Retrieval para | (no mention of the outage) | note the 2026-04-07→07-02 silent formula fallback + restore | STEWARDSHIP P3 |
| Benchmarks para | "26-feature reranker" | "31-feature V6 reranker" | STEWARDSHIP P3 |
| reranker.py row | "26-feature" | "31-feature (V6)" | STEWARDSHIP P3 |
| research/sources row | "90+ source analyses" | "184+ source analyses" | CLAUDE.md ("184 source analyses"); 186 files on disk 2026-07-03 |
| similar-systems.md row | "Comparative analysis with 14 profiled systems" | "Comparative analysis + 'write-path is the lever' framing" | no doc supports "14"; grep for "profiled systems" in docs/ returns nothing |
| Key files table | (declined.md, ideas-considered.md absent) | both added | both are current central honest-accounting docs |
| What-Somnigraph-lacks | "write-path quality gating" (flat) | split: quality/salience gate is the gap, redaction already on write path | STEWARDSHIP P2 (2026-07-01 privacy.py audit) |

---

## 4. Version-log proposal

**Recommendation: a `## Changelog` section appended to `source-analysis-prompt.md` itself, plus a one-line version stamp at the top of the file.**

Rationale:
- It matches the convention the project already runs — `STEWARDSHIP.md` keeps its own `## Changelog` (40-80-word entries + links) at the bottom of the file it governs. Same muscle memory, same review habit.
- Co-locating the log with the template is what makes embedded-context staleness *visible next time*: the block whose numbers drift (the Somnigraph Context) and the log that dates those numbers live in one file, so a session editing the template sees "context last verified 2026-07-03" and the changelog in the same view. A separate file would drift out of sight — exactly the failure this is meant to catch.
- A version header line (`<!-- Template v2 -->`) alone is too thin (no rationale, no date); a separate file is over-engineered for a single template. The changelog section is the middle path and the one the repo already validates.

Concrete mechanism:

1. Add a stamp as the file's second line:
   ```markdown
   # Source Analysis Agent Prompt

   <!-- Template v2 — last updated 2026-07-03. Embedded Somnigraph-context figures verified as of this date; see Changelog at the bottom before trusting them. -->
   ```
2. Append at the end of the file:
   ```markdown
   ---

   ## Changelog

   Newest first. When an embedded figure in "Somnigraph Context for Agents" changes (reranker version, GT size, benchmark, source count), bump the version, update the stamp date, and log it here — the whole point of this section is to make context-block staleness visible.

   - **v2 — 2026-07-03**: Refreshed the stale Somnigraph-context block (26→31-feature V6 reranker, 1032→4,316-query GT, NDCG 0.7958→0.8997, 90+→184+ sources, added declined.md/ideas-considered.md, noted the 2026-04-07→07-02 reranker outage/restore and the write-path redaction correction). Added: Connections "only cite systems you read" rule; DIVE/MAYBE/SKIP verdict line; git-log repo-provenance header; verify-absence-by-grep guidance; conditional name-collision block. Adjusted the length target to 120-180 (repo-only) / 200-300 (paper). See `docs/proposals/source-template-improvements.md`.
   - **v1 — (original)**: Initial template. Output structure (Paper Overview, Architecture, Key Claims, PERMA, Relevance, Worth Stealing, Not Useful, Connections, Summary), embedded Somnigraph-context block (26-feature reranker, 90+ sources), launch checklist. Target 200-300 lines.
   ```

The v1 entry is written retroactively so the log starts with a real baseline rather than a stub — future sessions can see what "the original" contained without spelunking git history.
