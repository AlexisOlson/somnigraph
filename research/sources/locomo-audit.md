# LoCoMo Audit — Source Analysis

*2026-03-22. Analysis of PenfieldLabs' independent audit of the LoCoMo-10 benchmark dataset and published evaluation frameworks.*

**Repository:** github.com/dial481/locomo-audit
**Authors:** PenfieldLabs (Claude Opus 4.6 audit with human review)
**Builds on:** 29 errors previously reported in snap-research/locomo#27, extended to 156 total (5x)

## 1. What This Is

A systematic audit of LoCoMo-10's ground truth answers, the LLM judge used for evaluation, and the methodology of five published systems (EverMemOS, Mem0, MemOS, MemU, Zep). The audit is rigorous: SHA256-verified source files, reproducible scripts, all 25 published scores reproduced exactly.

## 2. Ground Truth Errors

**156 total issues across 1,540 non-adversarial questions. 99 are score-corrupting (6.4%).**

| Error Type | Count | Pattern |
|---|---|---|
| HALLUCINATION | 33 | Golden answers contain facts not in the transcript |
| TEMPORAL_ERROR | 26 | Wrong date/time calculations |
| ATTRIBUTION_ERROR | 24 | Statement attributed to wrong speaker |
| AMBIGUOUS | 13 | Partially correct or debatable |
| INCOMPLETE | 3 | Missing facts stated in transcript |

The remaining 57 are citation-only errors (correct answer, wrong evidence dialog IDs) — don't affect scoring.

**Error rates by category:**
- Category 1 (multi-hop): 9.9% — worst, driven by hallucinations (20 of 28)
- Category 2 (temporal): 8.1% — driven by date miscalculations
- Category 3 (open-domain): 9.4%
- Category 4 (single-hop): 4.3% — cleanest, but 22 attribution errors

**Theoretical scoring ceiling: 93.57%.** A perfect system cannot score higher on the corrupted dataset.

### Hallucination patterns

Two systematic sources:

1. **Image query metadata leakage.** LoCoMo conversations include image shares. The `query` field (annotator search strings for stock photos — e.g., "Ferrari 488 GTB", "car museum", "gold chain", "transgender symbol") was never part of actual dialog, but golden answers reference these terms as if they were discussed. Affected IDs include locomo_9_qa1, locomo_9_qa19, locomo_9_qa60, locomo_0_qa56.

2. **Fabricated specifics.** Golden answers include details not present in any transcript: book title "Nothing is Impossible" (locomo_0_qa23/26), "A Court of Thorns and Roses" (locomo_3_qa34), "Gamecube, Playstation" (locomo_3_qa61), "animal keeper at a local zoo" (locomo_3_qa66), "Psychology, counseling certification" (locomo_0_qa2).

### Temporal error patterns

Consistent miscalculation of relative time expressions: "Last Saturday" consistently wrong (locomo_0_qa5), "six months" for 5 (locomo_1_qa31), "19 days" for 9 (locomo_6_qa31).

### Distribution

Errors appear across all 10 conversations (4.9%–9.9% per conversation), all 4 non-adversarial categories, following systematic annotator confusion patterns. This is pervasive, not sporadic.

## 3. LLM Judge Failures

Stress-tested the standard LLM judge (gpt-4o-mini with "be generous" instructions) using adversarial answer sets:

- **V1 (specific-but-wrong):** 10.61% accepted — judge catches obvious factual errors
- **V2 (vague-but-topical):** 62.81% accepted — answers like "sometime in early fall" for "October 2nd" fool the judge

The 6x gap isolates the failure mode: the judge is not fooled by wrong facts but **is fooled by topical vagueness**. The "be generous, as long as it touches on the same topic" instruction creates systematic bias toward vague, long answers.

V2 scores higher than Mem0 (64.20%) and MemU (66.67%) in several categories — a system that knows every answer and deliberately gets them wrong scores comparably to real memory systems.

## 4. Methodology Problems

### Prompt asymmetry

Four different answer prompts across five published systems. EverMemOS uses a 729-token 7-step CoT prompt with no word limit (producing ~49-word answers). Mem0/MemOS/MemU use a "5-6 words" constrained prompt. The prompt alone accounts for a **10.67-point accuracy difference** with the same model and identical context (81.95% vs 92.62%).

### Token cost misrepresentation

EverMemOS claims 2,298 avg tokens per question. Their own paper's Table 8 shows the real Phase III cost is 6,045–6,669 tokens/question (2.6–2.9x higher). Actual reduction vs full-context is 67%, not the claimed 89%.

### Full-context baseline exceeds EverMemOS

With the same CoT prompt, GPT-4.1-mini on full context scores **92.62%**, exceeding EverMemOS (92.32%). The memory system provides no measurable accuracy gain. The claimed 91.21% FC baseline is unverifiable (no code, no results published).

### Answer length correlation

Systems without word limits produce answers 10–11x the golden answer length, providing more surface area for the judge's generous matching. Spearman ρ = 0.64 between word count and accuracy.

### No statistical significance tests

None of the five evaluated papers report significance tests. At 6.4% GT error rate and 62.81% judge false-acceptance rate, claimed differences of 2–3 points are well within noise.

## 5. Ceiling Violations

After correcting all 99 GT errors, EverMemOS single-hop (96.08% adjusted vs 95.72% ceiling) and multi-hop (91.25% vs 90.07%) **exceed mathematically possible scores**. This indicates second-order judge leniency that no amount of GT correction can address — the judge is giving credit for wrong answers even on corrected questions.

## 6. Reproducibility

Third-party reproducibility is poor across the field:
- **EverMemOS:** 38.38% reproduced vs claimed 92.32% (GitHub issue #73)
- **Mem0:** ~20% reproduced on platform (issue #3944; root cause: timestamps stored as current date)
- **Zep:** Category 5 scoring bug inflated scores (issue #5, acknowledged and fixed)
- Cross-repository discrepancies: different models, prompts, scoring methods, judge models

## 7. Corrected Answers

The audit provides corrected answers for all 99 score-corrupting errors in `errors.json`. Each entry includes original question, golden answer, error type, detailed reasoning with transcript evidence, and corrected answer.

Net score adjustments across systems are small (<0.6pp), confirming errors don't systematically favor any one system — but they destroy the precision needed for system comparisons.

## 8. Implications for Somnigraph

### Our benchmark is affected

We run LoCoMo end-to-end QA (Priority 4 in STEWARDSHIP.md). Our 85.1% overall includes questions with wrong golden answers. Specific impacts:

1. **Multi-hop (our weakest: 61.5%)** has the highest GT error rate (9.9%). Some of our "misses" may be correct answers judged against wrong GT. This changes the interpretation of our 76% retrieval hit rate — the ceiling for that category is ~90%, not 100%.

2. **Temporal (our strongest after single-hop)** has 8.1% GT error rate with systematic date miscalculation. We should check whether our temporal answers match the *corrected* GT or the original.

3. **Our judge (GPT-4.1)** is different from the audited judge (gpt-4o-mini). The V2 vague-answer acceptance rate may differ. Our structured judge prompt may partially mitigate the "be generous" failure mode, but we haven't tested this.

4. **Our adversarial exclusion (cat 5) is validated.** The audit confirms no reliable GT exists for adversarial questions.

### Actionable items

1. **Import corrected GT.** Use `errors.json` to patch our local copy of locomo10.json. Rerun scoring against corrected answers to get adjusted numbers. This is the minimum.

2. **Audit our judge.** Run the V2 (vague-but-topical) adversarial test against our GPT-4.1 judge with our prompt. If acceptance rate is high, our 85.1% is inflated.

3. **Report both scores.** Once corrected GT is available, report original and corrected accuracy. Honest accounting (Priority 1).

4. **Category-level reanalysis.** Our multi-hop gap may be smaller than measured. Rerun just category 1 and 2 against corrected GT before investing more in expansion methods.

### What this doesn't change

- The expansion work on `expansion-wip` addresses real retrieval gaps regardless of GT quality. Even with corrected GT, multi-hop retrieval hit rate won't jump from 76% to 94%.
- The reranker was trained on our own ground truth, not LoCoMo's. No contamination.
- The relative ordering of systems on LoCoMo is approximately preserved (<0.6pp adjustment). Our 85.1% vs Mem0's 66.88% is not an artifact.

## 9. Broader Takeaways

### LoCoMo is useful but flawed

The audit doesn't invalidate LoCoMo — it calibrates it. A 93.57% ceiling means systems claiming 90%+ should be scrutinized. Systems in the 60–85% range are meaningfully differentiated. But the judge's 62.81% vague-acceptance rate means any score should be interpreted as a lower bound on how good the system *might* be, not a measurement of how good it *is*.

### The field needs better evaluation

PenfieldLabs' four-point wishlist (from the Reddit thread) is validated by their own audit:
1. Corpus larger than context window — LoCoMo's 9K–26K fits easily
2. Current models — many evaluations use gpt-4o-mini
3. Better judge — the "be generous" instruction is load-bearing and harmful
4. Realistic ingestion — synthetic conversations miss corrections, evolving relationships

### Prompt is a confound

The 10.67-point prompt effect is larger than most system-vs-system differences reported in the literature. Any LoCoMo comparison that doesn't control for prompt is measuring the prompt, not the system. Our benchmark should document our prompt and test sensitivity.

## 10. Key Takeaway

LoCoMo's ground truth has a 6.4% score-corrupting error rate, its judge accepts 62.81% of intentionally vague wrong answers, and the answer prompt alone explains 10.67 points of accuracy. These three facts compound: the benchmark is measuring a mixture of retrieval quality, prompt engineering, and judge tolerance, with substantial noise from GT errors.

For Somnigraph, the immediate action is importing corrected GT and rerunning scores. The expansion work remains well-motivated — retrieval gaps are real even after GT correction. But we should stop treating LoCoMo accuracy as a precise measurement and start treating it as a noisy indicator with known biases.

## See Also

- [[locomo]] — Original LoCoMo benchmark analysis
- [[locomo-plus]] — LoCoMo-Plus cognitive memory extension
- [[evermemos]] — EverMemOS architecture analysis (claims audited here)
- [[mem0-paper]] — Mem0 paper analysis
