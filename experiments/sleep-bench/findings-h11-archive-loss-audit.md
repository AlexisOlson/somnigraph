# H11 archive-loss audit: truly-gone vs credit-artifact, content-judged

**Date:** 2026-07-09. **Status:** complete (Phase 1 mechanical trace + Phase 2 content judging). **Not committed** — working analysis on branch `sleep-bench/iterated-cycle`.

## Method

- **Scope:** the 22 GT-evidence memory ids non-active in the c4 store (20 consolidate-archived with `superseded_by`, 2 REM-dormanted at c1_rem), and the 41 (query, archived-id) pairs whose `evidence_ids` include them, out of 196 conv0 queries / 133 distinct evidence ids.
- **Judge:** subagent-as-judge (Claude, Fable 5) reading full texts directly from the c4 store (sqlite `mode=ro`); no API judge scripts. **Single judge, n=1** — no inter-rater check; verdicts carry short decisive quotes so they're auditable.
- **Rubric (axis A, content):** strict specificity — a question asking when/how much/which is only PRESERVED if the stand-in carries that specific; topic-gesturing = PARTIAL; fact present only in the archived memory = LOST. Uncertain PARTIAL-vs-LOST ruled PARTIAL and flagged.
- **Stand-ins per archived id:** the `superseded_by` target detail (all 20 are depth-1, active, layer=detail) and the active covering summary (`generated_from` provenance; every one of the 22 ids has exactly one).
- **Axis B (final class per pair):** content verdict x retrievability (stand-in in c4 top-10, either scorer):
  - PRESERVED + retrieved = **CREDIT-ARTIFACT** (info accessible, evidence-id scorer blind)
  - PRESERVED + not retrieved = **RANKING-FAILURE** (info exists, doesn't surface)
  - PARTIAL/LOST = **TRUE-INFORMATION-LOSS**
- **Temporal questions:** LoCoMo turn content never contains absolute dates (they resolve via the `session-N` theme + session date). A stand-in was ruled at *anchor parity* — PRESERVED — when it keeps the relative phrase verbatim AND carries the same session anchor as the archived original. Verified in the DB: every merged superseder's session theme matches its archived member's session (e.g. 191f3766=session-18, e3129351=session-19, 74da7494=session-9, cc33ecd5=session-15, ca9cba74=session-4).

## Aggregates

**All 41 pairs:** 28 CREDIT-ARTIFACT (68%), 12 RANKING-FAILURE (29%), 1 TRUE-INFORMATION-LOSS (2%, heavily asterisked — see 0:43).
Content axis alone: 40/41 PRESERVED (2 of them vacuously — the archived turn was tangential to its question), 1 PARTIAL, 0 LOST.

**The 23 raw-damaged pairs** (evidence in frozen top-10, unretrievable in c4; identical set both scorers): **16 CREDIT-ARTIFACT, 6 RANKING-FAILURE, 1 TRUE-LOSS\***.

**Per-scorer note:** only pair 0:142 differs between scorers (covering summary retrieved under formula, not rrf). rrf-only classing: 27/13/1.

**The 4 mechanically truly-gone ids** (no stand-in retrieved for any of their queries):

| id | queried fact | survives? | verdict |
|---|---|---|---|
| 94a4e4ff | figurines bought yesterday (0:78, 0:80) | verbatim in unretrieved superseder e3129351; dropped by summary | ranking failure |
| 0261ed84 | pride parade attendance (0:32) | in unretrieved superseder b9de5f5e AND summary fb63a5af | ranking failure |
| b35bc10c | "abstract art" (0:43) | absent from stand-ins — and from the archived turn itself | pre-existing GT gap, not archive-caused |
| 21f368dd (dormant) | pottery types (0:48) | archived turn carried no pottery fact (friendship thanks); no incremental loss | vacuous / GT-noise pointer |

**Bottom line: zero queried facts were destroyed by consolidate's archives or REM's dormancy in this set.** Every non-vacuous fact survives in the superseder (usually verbatim), and usually also in a summary. The entire raw c4 damage decomposes into scorer blindness (16/23) and ranking failures of existing carriers (6/23), plus one pair whose answer text never existed in any memory (1/23).

## Per-pair table

Retrieved = stand-in in c4 top-10 (either scorer). Quotes are from the stand-in that preserves the fact.

| query | archived id | mech | content | where survives | retr. | final class | decisive quote / note |
|---|---|---|---|---|---|---|---|
| 0:106 | f9017270 | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "a pair of purple running shoes" |
| 0:108† | f09d70ab | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "has been great for her mental health" |
| 0:12† | eb66c8da | archive | PRESERVED | both | N | RANKING-FAILURE | "18th birthday (~ten years earlier, so roughly 2013)" |
| 0:13† | 8e41d9e5 | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "keen on counseling or working in mental health" |
| 0:14 | 5dac77e0 | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "pass that same love and support forward" (superseder explicit; retrieved summary adequate) |
| 0:142 | f2cdd4d2 | dormant | PRESERVED | summary | Y (formula only) | CREDIT-ARTIFACT | "ongoing journey of self-discovery, learning, and personal growth"; rrf side is a ranking failure |
| 0:143† | 3ef42836 | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | verbatim: "my son got into an accident" |
| 0:153 | eb365184 | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | verbatim: "Researching adoption agencies" |
| 0:154 | 7510d32c | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "they help LGBTQ+ folks with adoption" |
| 0:155† | 7510d32c | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "inclusivity and support really spoke to me" |
| 0:156 | d3944832 | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "thrilled to make a family for kids who need one" |
| 0:161† | eb66c8da | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "reminds me of art and self-expression" |
| 0:168† | f9017270 | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "purple running shoes" (GT itself confuses speaker) |
| 0:170† | f09d70ab | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "great for my mental health" |
| 0:18† | 19ea6b02 | archive | PRESERVED | both | N | RANKING-FAILURE | verbatim "camping in the mountains last week"; superseder sits just outside top-10 (inside eff-k) |
| 0:191† | 3ef42836 | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | as 0:143 |
| 0:2† | 8e41d9e5 | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "pursuing further education and leaning toward counseling" |
| 0:24† | 30948ac3 | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "Running helps her de-stress" |
| 0:3† | eb365184 | archive | PRESERVED | both | N | RANKING-FAILURE | verbatim "Researching adoption agencies" |
| 0:32 | 0261ed84 | archive | PRESERVED | both | N | RANKING-FAILURE | "A pride parade I went to a few weeks ago" |
| 0:34 | eceae1ac | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "joined a mentorship program for LGBTQ youth" |
| 0:36† | eceae1ac | archive | PRESERVED | superseder (summary drops date) | Y | CREDIT-ARTIFACT | "[session-9] ... Last weekend I joined a mentorship program" (anchor parity) |
| 0:38 | 06dbbc05 | archive | PRESERVED (vacuous) | superseder verbatim | N | RANKING-FAILURE | archived turn carried no activity facts — GT-noise evidence pointer |
| 0:39 | 1a6f7130 | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "staged an LGBTQ-themed art show of her paintings" |
| 0:39 | eceae1ac | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "joined a mentorship program for LGBTQ youth" |
| 0:43† | b35bc10c | archive | PARTIAL (flagged) | — | N | TRUE-LOSS\* | "abstract" absent from stand-ins AND from the archived turn itself — pre-existing GT gap (image-caption answer), archive caused no incremental loss |
| 0:48 | 21f368dd | dormant | PRESERVED (vacuous) | summary has "patterned bowl" | N | RANKING-FAILURE | dormanted turn was friendship thanks — carried no pottery fact |
| 0:63† | b85caa2f | archive | PRESERVED | superseder (summary unanchored) | Y | CREDIT-ARTIFACT | "[session-15] ... talent show for the kids next month" (anchor parity) |
| 0:7† | d3944832 | archive | PRESERVED | both | N | RANKING-FAILURE | "plans to adopt as a single mother" |
| 0:74† | 3ef42836 | archive | PRESERVED | superseder (summary drops date) | Y | CREDIT-ARTIFACT | verbatim "roadtrip this past weekend" + session-18 theme retained |
| 0:75 | 3ef42836 | archive | PRESERVED | superseder | N | RANKING-FAILURE | "my son's ok"; NB summaries blur count to "multiple kids" |
| 0:77† | 3ef42836 | archive | PRESERVED | superseder | N | RANKING-FAILURE | "our trip got off to a bad start. I was really scared" |
| 0:78 | 94a4e4ff | archive | PRESERVED | superseder only | N | RANKING-FAILURE | verbatim "These figurines I bought yesterday"; summary drops figurines |
| 0:80† | 94a4e4ff | archive | PRESERVED | superseder only | N | RANKING-FAILURE | "figurines I bought yesterday" + session-19 theme (anchor parity) |
| 0:81 | 9f089c6b | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "pursuing adoption as a lifelong dream" |
| 0:85 | eb365184 | archive | PRESERVED | superseder | N | RANKING-FAILURE | verbatim "Researching adoption agencies..." (summary loses future-plan framing) |
| 0:86† | 7510d32c | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "they help LGBTQ+ folks with adoption" |
| 0:87† | 7510d32c | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "inclusivity and support really spoke to me" |
| 0:88 | d3944832 | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "thrilled to make a family for kids who need one" |
| 0:89 | 6007fbc2 | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "calling her caring and predicting she'll be a great mom" |
| 0:94† | eb66c8da | archive | PRESERVED | both | Y | CREDIT-ARTIFACT | "reminds me of art and self-expression" |

† = one of the 23 raw-damaged pairs (evidence was in frozen top-10).

## What consolidate's superseders and REM's summaries systematically drop

- **Superseders are near-lossless in this sample.** Consolidate's merge/rewrite pattern appends the member turns *verbatim* below the synthesized paragraph, so quotes, specifics, and relative-time phrases all survive; and every superseder's `session-N` theme matched its archived member's session (temporal anchor parity, checked for all 5 temporal questions). 0 content losses attributable to superseders across 41 pairs. Latent risk observed but not realized here: a merged memory can only carry ONE session theme — if consolidate ever merges across sessions, relative phrases like "yesterday" would lose their anchor.
- **Summaries blur exactly the specifics the rubric targets:** exact counts ("multiple kids" for 3; "two younger kids"), incidental objects (figurines dropped entirely by d9c9801d), relative dates kept but unanchored ("talent show next month" with no session theme — summaries have no `session-N` by design), and speaker attribution loosened. Where the summary is the only retrieved stand-in, when/how-many questions are at risk; in this sample the superseder always backstopped them.
- **LoCoMo GT noise surfaced by the audit:** 2 evidence pointers tangential to their questions (0:38, 0:48), 1 golden answer never present in the pointed-at turn's text (0:43 "abstract art", likely image-caption evidence), and 2 questions with speaker attribution swapped (0:168, 0:94 mirror 0:106/0:161 with the names exchanged).

## Relation to the mechanical trace (Phase 1)

Phase 1 (same session) found: 22 non-active evidence ids, 18 with a stand-in retrieved somewhere, 4 truly-gone; 16/23 damaged pairs had a top-10 stand-in. Phase 2 sharpens this to: the mechanical bound is in fact realized — **all 16 retrieved-stand-in damaged pairs are genuine credit artifacts** (the retrieved stand-in carries the queried fact), and the remaining 7 are 6 ranking failures + 1 pre-existing GT gap, **not** content destruction. Scratch artifacts (dossier, CSVs, scripts) in the session scratchpad: `gone_vs_uncredited.py`, `judge_dossier.json`, `archived_evidence_ids.csv`, `query_id_pairs.csv`.
