# 2WikiMultiHopQA -- Analysis

*Generated 2026-03-27 by Opus 4.6 agent reading the GitHub repo and COLING 2020 paper*

---

## Paper Overview

**Paper**: Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, Akiko Aizawa (The Graduate University for Advanced Studies / NII / AIST, Japan). "Constructing A Multi-Hop QA Dataset for Comprehensive Evaluation of Reasoning Steps." COLING 2020, pages 6609-6625. Code: https://github.com/Alab-NII/2wikimultihop (Apache 2.0, ~144 stars).

**Problem addressed**: HotpotQA (2018) enabled multi-hop QA evaluation but provided only sentence-level supporting facts as evidence -- enough to verify *where* a model looked, not *how it reasoned*. Models can achieve high scores on HotpotQA without genuine multi-hop reasoning (Min et al. 2019 showed single-hop shortcuts suffice). 2WikiMultiHopQA adds structured evidence triples that explicitly describe the reasoning chain, enabling evaluation of whether models can *explain* their reasoning path, not just retrieve correct answers.

**Core claim**: By combining Wikidata (structured knowledge base triples) with Wikipedia (unstructured text), the dataset provides three evaluation layers per question: answer prediction, sentence-level supporting facts, and structured evidence triples. The evidence triples are the novel contribution -- they make the reasoning chain explicit and machine-verifiable.

**Scale**: 192,606 questions across train/dev/test splits. Evaluated with EM/F1 on three tasks (answer, supporting facts, evidence) plus joint metrics.

---

## Dataset Construction

### Three-step pipeline

**1. Template creation.** Comparison templates: extracted NER tags from 17,456 HotpotQA comparison questions, replaced entity names with slots, focused on top-50 popular entity types. Inference/compositional templates: manually authored. All templates explicitly filtered against Min et al.'s taxonomy to discard single-hop and context-dependent multi-hop patterns.

**2. Algorithmic generation.** For comparison: randomly select two entities from the same Wikidata `instance_of` group, find mutual relations, verify values appear in Wikipedia article summaries, fill template. For bridge (inference/compositional): randomly select entity, traverse Wikidata relations to find 2-hop paths, verify bridge entity requirements in Wikipedia text, ensure answer is extractable as a span. Key: the answer must be in the text, not just in Wikidata.

**3. Post-processing.** Balance yes/no for comparison. Discard ambiguous cases (multiple valid answers). Remove questions where compared numerical values are equal.

**Data source**: English Wikipedia dump (Jan 1, 2020) + English Wikidata dump (Dec 31, 2019). 5,950,475 entities categorized into 23,763 groups.

**Distractor paragraphs**: Following HotpotQA, bigram TF-IDF retrieves top-50 similar paragraphs from Wikipedia, then top-8 (top-6 for bridge-comparison) matching the entity type. Each question has 10 total paragraphs (gold + distractors), shuffled.

### Four question types

| Type | Count | Hops | Description |
|------|-------|------|-------------|
| **Comparison** | 57,989 | 2 | Compare two entities on the same property. Many yes/no. |
| **Inference** | 7,478 | 2 | Logical rules derive new relations (28 rules, Table 10). E.g., father(a,b) + father(b,c) => grandfather(a,c). |
| **Compositional** | 86,979 | 2 | Chain two relations sequentially, no inference. Largest category. 799 templates from 13 first-hop + 22 second-hop properties. |
| **Bridge-comparison** | 40,160 | 3-4 | Bridge reasoning + comparison. Uses 4 gold paragraphs instead of 2. Limited to human+film entity combinations. |

### Dataset splits

| Split | Size | Selection criterion |
|-------|------|---------------------|
| train-medium | 154,878 | Solved by single-hop BERT (86.7% F1, 5-fold CV) |
| train-hard | 12,576 | Not solved by single-hop BERT |
| dev | 12,576 | Same |
| test | 12,576 | Same |

Average question length: 12.64 words. Average answer length: 1.94 words. 708 unique answer types via Wikidata `instance_of`. Answer distribution: yes/no 31.2%, date 16.9%, film 13.5%, human 11.7%, big city 4.7%, other 22.0%.

---

## Evidence Structure

Each question has three annotation layers:

**`supporting_facts`**: List of `[title, sent_id]` pairs pointing to specific sentences in gold paragraphs. Same format as HotpotQA. Binary classification task per sentence.

**`evidences`**: List of `[subject_entity, relation, object_entity]` triples representing the structured reasoning path. This is the novel annotation. For a 2-hop question, 2 triples; for bridge-comparison, more.

```json
{
  "evidence": [
    ["silver star 1910 song", "composer", "charles l johnson"],
    ["charles l johnson", "place of birth", "kansas city kansas"]
  ]
}
```

**`evidences_id`** (added Dec 2020): Wikidata IDs for subject and object entities, enabling alias-aware evaluation. An `id_aliases.json` maps Wikidata IDs to sets of valid aliases + demonyms.

The evidence triples provide what the paper calls both **justification** (evidence supporting the answer) and **introspective explanation** (showing how the reasoning proceeds). Explicitly contrasted with HotpotQA (justification only) and R4C (semi-structured derivations, only 4,588 questions).

---

## Key Findings

**Harder than HotpotQA.** Same multi-hop model (Yang et al. 2018): HotpotQA Ans F1 58.54 vs 2WikiMultiHop 40.95 (-17.6pp). Joint EM 10.97 vs 9.22.

**Reduced single-hop shortcuts.** Single-hop BERT F1: HotpotQA 64.6, 2WikiMultiHop 55.9. The 8.7pp gap confirms more questions genuinely require multi-hop reasoning. But not eliminated (see Limitations).

**Evidence generation is extremely hard.** Baseline: Evi EM 1.07 / F1 14.94. Human performance: Evi EM 64.0 / F1 78.81. The 62pp model-human gap is the largest across all three tasks.

**Per-type difficulty.** Inference questions are hardest overall (Joint F1 1.40). Bridge-comparison has highest Sup F1 (89.16) but lowest Ans F1 (20.45) -- models can find the right paragraphs but can't do the comparison. Compositional has highest Ans F1 (59.94) and Evi F1 (17.65).

**Wikipedia/Wikidata mismatch.** ~8% of randomly checked training samples have mismatches between Wikipedia content and Wikidata triples (e.g., Wikidata says X is spouse of Y, but the article doesn't mention this relationship).

---

## Evaluation Metrics

Three tasks evaluated simultaneously:

1. **Answer prediction**: EM and F1 (SQuAD-style)
2. **Supporting facts**: Set-level P/R/F1 on `[title, sent_id]` tuples
3. **Evidence generation**: Set-level P/R/F1 on `[subject, relation, object]` triples

**Joint metrics**: `P_joint = P_ans * P_sup * P_evi`, `R_joint = R_ans * R_sup * R_evi`, `Joint F1 = 2 * P_joint * R_joint / (P_joint + R_joint)`. Joint EM = 1 only when all three tasks are exact match.

v1.1 evaluation adds alias handling: answers checked against all Wikidata aliases + demonyms; evidence triples expand to all subject/object alias combinations.

---

## Known Limitations and Criticisms

### Internal limitations

- **Template-based construction**: All questions follow predefined patterns, limiting diversity. 799 compositional templates, 62 bridge-comparison templates.
- **Wikipedia/Wikidata inconsistency**: ~8% mismatch rate is a known noise source.
- **Entity ambiguity**: Partially addressed by Dec 2020 alias update but not fully resolved.
- **Limited domain coverage**: Only Wikipedia summaries (first paragraphs), not full articles. Bridge-comparison limited to human+film entities.

### External critiques (MuSiQue, Trivedi et al. 2022)

MuSiQue's Table 4 provides the definitive comparative vulnerability analysis:

| Vulnerability test | 2WikiMultiHop | HotpotQA | MuSiQue-Ans |
|--------------------|---------------|----------|-------------|
| DiRe score (answer F1) | 63.4 | 68.8 | **37.8** |
| DiRe score (support F1) | 98.5 | 93.0 | **63.4** |
| 1-Para model (answer F1) | 60.1 | 64.8 | **32.0** |
| C-only model (answer F1) | 50.1 | -- | **15.2** |
| C-only model (support F1) | 92.0 | -- | **55.2** |
| Human-model gap (Ans F1) | ~5pp | ~10pp | **~27pp** |

- **DiRe = 63.4**: Models can get 63.4% answer F1 by answering sub-questions independently without connecting reasoning steps.
- **C-only = 92.0 support F1**: Supporting paragraphs can be identified with 92% F1 *without seeing the question*, indicating severe distributional bias in how gold paragraphs differ from distractors.
- **Human-model gap ~5pp**: The dataset is nearly "solved" for answer prediction.

MuSiQue's core critique: "While 2WikiMultihopQA was also constructed via composition, they use a limited set of hand-authored compositional rules, making it easy for large language models."

---

## Relevance to Somnigraph

### Direct relevance: vocabulary gap characterization

The inference question type is a **pure vocabulary-gap scenario** -- precisely the pattern causing Somnigraph's multi-hop retrieval failures (88% zero content-word overlap between query and evidence in LoCoMo multi-hop misses).

Example: "Who is the paternal grandfather of John Cecil, 7th Earl of Exeter?" Evidence chain: `father(John Cecil 7th, John Cecil 6th)` + `father(John Cecil 6th, John Cecil 5th)`. The word "grandfather" never appears in either supporting paragraph -- it exists only via the logical rule. The 28 inference rules (Table 10) are essentially a **catalog of vocabulary gaps**: "grandchild" requires "child+child", "mother-in-law" requires "spouse+mother", "educated_at" requires "doctoral_advisor+employer".

Compositional questions also create vocabulary gaps when the composed phrasing ("the founder of the company that distributed La La Land") doesn't match either paragraph individually. The bridge entity connects paragraphs that share no vocabulary with each other.

### Where the relevance diverges

2WikiMultiHop is a **reading comprehension** benchmark (10 provided paragraphs), not an open-retrieval benchmark. Its retrieval challenge is within a small context window, not a large corpus. The MuSiQue vulnerabilities (DiRe 63.4, C-only 92.0 support F1) mean its multi-hop difficulty is significantly overstated -- a system could achieve good scores through distributional shortcuts rather than genuine multi-hop reasoning.

For Somnigraph's purposes, MuSiQue is the more rigorous multi-hop benchmark (DiRe 37.8 vs 63.4, human-model gap 27pp vs 5pp). However, 2WikiMultiHop's structured evidence triples and explicit inference rules have no equivalent in MuSiQue, and the rules systematically catalog the kinds of vocabulary gaps that cause retrieval failures.

### Actionable ideas

1. **Vocabulary gap taxonomy from inference rules.** The 28 rules in Table 10 define the structure of vocabulary gaps -- useful for categorizing Somnigraph's multi-hop failures and designing targeted vocabulary bridges. Not all gaps are the same: "grandfather = father+father" is lexically close, while "educated_at = doctoral_advisor+employer" is lexically distant.

2. **Controlled vocabulary-bridge evaluation.** The template-based construction means the exact nature of each vocabulary gap is known. One could construct a retrieval task from 2WikiMultiHop (pool all paragraphs into a shared corpus, require retrieval) specifically to test whether graph-based vocabulary bridges close known gaps. The controlled structure makes failure analysis precise in a way LoCoMo's natural conversations don't.

3. **The inference rules are the real contribution.** Despite the benchmark's vulnerabilities, the 28 logical rules are a clean formalization of multi-hop reasoning types. They could inform what kinds of synthetic nodes the graph pipeline should produce -- currently Somnigraph's synthetics are claim/segment extractions, but inference-style synthetics ("if A's father is B and B's father is C, then A's grandfather is C") would directly target the vocabulary gap.

### What not to do

Don't use 2WikiMultiHop as a comparative benchmark -- MuSiQue's critique is devastating on the multi-hop rigor front (DiRe 63.4 means 63% of questions are solvable without multi-hop reasoning). LoCoMo and MuSiQue remain better evaluation targets. The value is in the inference rules and evidence structure, not the benchmark scores.

---

## Comparison with Benchmarks in Catalog

| Feature | 2WikiMultiHop | LoCoMo | MuSiQue | HotpotQA |
|---------|---------------|--------|---------|----------|
| Year | 2020 | 2024 | 2022 | 2018 |
| Domain | Wikipedia | Conversations | Wikipedia | Wikipedia |
| Size | 192,606 | ~1,000 | ~25K | 112,779 |
| Hops | 2-4 | 1-3 | 2-4 | 2 |
| Evidence annotation | Triples + sentences | Sentences | Paragraphs + DAG | Sentences |
| Multi-hop rigor (DiRe) | 63.4 | Not measured | **37.8** | 68.8 |
| Construction | Template + Wikidata | Natural conversation | Bottom-up composition | Crowdsourced |
| Logical inference rules | Yes (28) | No | No | No |
| Retrieval setting | 10 paragraphs | Open (full DB) | 20 paragraphs | 10 paragraphs |
| Somnigraph use | Vocabulary gap taxonomy | Primary benchmark | Potential benchmark | Superseded |
