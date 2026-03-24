# Precise Zero-Shot Dense Retrieval without Relevance Labels -- Analysis

*Generated 2026-03-22 by Opus 4.6 agent reading arXiv:2212.10496v1 (HyDE)*

---

## Paper Overview

**Paper**: Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan (Carnegie Mellon University, University of Waterloo). "Precise Zero-Shot Dense Retrieval without Relevance Labels." ACL 2023. arXiv:2212.10496v1, Dec 2022. 9 pages + appendix. Code: https://github.com/texttron/hyde.

**Problem addressed**: Zero-shot dense retrieval remains difficult because learning query-document similarity requires relevance labels that are expensive or unavailable. Transfer learning from MS-MARCO helps but assumes access to a large supervised corpus. Without relevance labels, unsupervised dense retrievers (Contriever) underperform classical BM25 on many tasks.

**Core claim**: By decomposing retrieval into two steps -- (1) an instruction-following LLM generates a hypothetical document that *would* answer the query, and (2) an unsupervised contrastive encoder embeds that hypothetical document for corpus search -- the query-document similarity problem is replaced by a document-document similarity problem. The encoder's dense bottleneck acts as a lossy compressor, filtering out hallucinated details while preserving the relevance pattern. No models are trained or fine-tuned.

**Scale**: 11 query sets across web search (TREC DL19/DL20), low-resource retrieval (6 BEIR datasets: SciFact, ArguAna, TREC-COVID, FiQA, DBPedia, TREC-NEWS), and multilingual retrieval (Mr.TyDi: Swahili, Korean, Japanese, Bengali). Compared against BM25, Contriever/mContriever (unsupervised), DPR, ANCE, ContrieverFT/mContrieverFT (supervised). Generative models tested: InstructGPT (175B), Cohere (52B), FLAN-T5 (11B).

---

## Architecture / Method

### The Core Insight

Standard dense retrieval learns two encoder functions -- `enc_q` for queries and `enc_d` for documents -- trained so their inner product captures relevance:

```
sim(q, d) = <enc_q(q), enc_d(d)>
```

Without relevance labels, learning `enc_q` is intractable: there's no signal for what a "relevant" query embedding should look like. HyDE sidesteps this by eliminating the query encoder entirely. Instead:

1. An instruction-following LLM generates a hypothetical document `d_hat` that would answer the query
2. A contrastive encoder (the same one used for corpus documents) embeds `d_hat`
3. The resulting vector searches the corpus via document-document similarity

```
sim(q, d) = <f(g(q, INST)), f(d)>
```

Where `g` is the generative LLM, `f` is the contrastive encoder, and `INST` is a task-specific instruction (e.g., "write a passage to answer the question").

### Why This Should Work (and When It Doesn't)

The method rests on two assumptions:

1. **The LLM captures relevance patterns.** Even when it hallucinates specific facts, the generated document uses vocabulary, structure, and topic focus that resemble a real relevant document. The generation is an *example* of what a relevant document looks like.

2. **The encoder is a lossy compressor.** The dense bottleneck (768-dim for Contriever) filters out hallucinated details by mapping the hypothetical document to a neighborhood of real documents with similar patterns. Factual errors in the hypothetical document are "smoothed away" because they don't recur in real corpus documents.

The paper acknowledges but doesn't deeply explore when these assumptions fail. The encoder cannot filter hallucinated details when: (a) the hallucination shifts the entire embedding to a wrong region (systematic topical drift), (b) the corpus is specialized enough that the LLM's general-knowledge hallucinations resemble real documents on a different topic, or (c) the query is ambiguous and the LLM generates a hypothesis for the wrong interpretation.

### Multi-Hypothesis Averaging

The formalization supports generating N hypothetical documents and averaging their embeddings:

```
v_q = (1/N) * sum(f(d_hat_k))   for k = 1..N
```

Or including the original query as an additional hypothesis:

```
v_q = (1/(N+1)) * [sum(f(d_hat_k)) + f(q)]
```

The paper uses N=1 in all experiments (single generation with temperature 0.7). Multi-hypothesis averaging is formalized but not empirically explored -- a notable omission given that averaging could mitigate directional hallucination risk.

### Task-Specific Instructions

Each dataset gets a tailored instruction prompt:

| Task | Instruction |
|------|-------------|
| Web Search | "Please write a passage to answer the question" |
| SciFact | "Please write a scientific paper passage to support/refute the claim" |
| ArguAna | "Please write a counter argument for the passage" |
| TREC-COVID | "Please write a scientific paper passage to answer the question" |
| FiQA | "Please write a financial article passage to answer the question" |
| TREC-NEWS | "Please write a news passage about the topic" |
| Mr.TyDi | "Please write a passage in [language] to answer the question in detail" |

These instructions are the *only* task-specific component. The key engineering insight: the instruction shapes the *genre* and *register* of the generated document, not just its content. Asking for a "scientific paper passage" produces different vocabulary and structure than a "news passage," and the contrastive encoder maps these to different corpus neighborhoods.

### Interaction with Fine-Tuned Encoders

An analysis experiment replaces the unsupervised Contriever with ContrieverFT (fine-tuned on MS-MARCO). Results on TREC DL19/DL20:

| Generator | w/ Contriever | w/ ContrieverFT |
|-----------|--------------|-----------------|
| None (baseline) | 44.5 / 42.1 | 62.1 / 63.2 |
| FLAN-T5 (11B) | 48.9 / 52.9 | 60.2 / 62.1 |
| Cohere (52B) | 53.8 / 53.8 | 61.4 / 63.1 |
| InstructGPT (175B) | 61.3 / 57.9 | 67.4 / 63.5 |

Weaker LMs *hurt* the fine-tuned encoder (FLAN-T5 degrades ContrieverFT on DL19 from 62.1 to 60.2). InstructGPT still helps, suggesting the LLM captures factors the fine-tuned encoder misses. But the degradation with weaker LMs is important: a bad hypothesis is worse than no hypothesis when the encoder is already well-calibrated.

---

## Key Claims & Evidence

### Web Search (TREC DL19 / DL20, NDCG@10)

| System | DL19 | DL20 | Relevance Labels? |
|--------|------|------|--------------------|
| BM25 | 50.6 | 48.0 | No |
| Contriever | 44.5 | 42.1 | No |
| **HyDE** | **61.3** | **57.9** | **No** |
| DPR | 62.2 | 65.3 | Yes (MS-MARCO) |
| ANCE | 64.5 | 64.6 | Yes (MS-MARCO) |
| ContrieverFT | 62.1 | 63.2 | Yes (MS-MARCO) |

HyDE closes the gap between unsupervised and supervised retrievers. On DL19, HyDE matches ContrieverFT; on DL20, it trails by ~5 NDCG points.

### Low-Resource Retrieval (BEIR, NDCG@10)

| Dataset | BM25 | Contriever | HyDE | ContrieverFT |
|---------|------|------------|------|--------------|
| SciFact | 67.9 | 64.9 | **69.1** | 67.7 |
| ArguAna | 39.7 | 37.9 | **46.6** | 44.6 |
| TREC-COVID | **59.5** | 27.3 | 59.3 | 59.6 |
| FiQA | 23.6 | 24.5 | 27.3 | **32.9** |
| DBPedia | 31.8 | 29.2 | 36.8 | **41.3** |
| TREC-NEWS | 39.5 | 34.8 | **44.0** | 42.8 |

HyDE beats Contriever on all 6 datasets. It beats ContrieverFT (supervised) on SciFact, ArguAna, and TREC-NEWS. The FiQA and DBPedia gaps suggest domain-specific retrieval (financial posts, entities) benefits more from supervised fine-tuning than from hypothetical document generation.

### Multilingual Retrieval (Mr.TyDi, MRR@100)

| Language | BM25 | mContriever | HyDE | mContrieverFT |
|----------|------|-------------|------|---------------|
| Swahili | 38.9 | 38.3 | 41.7 | **51.2** |
| Korean | 28.5 | 22.3 | 30.6 | **34.2** |
| Japanese | 21.2 | 19.5 | **30.7** | 32.4 |
| Bengali | **41.8** | 35.3 | 41.3 | 42.3 |

HyDE improves mContriever consistently but trails mContrieverFT by larger margins than in English. The authors attribute this to under-trained non-English capabilities in the backbone LLM (InstructGPT).

### Generator Model Scaling (TREC DL19/DL20, NDCG@10)

| Generator Size | DL19 | DL20 |
|---------------|------|------|
| Contriever (no HyDE) | 44.5 | 42.1 |
| FLAN-T5 (11B) | 48.9 | 52.9 |
| Cohere (52B) | 53.8 | 53.8 |
| InstructGPT (175B) | 61.3 | 57.9 |

Larger LMs produce better hypothetical documents. The gap between 11B and 175B is substantial (12.4 NDCG on DL19).

### Methodological Strengths

- **Genuinely zero-shot.** No training, no fine-tuning, no relevance labels. The only prior knowledge is in the LLM's instruction-following capability and the contrastive encoder's pre-training.
- **Clean factorization.** The method cleanly separates relevance modeling (LLM) from representation learning (encoder), making each component independently upgradable.
- **Practical lifecycle framing.** The conclusion explicitly positions HyDE as a bootstrap system: use it when no relevance labels exist, then gradually transition to a supervised retriever as search logs accumulate. This is honest and operationally useful.
- **Broad evaluation.** 11 datasets spanning web search, domain-specific retrieval, and 4 non-English languages. Not just one favorable benchmark.
- **Minimal hyperparameters.** Temperature (0.7 default) and instruction text are the only parameters. No tuning infrastructure required.

### Methodological Weaknesses

- **No multi-hypothesis experiments.** The formalism supports N>1 samples, but all experiments use N=1. This leaves the most obvious improvement (variance reduction via averaging) untested and the robustness to hallucination unmeasured.
- **No failure analysis.** The paper reports aggregate metrics without examining *which* queries HyDE helps vs. hurts. Are there query types where the hypothetical document systematically misleads? The ArguAna task (counter-arguments) is interesting -- HyDE works well -- but no per-query analysis explains why.
- **No latency analysis.** Generating a hypothetical document via InstructGPT adds significant latency (likely 1-3s at 2022 API speeds). For interactive retrieval, this is a material cost. The paper doesn't measure it.
- **No analysis of hallucination filtering.** The "lossy compressor" claim -- that the encoder filters hallucinated details -- is central but only argued theoretically. No experiment measures whether hallucinated entities or facts in generated documents lead to retrieving wrong documents.
- **Instruction sensitivity unexplored.** Each dataset gets a hand-crafted instruction. How sensitive are results to instruction wording? A robustness analysis (varying instructions) would strengthen the claim that HyDE generalizes.
- **December 2022 vintage.** InstructGPT (text-davinci-003) is the backbone. Modern instruction-following models are dramatically better, which makes all the numbers conservative -- but also means the exact results aren't reproducible as models are deprecated.

---

## Relevance to claude-memory

### What HyDE Does That We Don't

1. **Document-space query reformulation.** Somnigraph's `impl_recall()` embeds the raw query string via `embeddings.py` and searches against memory embeddings. HyDE proposes: generate what a relevant *memory* would look like, then embed *that*. This is the LLM-based expansion method the expansion-wip branch hasn't tried. The 6 implemented methods (entity focus, multi-query, keyword, session, entity bridge, Rocchio PRF) all manipulate the retrieval channels mechanically; HyDE uses LLM generation to bridge the semantic gap between query vocabulary and document vocabulary.

2. **Bridging the cue-trigger disconnect.** Somnigraph's roadmap Tier 1 #10 (prospective indexing) addresses the same problem from the write side: generate hypothetical future queries at `remember()` time. HyDE addresses it from the read side: generate a hypothetical answer at `recall()` time. These are complementary, not competing. Prospective indexing enriches the stored representation; HyDE enriches the query representation. Kumiho's prospective indexing eliminated the >6-month accuracy cliff on LoCoMo-Plus (37.5% to 84.4%); HyDE addresses the same semantic disconnect from the opposite direction.

3. **Task-adapted instructions.** Each retrieval task gets an instruction that shapes the *genre* of the generated document. Somnigraph's recall path doesn't differentiate by memory category at query time -- a query about a procedural memory gets the same embedding treatment as one about an episodic memory. HyDE suggests: instruct the LLM to generate a hypothetical document in the *style* of the expected memory category ("write a procedural memory about...", "write an episodic memory about..."). This could improve retrieval for category-specific queries.

### What We Already Do Better

1. **Learned ranking over raw retrieval.** HyDE is a retrieval-only method: it improves the query vector but has no scoring pipeline. Somnigraph's 26-feature LightGBM reranker learns non-linear interactions between retrieval signals, metadata, and graph structure. HyDE's contribution would be as one additional signal (a new retrieval channel or reranker feature), not as a replacement for the scoring pipeline.

2. **Feedback loop.** HyDE is stateless -- it generates a fresh hypothesis for every query with no learning from past retrievals. Somnigraph's feedback loop (per-query r=0.70 GT correlation) accumulates signal over time. HyDE can't learn that a particular memory was useful last time; the reranker can.

3. **Multi-channel fusion.** HyDE replaces the query vector entirely. Somnigraph's RRF fusion runs vector and FTS channels independently and fuses them. A HyDE-enhanced vector search would be one channel; FTS would still catch exact-match queries that HyDE's generated document might miss. The expansion-wip work already demonstrated that no single expansion method dominates -- multi-channel fusion is structurally superior.

4. **No LLM dependency at query time (current).** `recall()` currently requires only an embedding API call (~100ms). HyDE adds an LLM generation step (300-2000ms depending on model and length). For mid-conversation retrieval where latency matters, this is a real cost. Somnigraph's current design keeps LLM calls in the write path (`remember()`, sleep) and out of the read path.

---

## Worth Stealing (ranked)

### 1. HyDE as a Reranker Feature (High Value, Low Effort)

**What**: Instead of using HyDE to replace the query vector (the paper's approach), generate a hypothetical memory, embed it, and compute cosine similarity between the hypothetical embedding and each candidate's embedding. Feed this similarity score as a new feature to the LightGBM reranker. The reranker learns *when* the HyDE signal is useful (and when to ignore it).

**Why it matters**: The expansion-wip work showed that all 6 non-LLM expansion methods are neutral because the reranker can't elevate evidence from rank 100+. The bottleneck is *ranking*, not *coverage*. A HyDE similarity feature gives the reranker a fundamentally new signal: "how much does this candidate resemble an LLM's idea of what a good answer looks like?" This is orthogonal to FTS rank, vector distance, and graph signals.

**Implementation**: At `impl_recall()` time, after candidate retrieval: (1) prompt the conversation's LLM to generate a 2-3 sentence hypothetical memory answering the query, (2) embed it via `embed_text()`, (3) compute cosine similarity between the hypothetical embedding and each candidate's embedding, (4) add `hyde_sim` as feature #27 in `reranker.py`. For the LoCoMo benchmark, the generation step can use Haiku/Sonnet for cost efficiency. Key constraint: this feature requires an LLM call on the read path -- keep generation short (50-100 tokens) to bound latency.

**Effort**: Low. One LLM prompt, one embedding call, one cosine computation per candidate (vectorized), one new feature column. No schema changes. The feature can be disabled in production and enabled only for benchmarking until latency is acceptable.

### 2. Multi-Hypothesis Averaging for Ambiguous Queries (Medium Value, Low Effort)

**What**: For queries with high uncertainty (multiple possible interpretations), generate N=3-5 hypothetical documents with temperature>0 and average their embeddings. This creates a broader search cone that covers multiple interpretations.

**Why it matters**: Somnigraph queries often have implicit context that the system can't see. "What did we decide about the architecture?" could mean project architecture, memory architecture, or system architecture. A single hypothesis commits to one interpretation; averaging spreads the search across possibilities. The paper formalizes this but never tests it.

**Implementation**: If HyDE is implemented as a feature (idea #1), extend to N samples: generate N hypothetical documents, embed each, compute cosine similarity of each candidate to each hypothesis, take the max similarity as the feature value. Max (not mean) avoids the dilution problem where a correct hypothesis is averaged with incorrect ones.

**Effort**: Low incremental cost over idea #1. N additional embedding calls (cheap) but same number of LLM calls if all hypotheses are generated in a single prompt with few-shot examples.

### 3. Category-Adaptive Query Expansion (Medium Value, Medium Effort)

**What**: Use different HyDE prompts for different expected memory categories. When a query seems procedural ("how do I..."), generate a hypothetical procedural memory; when it seems episodic ("when did we..."), generate a hypothetical episodic memory. The generated document then shares vocabulary and structure with the target memory type.

**Why it matters**: Somnigraph stores 5 memory categories with different structures. A procedural memory ("To configure lc0, set these flags...") uses different vocabulary than an episodic memory ("We discussed lc0 configuration and decided..."). HyDE-style expansion could be category-conditioned to match the right register.

**Implementation**: Classify query intent (simple keyword/pattern matching, or the LLM call that generates the hypothesis can also classify), then select from a template bank:
- Procedural: "Write a step-by-step instruction that would answer: {query}"
- Episodic: "Write a brief account of an event that would be relevant to: {query}"
- Semantic: "Write a factual note that captures: {query}"
- Reflection: "Write a reflective observation about: {query}"

**Effort**: Medium. Requires query classification (simple but needs testing), template design, and evaluation of whether category-conditioning actually helps vs. a generic prompt. Could be a significant win for procedural queries, which have distinctive structure.

### 4. Prospective + Retrospective Symmetry (Medium Value, Medium Effort)

**What**: Combine write-side prospective indexing (roadmap Tier 1 #10) with read-side HyDE in a symmetric architecture. At write time, generate hypothetical future queries for each memory. At read time, generate a hypothetical memory for each query. Both enrichments live in the same embedding space, creating a double bridge across the cue-trigger disconnect.

**Why it matters**: Kumiho's prospective indexing results (37.5% to 84.4% on LoCoMo-Plus) suggest the cue-trigger disconnect is a dominant error mode. HyDE addresses the same gap from the other direction. The combination could be multiplicative: prospective indexing moves memories toward query-space; HyDE moves queries toward memory-space. If each closes half the gap independently, together they might close more.

**Implementation**: This is an experiment design, not an implementation. Run the LoCoMo benchmark four ways: (1) baseline, (2) prospective indexing only, (3) HyDE feature only, (4) both. Measure whether the improvements are additive, sub-additive, or super-additive. If sub-additive, they address the same gap and one suffices. If super-additive, the combination is worth the cost.

**Effort**: Medium. Requires implementing both #1 (HyDE feature) and prospective indexing (Tier 1 #10). The experiment itself is straightforward once both are implemented.

---

## Not Useful For Us

### Zero-Shot Framing

HyDE's primary value proposition is zero-shot retrieval without relevance labels. Somnigraph has 1,032 labeled queries with GT judgments, a trained reranker, and a live feedback loop. The zero-shot setting is irrelevant to our operational context. HyDE's *mechanism* (hypothetical document generation) is useful; its *motivation* (no labeled data) doesn't apply.

### Replacing the Query Encoder

The paper's architecture replaces `enc_q(q)` entirely with `f(g(q))`. In a system with multi-channel fusion and a learned reranker, replacing the query vector discards the FTS channel's ability to match exact terms (which HyDE's generated document may paraphrase away) and forces all ranking through a single LLM-generated representation. Using HyDE as an *additional* signal (feature) is superior to using it as a *replacement* for the query path.

### Task-Specific Instruction Engineering

The paper hand-crafts instructions per dataset type. In Somnigraph's context, all queries target the same corpus (personal memories), and the "task" doesn't change between queries in the way that web search vs. fact verification differs. Category-adaptive prompts (idea #3) are a lightweight version of this; full task-specific instruction engineering doesn't apply.

### InstructGPT as Generator

The backbone LLM (text-davinci-003) is deprecated. Any implementation would use a current model (Claude, GPT-4o-mini, Gemini Flash). The paper's specific results aren't reproducible, though the directional findings (larger models help more) likely transfer.

### Contriever as Encoder

Somnigraph uses OpenAI text-embedding-3-small (1536-dim), not Contriever (768-dim). The encoder choice changes the embedding space geometry and potentially the "lossy compressor" dynamics. The paper's specific encoder analysis doesn't transfer, though the general principle (unsupervised encoders benefit more from HyDE than fine-tuned ones) is informative.

---

## Impact on Implementation Priority

### Tier 1 #10: Prospective indexing -- Strengthened

HyDE provides the read-side complement to prospective indexing's write-side enrichment. Both address the cue-trigger semantic disconnect. The combination experiment (write-side + read-side) should be part of the prospective indexing evaluation. If HyDE alone closes much of the gap, prospective indexing's value is reduced (and vice versa). Testing both together resolves which direction of bridging matters more.

### Expansion-wip branch -- Modified

The 6 non-LLM expansion methods are neutral because the bottleneck is ranking, not coverage. HyDE offers a fundamentally different approach: instead of finding more candidates, it provides the reranker with a new signal to re-rank existing ones (as a feature, not a retrieval method). This reframes the expansion problem: the branch should consider `hyde_sim` as a reranker feature alongside the existing expansion-derived features (`entity_fts_rank`, `sub_query_hits`, `seed_keyword_overlap`).

### Reranker improvement experiments -- New consideration

The remaining P2 experiment (raw-score features) should include `hyde_sim` as a candidate feature. This is the first proposed feature that introduces an LLM call in the read path, so it needs a latency-vs-quality tradeoff analysis. For the LoCoMo benchmark (offline evaluation), latency doesn't matter; for production, it needs to be justified by a meaningful NDCG improvement.

### Overall retrieval architecture -- Unchanged

HyDE doesn't change the fundamental architecture (multi-channel RRF fusion + learned reranker). It's a potential new feature within the existing framework, not an architectural shift. The paper validates the existing design philosophy: multiple complementary signals fused by a learned ranker outperform any single retrieval method.

---

## Connections

### To Expansion-wip (this branch)

The 6 implemented expansion methods and HyDE represent two strategies for the same problem:

| Dimension | Non-LLM Expansion (expansion.py) | HyDE |
|-----------|----------------------------------|------|
| **Goal** | Increase candidate pool coverage | Improve query representation |
| **Mechanism** | FTS/vector sub-queries, entity bridging, Rocchio PRF | LLM generates hypothetical answer |
| **Bottleneck addressed** | Coverage (finds candidates at rank 100+) | Ranking (provides new signal for reranker) |
| **Cost per query** | ~0 (FTS + vector ops) | 1 LLM call + 1 embedding call |
| **Result** | Neutral -- evidence exists but reranker can't use it | Unknown -- needs testing as reranker feature |

The key lesson from expansion-wip applies to HyDE: **new signals must be new features, not overwrites.** Overwriting `fts_ranked` with entity-focused ranks degraded multi-hop by 18pp because the reranker's learned feature semantics were violated. A `hyde_sim` feature must be added as feature #27, not used to replace `vec_dist` or `vec_rank`.

### To HippoRAG (hipporag.md)

Both address the multi-hop retrieval problem, but from opposite directions. HippoRAG builds explicit entity-level cross-links (graph structure) to enable multi-hop traversal. HyDE generates a hypothetical document that implicitly captures multi-hop relevance patterns (the LLM "knows" the answer involves connecting multiple facts).

| Dimension | HippoRAG | HyDE |
|-----------|----------|------|
| **Multi-hop mechanism** | PPR over entity graph | LLM generates multi-hop answer |
| **Infrastructure** | Knowledge graph + entity extraction | Single LLM prompt |
| **Strengths** | Principled, interpretable, reusable | Zero infrastructure, immediate |
| **Weaknesses** | Heavy indexing (LLM per document), static graph | No guaranteed grounding, latency per query |
| **Somnigraph fit** | PPR already implemented (wm19), graph exists | Feature for reranker, no new infrastructure |

For Somnigraph, HippoRAG's contribution (PPR graph traversal) is already integrated. HyDE would add an orthogonal signal -- the LLM's implicit multi-hop reasoning captured in a hypothetical document -- without requiring the explicit entity graph that HippoRAG demands.

### To A-Mem (a-mem.md)

A-Mem's enriched embeddings (embedding over content + keywords + tags + contextual description) are conceptually the *write-side* version of HyDE's approach. Both enrich the text before embedding to capture relevance patterns beyond surface content. A-Mem enriches at write time (the memory gets a richer embedding); HyDE enriches at read time (the query gets a document-like embedding).

Somnigraph already does write-side enrichment: embeddings concatenate content + category + themes + summary (see `embeddings.py`). HyDE would add read-side enrichment. The combination -- enriched memory embeddings + enriched query embeddings -- means both endpoints of the similarity computation capture more than raw content.

### To Kumiho (kumiho.md)

Kumiho's prospective indexing (generating hypothetical future recall queries at write time) and HyDE's hypothetical document generation at read time are symmetric operations addressing the same cue-trigger semantic disconnect. Kumiho reports this eliminated the >6-month accuracy cliff (37.5% to 84.4% on LoCoMo-Plus). The question is whether read-side bridging (HyDE) achieves similar gains, or whether write-side bridging (prospective indexing) captures strictly more of the gap. The experiment in "Worth Stealing" #4 would answer this.

### To LoCoMo Audit (locomo-audit.md)

The LoCoMo benchmark has a 6.4% GT error rate and 93.57% theoretical ceiling. Any HyDE experiment on LoCoMo should use the corrected GT (already vendored) and dual-judging. Improvements below the noise floor (~2pp given the error rate) are not meaningful. HyDE's value should be measured on the corrected GT with confidence intervals.

---

## Summary Assessment

HyDE is an elegant, minimal paper with one good idea: instead of learning query-document similarity, use an LLM to generate a hypothetical document and search in document-document space. The idea is well-validated across 11 datasets and the factorization (generative LLM + contrastive encoder, independently upgradable) is clean. The paper's main limitation is analytical: it demonstrates that HyDE works but doesn't deeply explore *when* and *why* it fails, leaving the hallucination-filtering claim unsubstantiated beyond intuition.

**For Somnigraph specifically:**

- **Strongest takeaway**: HyDE's value is as a reranker feature (`hyde_sim`), not as a retrieval method. The expansion-wip branch proved that coverage isn't the bottleneck -- ranking is. A cosine similarity between candidates and a hypothetical memory gives the reranker a fundamentally new signal type (LLM-generated relevance pattern) orthogonal to existing retrieval and metadata features. This is the first proposed feature that adds an LLM call to the read path, making the latency-vs-quality tradeoff the central design question.

- **Second takeaway**: HyDE and prospective indexing (roadmap Tier 1 #10) are symmetric approaches to the same problem (cue-trigger disconnect). Testing both together will reveal whether one subsumes the other or they compound. If HyDE-as-feature provides meaningful NDCG gains, it may reduce the urgency of prospective indexing (which requires a write-path LLM call per memory).

- **Third takeaway**: The paper's generator scaling results (FLAN-T5 11B to InstructGPT 175B spans 12 NDCG points on DL19) suggest that for HyDE-as-feature, using a small fast model (Haiku, GPT-4o-mini) may sacrifice most of the benefit. The feature needs to be tested with the conversation's own LLM (which is already loaded) to avoid both latency and quality penalties.

- **Connection to expansion-wip**: The key lesson transfers directly -- overwriting existing reranker features with semantically different values degrades performance. `hyde_sim` must be a new feature, not a replacement for `vec_dist`. The expansion branch's negative results validate the reranker's sensitivity to feature semantics.

**Quality of the work**: Solid. ACL 2023, clean experimental design, broad evaluation. The idea is influential (>1,500 citations as of 2026) and has been widely adopted in RAG systems. The paper is unusually honest about its practical positioning (bootstrap system, not permanent solution). The main gap is analytical depth -- the "lossy compressor" claim needed empirical validation, not just theoretical argument. The lack of multi-hypothesis experiments and failure analysis leaves the most interesting questions unanswered.
