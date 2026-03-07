# LoCoMo Benchmark + Letta Filesystem Evaluation -- Source Analysis

*Phase 14, 2026-03-01. Analysis of Snap Research's LoCoMo benchmark and Letta's "Is a Filesystem All You Need?" evaluation.*

## 1. Architecture Overview

**LoCoMo (Long Conversation Memory)** is a benchmark for evaluating LLM agent memory across extended conversations. Published at ACL 2024 by researchers from UNC Chapel Hill, USC, and Snap Inc.

**Paper:** "Evaluating Very Long-Term Conversational Memory of LLM Agents" (Maharana et al., ACL 2024, arXiv:2402.17753)
**Code:** github.com/snap-research/locomo

**Dataset:**
- 10 conversations, ~300 turns each, ~9,000 tokens average (up to ~26K)
- Up to 35 sessions per conversation
- Generated via machine-human pipeline: LLM agents seeded with multi-sentence personas and causal, temporally organized event graphs (up to 25 events over 6-12 months)
- Multimodal: agents can share and react to images
- Human annotators verify long-range consistency

**Five question reasoning types:**
1. **Single-hop** — factual recall from a single session
2. **Multi-hop** — cross-session reasoning requiring information synthesis
3. **Temporal** — event sequence/ordering reasoning
4. **Open-domain** — external/world knowledge related to persona facts
5. **Adversarial** — misleading questions designed to induce hallucination

**Primary metric:** LLM-as-Judge accuracy (J score, binary: grade A = 1.0, else 0.0). Also F1, BLEU-1.

**Letta's filesystem evaluation** is a blog post benchmark ("Benchmarking AI Agent Memory: Is a Filesystem All You Need?") demonstrating that a GPT-4o mini agent using filesystem tools (search_files, grep, open/close, answer_question) achieved **74.0%** on LoCoMo — exceeding Mem0's knowledge graph at **68.5%**.

## 2. Memory Type Implementation

LoCoMo itself is a benchmark, not a memory system. The key memory approaches it evaluates:

**Letta filesystem agent:**
- Conversation histories stored as files, automatically parsed and embedded
- OpenAI text-embedding-3-large (1536-dim) for semantic search
- Chunking strategies tested: session-based, turn-based, time-window, SeCom (semantically coherent segments)
- Tools: `search_files` (vector search), `grep` (pattern matching), `open`/`close` (file reading)
- Constrained autonomy via tool rules: must start with search, can iterate, terminates with answer

**Mem0 (two variants):**
- **Mem0 base:** Dynamic memory extraction + consolidation. LLM identifies salient facts, compares to existing memories via vector similarity, decides ADD/UPDATE/DELETE/NOOP.
- **Mem0^g (graph-enhanced):** Adds directed labeled knowledge graph. Nodes = entities (with types, embeddings, metadata). Edges = relationship triplets. Dual retrieval: entity-centric graph traversal + semantic triplet matching.

## 3. Retrieval Mechanism

**Letta filesystem:**
- Agent-driven iterative search loop
- Vector search over file contents (primary)
- grep for pattern matching (secondary)
- Agent decides what to search for and when to stop
- No fusion algorithm — the agent itself integrates across search results

**Mem0 retrieval:**
- Entity-centric: identifies key entities in query, finds nodes by semantic similarity, explores relationships
- Semantic triplet: query encoded as embedding, matched against relationship triplet encodings
- Graph traversal for Mem0^g adds ~2.5x latency (p50: 0.476s vs 0.148s)

## 4. Standout Feature

**The full-context paradox.** The most striking finding: a simple full-context baseline (feeding the entire conversation into the LLM's context window) scores **~72.9%** on LoCoMo. This is:
- Higher than Mem0^g (68.4%) and Mem0 base (66.9%)
- Comparable to Letta filesystem (74.0%)
- Lower than top systems like MemMachine (84.9%) and Backboard (90.1%)

This means many "memory systems" score *lower* than simply not using a memory system at all. LoCoMo conversations are small enough (avg. 9K tokens) to fit in modern context windows, which undermines the benchmark's ability to test actual memory retrieval under resource constraints.

However, full-context has severe practical limitations: p95 latency 17.1s vs ~0.2s for RAG, 10x+ token cost, doesn't scale beyond context window limits.

## 5. Other Notable Features

1. **Benchmark controversy.** Every vendor achieves different scores for competitors depending on implementation choices. Zep critiqued Mem0's evaluation (incorrect user modeling, non-standard timestamps, sequential search inflating latency). Mem0 counter-rebutted Zep's claims (adversarial question handling, modified system prompts, single vs. 10 evaluation runs). Cross-paper comparisons are unreliable.

2. **Agent familiarity matters.** Letta's key argument: filesystem operations (grep, file search) are well-represented in LLM training data. Models perform better with tools they've "seen" during training than with novel memory APIs. "Memory is more about how agents manage context than the exact retrieval mechanism used."

3. **LoCoMo-Plus successor (arXiv:2602.10715, Feb 2026).** Tests "cognitive memory" — implicit constraints (user state, goals, values) not explicitly queried. Introduces "cue-trigger semantic disconnect" where retrieval cues have low lexical/semantic overlap with stored information. This is the harder, more realistic test.

4. **Leaderboard snapshot (as of 2026):**

| System | Overall (J) | Notes |
|--------|-------------|-------|
| Backboard | 90.1% | Gemini 2.5 Pro ingestion |
| MemMachine | 84.9% | Best balanced across types |
| Memobase v0.0.37 | 75.8% | Strong temporal (85.1%) |
| Zep (corrected) | 75.1% | Per Zep's own re-evaluation |
| Letta Filesystem | 74.0% | GPT-4o mini, filesystem tools |
| Full-context baseline | ~72.9% | No memory system at all |
| Mem0^g (graph) | 68.4% | Graph-enhanced |
| Mem0 (base) | 66.9% | Vector-only |

5. **Where graphs help.** Mem0^g outperforms Mem0 base on temporal reasoning (+2.6 points) and open-domain questions (+2.8 points). Graph structure helps with relational queries — but overall improvement is modest (~2 points) with 2-3x latency cost.

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | N/A | Benchmark, not a system |
| Multi-Angle Retrieval | N/A | Tests retrieval quality, doesn't implement it |
| Contradiction Detection | 10% | Adversarial questions test hallucination resistance but not contradiction detection |
| Relationship Edges | N/A | Knowledge graph variant tested but benchmark-agnostic |
| Sleep Process | 0% | No consolidation mechanism in benchmark |
| Reference Index | N/A | Benchmark |
| Temporal Trajectories | 30% | Temporal reasoning is one of five question types |
| Confidence/UQ | 0% | No confidence evaluation |

*Gap ratings are less applicable here since LoCoMo is a benchmark, not a memory system. Ratings reflect what the benchmark measures.*

## 7. Comparison with claude-memory

**What LoCoMo reveals about our system:**

Our system is neither pure filesystem nor pure knowledge graph — it's a hybrid (SQLite + FTS5 + sqlite-vec + curated summaries + relationship edges). This positions us between the approaches tested:

- **vs. Letta filesystem:** We share the "curated text + search" philosophy. Our summary+themes fields function like well-organized files. Our agent constructs keyword queries like Letta's grep. But we add vector search (which our Phase 14 experiment showed adds 0 MRR at our scale) and structured metadata.
- **vs. Mem0 KG:** We have relationship edges (memory_edges table) but traverse them during sleep consolidation, not during retrieval (yet). Our edge system is lighter than Mem0^g's full KG.
- **vs. Full-context baseline:** Our startup_load + recall pattern is a selective version of full-context — we inject the most relevant subset rather than everything. This gives us the latency and cost benefits of RAG without the retrieval failures of pure vector search.

**Our Phase 14 vector search result (FTS-only MRR = Vec-only MRR = Fused MRR ≈ 0.16) directly parallels LoCoMo's finding**: simple retrieval with good curation beats complex retrieval with poor organization.

## 8. Insights Worth Stealing

1. **Agent-familiar tool interfaces** (effort: low, impact: medium). Letta's strongest finding: models perform better with tools modeled after familiar operations (grep, file search) than novel APIs. Our `recall(query)` is already natural language, but we could add `recall(query, grep_mode=True)` for keyword-exact matching when the agent wants precision.

2. **Iterative search with stopping criteria** (effort: medium, impact: medium). Letta's agent decides when to stop searching. Our recall returns results in one shot. An iterative mode where the agent can say "search more" or "search differently" could improve multi-hop questions.

3. **Benchmark for our own system** (effort: medium, impact: high). LoCoMo-Plus's "cue-trigger semantic disconnect" tests exactly what we should worry about: can we retrieve information when the query doesn't share keywords with the stored memory? Our BM25 field-weighting on curated summaries partially addresses this (summary is human-curated, not verbatim), but we should test it.

## 9. What's Not Worth It

- **Knowledge graph complexity for modest gains.** Mem0^g's graph added ~2% overall accuracy over Mem0 base at 2.5x latency cost. At our scale (~300 memories), graph traversal during retrieval would add complexity without proportional benefit.
- **LoCoMo as our benchmark.** The conversations are too short (9K-26K tokens) and the questions test different capabilities than ours. LongMemEval (115K tokens) or a custom benchmark would be more relevant.
- **Agentic search loops for every query.** Letta's approach works because the agent runs multiple searches per question. Our recall is called within a session where the agent is already doing other work — we can't afford multi-turn retrieval loops for every recall.

## 10. Key Takeaway

LoCoMo's most important contribution to our project is the **full-context paradox**: elaborate memory systems often underperform a simple dump-everything-into-context baseline. The systems that beat full-context (Letta filesystem at 74%, MemMachine at 85%, Backboard at 90%) succeed through **organizational quality** (good chunking, curated summaries, structured extraction) rather than architectural complexity (knowledge graphs, multi-hop traversal).

This directly validates our Phase 14 experimental finding: at our scale with curated summary+themes fields, FTS5-only retrieval matches hybrid RRF. The curation is doing the work, not the retrieval mechanism. Letta's argument — "the quality of an agent's memory depends more on context management than on memory tools themselves" — is exactly what our data shows.

The qualifier: LoCoMo conversations are short enough to fit in context windows, so this finding may not generalize to larger memory stores. At 3,000+ memories with longer access intervals, vector search and graph traversal may become necessary. Our architecture retains these capabilities for when scale demands them.

## See Also

- [[agent-output-mem0-paper]] — Mem0's architecture and academic evaluation
- [[agent-output-longmemeval]] — Alternative benchmark (115K tokens, tests different memory abilities)
- [[retrospective-experiments]] — Phase 14 experiments confirming simple retrieval sufficiency at our scale
- [[agent-output-claudest]] — FTS5-only philosophy validated by this benchmark
