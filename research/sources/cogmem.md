# CogMem: Cognitive Memory Architecture — Analysis

*Generated 2026-02-18 by Opus agent reading 2512.14118v1*

---

## Paper Overview

- **Authors**: Yiran Zhang, Jincheng Hu, Mark Dras, Usman Naseem
- **Affiliations**: Macquarie University, Independent Researcher
- **Venue**: arXiv preprint, December 16, 2025
- **Paper**: arXiv:2512.14118v1 [cs.CL]
- **Benchmark**: TurnBench-MS (classic mode, 45 setups, 3 difficulty levels)

**Core problem**: LLMs maintain accuracy on single-turn reasoning but degrade across extended multi-turn interactions. Recurring failure modes identified via TurnBench: reasoning bias, task drift, hallucination, overconfidence, and memory decay. Current approaches concatenate full conversation histories, causing unbounded context growth, higher computational costs, and degraded reasoning efficiency.

**Key contribution**: A three-layer memory-augmented reasoning architecture inspired by Oberauer's (2002) model of working memory. The layers -- Long-Term Memory (LTM), Direct Access (DA), and Focus of Attention (FoA) -- interact through continuous cycles of retrieval, reconstruction, and refinement. The system tightly couples reasoning and memory via dedicated cooperating agents rather than treating them as loosely connected modules.

**Result**: On TurnBench-MS classic mode, the full CogMem system (FoA + DA + LTM) achieves 0.93 accuracy vs 0.76 baseline (Gemini 2.5 Flash), with perfect scores on easy and medium and 0.80 on hard. Token consumption stays bounded and flat, using less than half the baseline's tokens after 15 turns.

**Why this paper matters for us**: Our broader-landscape.md identified CogMem as "closest to our architecture in spirit." The paper's Focus of Attention mechanism directly parallels our `startup_load()` + `recall()` pattern: construct the right context dynamically from stored memories rather than loading everything. The Oberauer model provides formal cognitive grounding for the tiered memory design we've been building intuitively.

---

## Architecture / Method

### The Oberauer Model (Theoretical Foundation)

CogMem maps Oberauer's (2002) three-level model of working memory:

| Oberauer Level | CogMem Layer | Function | Persistence |
|----------------|-------------|----------|-------------|
| **Long-term memory (activated portion)** | Long-Term Memory (LTM) | Cross-session reasoning strategies, reusable patterns | Persistent across sessions, stored in vector DB (Milvus) |
| **Direct Access region** | Direct Access (DA) | Session-level notes: intermediate conclusions, sub-goals, plans | Session-scoped, stored in RAM (optional Redis) |
| **Focus of Attention** | Focus of Attention (FoA) | Dynamically reconstructed minimal context for current turn | Per-turn, ephemeral |

Oberauer's key insight: working memory is not a single buffer. The focus of attention holds only one or a few items at a time; the direct-access region holds a broader set of readily available items; and long-term memory holds everything else. Retrieval is the process of moving items between these tiers. CogMem operationalizes this with distinct storage, access patterns, and lifecycle management for each tier.

**Note on broader-landscape.md correction**: The broader-landscape.md described CogMem's tiers as "Sensory Memory / Short-Term Memory / Long-Term Memory." The paper itself uses "Focus of Attention / Direct Access / Long-Term Memory" -- the Oberauer model, not the Atkinson-Shiffrin model. The distinction matters: Oberauer's model emphasizes the *dynamic reconstruction* of attention from working memory, while Atkinson-Shiffrin is a more static pipeline. CogMem's FoA is not a buffer that passively receives input; it's an active reconstruction mechanism.

### Two Cooperative Agents

The system operates via two agents coordinated by a session manager and memory manager:

| Agent | Model | Role |
|-------|-------|------|
| **Reasoning Agent** | Gemini 2.5 Flash | Performs inference over constructed context windows |
| **Memory Agent** | Gemini 2.5 Flash Lite | Summarization, retrieval, note generation (generative reasoning disabled) |

This separation is deliberate: the memory agent is lighter and cheaper, handling the bookkeeping tasks without the overhead of full reasoning capabilities. The reasoning agent focuses exclusively on the problem at hand, working from the context the memory agent constructs.

### Long-Term Memory (LTM)

- Stores distilled reasoning strategies and reusable problem-solving patterns accumulated across sessions
- Resides in Milvus (vector database) for semantic similarity retrieval
- Updated at session end: the LTM reviews the completed session and decides whether to merge, add, or exclude distilled knowledge
- At session start, the memory agent queries LTM to populate the DA with relevant long-term memories

**Lifecycle**: Write happens at session conclusion (selective refinement). Read happens at session initialization and on-demand during reasoning. Delete/update happens during cleanup of inactive sessions. This is an explicitly offline write cycle -- new knowledge enters LTM only after the session is complete and the system can evaluate what was genuinely useful.

### Direct Access Memory (DA)

- Session-level working memory: concise notes of intermediate conclusions, sub-goals, ongoing plans
- Created empty at session start, populated from LTM and updated asynchronously after each reasoning step
- Two components:
  - **Notepad**: Plans, conclusions, intermediate reasoning products
  - **Message Summary History**: Compressed summaries of each turn (turn identifier + summary)
- Stored in RAM for fast access, optionally backed by Redis
- The key innovation: DA stores *notes about* the conversation, not the conversation itself

### Focus of Attention (FoA)

The most interesting mechanism. FoA dynamically reconstructs the minimal reasoning context at each turn through a two-phase process:

**First Context Window**:
- System message
- Session notes from DA (notepad + message summary history with turn identifiers)
- Current user message
- Any activated LTM items retrieved by the memory agent

The reasoning agent evaluates this context. If it determines the context is sufficient, it proceeds to inference. If not, it returns identifiers of specific previous turns it needs to review.

**Second Context Window**:
- Everything from the first context window
- Plus: detailed content of the requested turn(s)
- The reasoning agent performs inference over this refined context

This two-phase approach is the FoA in action: start with summaries, drill into details only where needed. The agent actively decides what it needs, rather than passively receiving a pre-determined context. This is retrieval-as-context-construction, not retrieval-as-lookup.

### Processing Workflow (Per Turn)

```
1. User input arrives
2. Session manager checks: reusable session? inheritable session? new session?
   - Reusable: restore stored results and memory states
   - Inheritable: prior dialogue forms prefix of current input; restore and continue
   - New: create empty DA, query LTM for relevant memories
3. Construct First Context Window (FoA)
4. Reasoning agent evaluates: context sufficient?
   - Yes: proceed to inference
   - No: return turn identifiers for missing context
5. If needed: construct Second Context Window with requested turn details
6. Reasoning agent performs inference
7. Memory agent summarizes the exchange
8. DA updated asynchronously (user gets immediate response)
9. Turn record cached for reuse
```

### Session Lifecycle

- **Session inheritance**: If current input is a continuation of a prior session, the system detects this and restores the prior DA and memory states rather than starting fresh
- **Cleanup**: Inactive sessions trigger garbage collection. Expired sessions and DA entries are deleted. FoA turn data use reference counting for turn-level reuse.
- **LTM refinement at session end**: Before removing a session, LTM reviews it to determine whether distilled knowledge should be merged, added, or excluded. Only meaningful reasoning traces persist.

### Implementation

- Modular, compatible with OpenAI SDK, works with any base LLM
- Vector storage: Milvus
- Short-term data: RAM (optional Redis)
- Event-triggered garbage collection for consistent latency regardless of session length

---

## Key Claims & Evidence

### Claim 1: Additive benefit from each memory layer

| Configuration | Total Accuracy | Easy | Medium | Hard |
|--------------|---------------|------|--------|------|
| Baseline (Gemini 2.5 Flash) | 0.76 | 0.87 | 0.93 | 0.47 |
| Baseline + FoA | 0.76 | 0.93 | 0.84 | 0.53 |
| Baseline + FoA + DA | 0.84 | 0.93 | 0.93 | 0.66 |
| **Baseline + FoA + DA + LTM** | **0.93** | **1.00** | **1.00** | **0.80** |
| Random Guess | 0.0085 | 0.0079 | 0.0098 | 0.0077 |

**Assessment**: The ablation is clean and well-structured. Each layer adds incrementally:
- FoA alone: minimal impact on total (0.76 -> 0.76) but shifts distribution (easy up, medium down). The context reconstruction helps on easier problems but may lose needed detail on medium ones.
- DA adds structured note-keeping: big jump (+0.08 total), especially on hard (+0.13). Within-session consistency matters most when problems are difficult.
- LTM adds cross-session learning: another big jump (+0.09 total), hard improves from 0.66 to 0.80. The model learns generalizable strategies across runs.

**Caveat**: The FoA-only result is puzzling -- medium drops from 0.93 to 0.84. This suggests that FoA's context compression can discard useful information for medium-difficulty problems. The paper doesn't address this regression. The FoA mechanism is only clearly beneficial when paired with DA (which preserves the information FoA might discard).

### Claim 2: Token consumption stays bounded

The paper reports (Appendix C, Figure 2) that CogMem maintains "a bounded and much flatter growth curve" compared to the baseline's near-linear increase. After 15 turns, CogMem uses "less than half the tokens" of the baseline.

**Assessment**: This is the expected consequence of any system that summarizes rather than concatenates. The claim is valid but not novel -- any RAG or summarization-based approach achieves bounded token growth. What CogMem adds is that the bounded growth comes with *improved* accuracy, not degraded accuracy. That's the meaningful claim: you can have both efficiency and correctness.

### Claim 3: Tight reasoning-memory coupling mitigates failure modes

The paper claims CogMem addresses: reasoning bias, task misconception, hallucination, overconfidence, memory decay, and unbounded context growth.

**Assessment**: The paper provides no per-failure-mode breakdown. The overall accuracy improvement is demonstrated, but attributing specific gains to specific failure mode mitigation is speculative. The paper would be stronger with error analysis showing which failure modes each layer addresses.

### Limitations Acknowledged

- All experiments on TurnBench-MS with a single base model (Gemini 2.5 Flash)
- Classic mode only (no nightmare mode evaluation)
- Generalizability to other datasets, task types, and models is unvalidated

### Limitations Not Acknowledged (My Assessment)

1. **No comparison with existing memory systems.** CogMem is compared only against the bare baseline (no memory). No comparison with MemGPT/Letta, Mem0, A-Mem, Mem1, or any other memory-augmented system. This is a significant omission -- we can't tell how CogMem compares to the systems it claims to improve upon.

2. **No formulas provided for retrieval, scoring, promotion/demotion.** The paper is entirely qualitative about its mechanisms. No similarity threshold, no retrieval scoring function, no criteria for when LTM decides to merge/add/exclude. The "how" is described in natural language only. This limits reproducibility.

3. **Single benchmark, single model.** TurnBench-MS classic mode with Gemini 2.5 Flash only. No cross-benchmark validation (LongMemEval, LOCOMO, etc.). No cross-model validation (GPT-4, Claude, etc.).

4. **FoA-only regression on medium unexplained.** FoA alone drops medium accuracy from 0.93 to 0.84, which contradicts the claim that FoA universally improves reasoning. This is a meaningful failure mode (context compression discarding useful detail) that deserves investigation.

5. **The paper is 4 pages of content + 4 pages of appendix/references.** The method section is quite brief. Many implementation details are deferred to "the internal code repository (link omitted for anonymization)."

6. **No evaluation of the LTM learning curve.** The paper doesn't show how many sessions are needed before LTM provides meaningful benefit. The cross-session claim is central but the learning dynamics are not characterized.

7. **Nightmare mode results absent.** TurnBench-MS nightmare mode (18% baseline accuracy) is mentioned in the introduction but not evaluated. This is where multi-turn reasoning difficulties are most severe and where CogMem's contribution should be most visible.

---

## Relevance to claude-memory

### Architectural Parallels

CogMem's three tiers map surprisingly well to our existing system:

| CogMem Layer | claude-memory Equivalent | Alignment |
|-------------|------------------------|-----------|
| **LTM** (cross-session strategies in vector DB) | `memories` table in Supabase (pgvector) | Strong. Both store persistent cross-session knowledge with semantic retrieval. |
| **DA** (session-level notes, plans, conclusions) | Session context + `startup_load()` results | Moderate. We don't have a structured DA per se -- our session context is the conversation itself plus loaded memories. |
| **FoA** (dynamically reconstructed minimal context) | `startup_load()` + `recall()` + token-budgeted retrieval | Strong. Both reconstruct context dynamically rather than loading everything. |

**Key difference**: CogMem's DA is a *structured note-taking layer* that the memory agent maintains asynchronously. Our system doesn't have this. The conversation itself serves as implicit "notes," but there's no separate, compressed representation of "what we've concluded so far" that persists within a session. This is the most interesting architectural gap.

**Key similarity**: CogMem's FoA -- the two-phase context construction where the reasoning agent can request specific turn details -- is functionally equivalent to our `recall()` pattern. The agent evaluates whether it has enough context and requests more if needed. The difference is that CogMem formalizes this as an architectural mechanism (two context windows), while we rely on the agent's judgment about when to call `recall()`.

### Where CogMem Aligns With What We've Already Built

1. **Dynamic context construction over static loading.** CogMem's FoA constructs minimal context per turn. Our `startup_load()` loads high-priority memories at session start, and `recall()` adds topic-specific context on demand. Same principle: construct what you need, not everything you have.

2. **Separation of reasoning and memory concerns.** CogMem uses two dedicated agents. We separate the roles differently (the agent + MCP server), but the principle is the same: the reasoning model shouldn't also be doing bookkeeping.

3. **Session lifecycle management.** CogMem's session manager handles inheritance, caching, expiration, and garbage collection. Our session model is simpler (each Claude Code session is independent), but `startup_load()` serves the same purpose as CogMem's session initialization: bringing relevant context forward.

4. **LTM as selective refinement, not raw storage.** CogMem's LTM only stores "distilled reasoning strategies" after session review. Our `remember()` is already selective (the agent/user chooses what to store), and `/sleep` is designed to further refine. Same philosophy: LTM should contain understanding, not transcripts.

### Where CogMem Highlights Gaps in Our System

1. **No explicit session-level note-taking layer.** CogMem's DA maintains structured notes (plans, conclusions, intermediate results) that persist within a session independently of the conversation history. Our system relies on the conversation context itself. When context gets long or the session resets, those intermediate conclusions are lost unless explicitly `remember()`-ed.

2. **No formalized two-phase retrieval.** CogMem's First Context Window / Second Context Window pattern -- start with summaries, drill into details if needed -- is an elegant mechanism we implement informally but don't enforce architecturally. Our `recall()` returns memories at full detail; we don't have a "summary first, detail on demand" retrieval mode.

3. **No asynchronous memory processing.** CogMem's memory agent updates DA *asynchronously* after the reasoning agent responds, so the user gets an immediate answer while memory is updated in the background. Our `remember()` calls are synchronous -- the agent decides to store something, calls the tool, waits for completion. This is fine for explicit storage but means we can't do lightweight automatic summarization after each exchange.

4. **No session inheritance detection.** CogMem detects when a new conversation is a continuation of a prior one and automatically restores the relevant DA state. Our system relies on `startup_load()` for general context and `recall()` for specific context, but doesn't detect continuity between sessions. This is partially compensated by our priority/temperature system (recent relevant memories are naturally loaded), but it's not the same as explicit continuation detection.

---

## Worth Stealing (ranked)

### 1. Summary-First / Detail-On-Demand Retrieval Pattern -- LOW effort, HIGH value

CogMem's two-phase FoA (First Context Window with summaries, Second Context Window with details) is a pattern we could implement in `recall()`. Currently `recall()` returns full memory content. An alternative: return compact summaries first, with memory IDs. If the agent needs more detail, it calls a `recall_detail(id)` function for the full content.

**Why**: Reduces noise in recall results. The agent sees an overview of what's available before deciding what to load fully. This is especially valuable when recall returns many results -- currently all compete for token budget at full detail. With summaries first, the agent can be more selective.

**Effort**: Low. The `summary` field already exists on memories. A new `recall_summaries()` function that returns `{id, summary, category, priority}` tuples, plus a `recall_detail(id)` for expansion, would implement this pattern. The two-phase behavior is then up to the calling agent.

### 2. The Oberauer Vocabulary for Our Tier Structure -- ZERO effort, HIGH value

CogMem's use of Oberauer's terminology (Focus of Attention / Direct Access / Long-Term Memory) provides a cleaner vocabulary than our current informal descriptions. Adopting this vocabulary:

| Our Current Language | Oberauer Equivalent |
|---------------------|-------------------|
| "startup_load context" | Focus of Attention (session-start FoA) |
| "recall results" | Focus of Attention (on-demand FoA) |
| "loaded memories" + session context | Direct Access region |
| "memories table" | Long-Term Memory |
| "summary/gestalt layers" | Promotion from LTM to compressed LTM |

**Why**: Better vocabulary enables clearer thinking. "Dynamically reconstructing the Focus of Attention" is a more precise description of what `startup_load()` does than "loading high-priority memories." The vocabulary also helps communicate what we're building to others and connects our design to established cognitive science.

**Effort**: Zero implementation effort. It's a naming and framing change for documentation and design discussions.

### 3. Session-Level Notepad Concept -- MEDIUM effort, MEDIUM-HIGH value

CogMem's DA includes a "notepad" that maintains plans and conclusions separately from the conversation. This could be implemented as a lightweight within-session memory layer: key conclusions, plans, and decisions are extracted and stored in a compressed format that survives context-window resets within a session.

**Why**: Currently, if a Claude Code session runs long and context is compressed, intermediate conclusions may be lost. A session notepad would preserve the most important within-session state. This is distinct from `remember()` (which is for cross-session persistence) -- the notepad is for within-session coherence.

**Implementation**: Could be as simple as a new MCP tool `session_note(content)` that appends to a session-scoped list, and a `session_notes()` reader that returns all notes for the current session. Notes would be automatically included in every `recall()` result. Cleared at session end (optionally promoted to memories via `/sleep`-like review).

**Effort**: Medium. Requires session identification (which we already track), a session-scoped storage mechanism, and integration into the recall flow.

### 4. Asynchronous Memory Summarization -- MEDIUM effort, MEDIUM value

CogMem's memory agent updates DA asynchronously after each reasoning turn. We could implement a lightweight version: after each `remember()` call, trigger a background process that checks the new memory against existing ones for relationships, contradictions, and grouping.

**Why**: Catches relationships at ingestion time rather than deferring everything to `/sleep`. This is the same point Zep's paper raised -- contradiction detection should happen at write time for fast feedback, with deeper analysis deferred to consolidation.

**Effort**: Medium. Requires background processing capability (which MCP doesn't natively support well). Could be approximated with a `remember()` post-hook that runs lightweight checks before returning.

### 5. Turn Summary History as a Compression Pattern -- LOW effort, LOW-MEDIUM value

CogMem's DA maintains a "Message Summary History" -- each turn is compressed to a summary with a turn identifier. When the reasoning agent needs a specific turn's details, it requests by identifier. This is a pattern for managing long conversations without full-history concatenation.

**Why**: Relevant for our system when sessions grow long. Rather than the model seeing the full conversation, it could see turn summaries with the option to expand specific turns. This is more of a prompt-engineering pattern than an infrastructure change.

**Effort**: Low. Could be implemented as a session-management prompt pattern: at each turn, summarize the previous exchange and add to a running summary list. The model sees summaries + current turn rather than full history. This doesn't require MCP changes -- it's a usage pattern.

---

## Not Useful For Us

### Dedicated Reasoning/Memory Agent Split

CogMem uses two separate LLM instances: a full-capability reasoning agent and a lightweight memory agent. For our system, the single Claude instance handles both reasoning and memory operations via MCP tool calls. Splitting into two models would add infrastructure complexity without clear benefit at our scale. The MCP architecture already provides the separation of concerns (memory operations are tool calls, not part of the reasoning prompt).

### TurnBench-MS Benchmark

TurnBench-MS evaluates iterative rule-discovery games -- a specific multi-turn reasoning task format. This doesn't test the personal memory, cross-session continuity, and long-horizon identity that claude-memory targets. It's a good benchmark for CogMem's specific claims but not relevant for evaluating our system. LongMemEval and LOCOMO remain our benchmark references.

### Session Inheritance Detection

CogMem's session manager detects when a new conversation continues a prior one by checking if the current input forms a prefix of a previous session's dialogue. This is specific to their multi-turn game format where conversations have deterministic structure. Our sessions don't have this property -- each Claude Code session starts fresh with potentially unrelated tasks. Our `startup_load()` and temperature-based memory surfacing handle the same problem (bringing forward relevant context) more generally.

### Milvus-Specific Implementation

CogMem uses Milvus for vector storage. Our Supabase + pgvector stack is already established and sufficient. No reason to change infrastructure.

### Event-Triggered Garbage Collection

CogMem's session-level garbage collection with reference counting for turn-level reuse is an optimization for their specific session management model. Our system is simpler: memories persist indefinitely, decay handles relevance over time, and `/sleep` handles consolidation. We don't need turn-level reference counting.

---

## Impact on Implementation Priority

### Current priority list (from index.md):

1. RRF fusion
2. Relationship edges
3. Graded contradiction detection
4. Decay floor + power-law
5. Sleep skill
6. Mutation log
7. Confidence scores
8. Reference index
9. Multi-angle indexing

### Assessment: No reordering needed.

CogMem validates the overall direction but doesn't provide specific implementation details that would change our priorities. The paper's contributions are architectural and conceptual rather than algorithmic. Specific adjustments:

**Add to recall design (informing #1)**: Consider implementing a two-phase recall: `recall_summaries()` returns compact overview, `recall_detail(id)` returns full content. This doesn't change the priority of RRF fusion but adds a retrieval UX pattern that would complement it.

**Strengthens the case for #5 (Sleep skill)**: CogMem's LTM refinement at session end ("LTM reviews the session to determine whether distilled knowledge should be merged, added, or excluded") is exactly what our `/sleep` Step 5 does. CogMem validates that selective refinement should happen at session boundaries, not continuously.

**New optional item after #9 -- #10: Session notepad**: A within-session note-taking layer (plans, conclusions, key decisions) that persists through context compression. Not a priority for the memory system itself, but a complementary tool that would improve within-session coherence. Low effort if implemented as a simple session-scoped list.

**Vocabulary update for design docs**: Adopt Oberauer terminology (FoA / DA / LTM) in our architecture documentation. This is a documentation change, not an implementation priority.

---

## Connections

### To Other Analyses in This Project

**[[agent-output-generative-agents]]**: CogMem and Generative Agents both implement tiered memory with dynamic context construction, but differ fundamentally in where intelligence lives:
- Generative Agents: intelligence is in the *retrieval formula* (recency x importance x relevance) and the *reflection trigger* (importance sum > 150). The memory system makes decisions about what to surface and when to synthesize.
- CogMem: intelligence is in the *reasoning agent's evaluation* of whether context is sufficient. The FoA asks the agent "do you have what you need?" rather than algorithmically deciding what to provide.

For claude-memory, we're closer to Generative Agents' approach (our retrieval scoring decides what to surface) but CogMem's pattern of letting the agent request more is also present in our `recall()` design. The hybrid is natural: algorithmic scoring for initial retrieval (like Generative Agents), with on-demand expansion for specific needs (like CogMem's Second Context Window).

**[[agent-output-continuum]]**: CMA defines six behavioral requirements. CogMem, evaluated against these:
- Persistence: PASS (LTM persists across sessions)
- Selective retention: PARTIAL (LTM refines at session end, but no active forgetting)
- Retrieval-driven mutation: FAIL (no evidence that retrieval changes future accessibility)
- Associative routing: FAIL (retrieval is semantic similarity only, no graph traversal)
- Temporal continuity: PARTIAL (session inheritance provides some temporal linking)
- Consolidation: PASS (LTM distillation at session end qualifies)

CogMem satisfies 2/6 CMA requirements fully. By CMA's definition, it's "heavily engineered RAG" -- which is ironic given the cognitive-science framing. This underscores that CogMem's contribution is in context management, not in memory lifecycle.

**[[agent-output-zep-paper]]**: Zep and CogMem address different dimensions of the memory problem:
- Zep: deep on *what to store* (bi-temporal facts, entity resolution, contradiction detection) and *how to retrieve* (triple search + RRF)
- CogMem: deep on *how to construct context* (FoA reconstruction, two-phase windows) and *when to store* (session-end LTM refinement)

For claude-memory, Zep gives us storage and retrieval mechanisms; CogMem gives us context construction patterns. Both are useful on different axes.

**[[agent-output-hindsight]]**: Hindsight's MPFP (Multi-Perspective Fact Profiling) and CogMem's FoA both address context quality -- ensuring the model sees the right information. Hindsight does this at the memory level (profile each memory from multiple angles), CogMem at the retrieval level (construct context in phases). They're complementary: MPFP improves what's stored, FoA improves what's surfaced.

**[[sleep-skill]]**: CogMem's LTM refinement at session end ("determine whether distilled knowledge should be merged, added, or excluded") maps directly to our sleep Step 5 (Build Layers) and Step 6 (Prune/Demote). CogMem validates the decision to make consolidation a discrete event at session boundaries rather than a continuous background process. However, CogMem's consolidation is simpler than our sleep design -- it doesn't build summary/gestalt layers, detect trajectories, or resolve contradictions with a graded model. It's binary: merge, add, or exclude.

**[[broader-landscape]]**: The broader-landscape entry for CogMem needs correction:
- The tiers are FoA / DA / LTM (Oberauer model), not Sensory / STM / LTM (Atkinson-Shiffrin)
- The Focus of Attention is a dynamic reconstruction mechanism, not a raw input buffer
- The Direct Access memory is session-level structured notes, not "current reasoning chains"

### To Prior Phase Findings

**Phase 6 "separation of concerns" principle** (from [[systems-analysis]]): CogMem's two-agent architecture validates the principle that memory management and reasoning should be separate responsibilities. Our MCP design already embodies this: the memory server handles storage/retrieval, the Claude session handles reasoning. CogMem adds the insight that the memory agent should be *lighter* than the reasoning agent -- it doesn't need full reasoning capability, just summarization and retrieval.

**Phase 5 "context is the bottleneck"** (from [[self-critique]]): CogMem directly addresses this finding. The FoA mechanism is explicitly designed to keep context minimal and relevant. The token consumption results (less than half the baseline after 15 turns) quantify the savings. Our `startup_load()` token budget and `recall()` token budget are implementations of the same principle.

---

## Summary Assessment

CogMem is an architecturally interesting paper with clean cognitive-science grounding (Oberauer model) and solid ablation results (0.76 -> 0.93 on TurnBench-MS). Its primary contribution is the Focus of Attention mechanism -- dynamic, two-phase context reconstruction where the reasoning agent actively evaluates and requests what it needs. The paper is limited by a narrow evaluation (single benchmark, single model, no comparison with existing memory systems) and lack of formal specification (no formulas for retrieval scoring, promotion/demotion, or consolidation criteria).

**For claude-memory**: The highest-value takeaways are conceptual rather than implementational:
1. The Oberauer vocabulary (FoA / DA / LTM) as a framing for our architecture
2. The two-phase retrieval pattern (summaries first, details on demand)
3. The session notepad concept (structured within-session notes separate from conversation)

CogMem provides no formulas, no thresholds, and no algorithms we can directly implement. Compared to Zep (which gave us bi-temporal fields and temporal extraction prompts), Hindsight (which gave us RRF and MPFP), or Engram (which gave us graded contradiction detection), CogMem's contribution is framing and validation rather than mechanism. It confirms we're on the right track and gives us better language for describing what we're building.

**Net impact**: Framing and vocabulary, not implementation. Adds a two-phase recall pattern as a design consideration and a session notepad as an optional future feature. Validates our existing architecture's alignment with established cognitive models. Does not change the implementation priority ordering.
