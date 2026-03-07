# MemOS: An Operating System for LLM Memory -- Analysis

*Generated 2026-02-18 by Opus agent reading 2505.22101v1*

---

## Paper Overview

- **Authors**: Zhiyu Li, Shichao Song, Hanyu Wang, Simin Niu, Ding Chen, et al. (19 authors)
- **Affiliations**: MemTensor (Shanghai) Technology Co., Shanghai Jiao Tong University, Renmin University of China, China Telecom Research Institute
- **Venue**: arXiv preprint, May 28, 2025 (short version)
- **Paper**: arXiv:2505.22101v1 [cs.CL]
- **Benchmark**: None. This is a system-design/vision paper with no empirical evaluation.

**Core problem**: LLMs treat memory as an afterthought. Parametric memory (model weights) is opaque and hard to update. Activation memory (KV-cache, hidden states) is ephemeral. RAG-based plaintext memory is an "ad hoc textual patch" without lifecycle management. The result is four critical gaps: (1) no long-term multi-turn state modeling, (2) poor knowledge evolution, (3) no persistent user/agent preference modeling, and (4) "memory silos" across platforms preventing reuse and migration.

**Key contribution**: A proposed *memory operating system* (MemOS) that elevates memory to a "first-class operational resource" for LLMs. The central abstraction is the **MemCube** -- a standardized container that wraps heterogeneous memory types (parametric patches, activation tensors, plaintext content) with unified metadata, enabling scheduling, lifecycle management, access control, and cross-type transformation. MemOS organizes around a three-layer architecture (Interface, Operation, Infrastructure) inspired by traditional OS design.

**Result**: No empirical results. The paper is a design document / architectural proposal. It presents the conceptual framework, data structures, and system modules, but does not report benchmarks, ablations, or deployment metrics. It is explicitly labeled "Short Version," suggesting a longer paper with evaluation is forthcoming.

**Why this paper matters for us**: MemOS is the most comprehensive attempt to formalize the OS metaphor for LLM memory that we have encountered. We already use a Letta/MemGPT-inspired register/RAM/disk analogy. MemOS extends this into a full system design with explicit scheduling policies, lifecycle state machines, access control, memory transformation pathways, and a marketplace concept. The gap between their vision and their evidence is large, but several architectural ideas are worth examining against our implementation priorities.

---

## Architecture / Method

### Three Memory Types

MemOS classifies all LLM memory into three types, each with distinct representation, lifecycle, and invocation semantics:

| Type | What It Is | Representation | Lifecycle | MemOS Role |
|------|-----------|----------------|-----------|------------|
| **Parametric** | Knowledge in model weights | Feedforward/attention layers, LoRA patches | Long-term, updated via fine-tuning | "Backbone" -- zero-shot capabilities, domain skill modules |
| **Activation** | Inference-time cognitive state | Hidden activations, attention weights, KV-cache | Ephemeral per-inference | "Working memory" -- context awareness, style control |
| **Plaintext** | Explicit external knowledge | Documents, knowledge graphs, prompt templates | Editable, shareable, governable | "Knowledge base" -- rapid updates, personalization, multi-agent collaboration |

**Transformation pathways** (Figure 3) are a key concept:
- *Plaintext -> Activation*: Frequently accessed text is pre-encoded into activation templates (reducing re-decoding cost)
- *Plaintext/Activation -> Parametric*: Stable, reusable knowledge is distilled into LoRA patches via fine-tuning
- *Parametric -> Plaintext*: Rarely used parameters are externalized back to editable text
- *Activation -> Plaintext*: Inference states are decoded/summarized into structured text

These transformations are triggered by behavioral indicators (access frequency, context relevance, version lineage), making the system a kind of memory promotion/demotion hierarchy analogous to CPU cache hierarchies.

### MemCube: The Core Abstraction

The MemCube is the fundamental unit of memory in MemOS. Every memory resource -- whether a LoRA patch, a KV-cache tensor, or a text document -- is wrapped in a MemCube with:

**Metadata Header** (three categories):
1. **Descriptive Metadata**: timestamps, origin signature (user input vs. inference output), semantic type (user preference, task prompt, domain knowledge), model association
2. **Governance Attributes**: access permissions (user/role lists), lifespan policies (TTL, frequency-based decay), priority level, sensitivity tags, watermarking, access logging
3. **Behavioral Indicators**: access frequency, context relevance scores, version lineage -- collected automatically at runtime, used to drive scheduling and cross-type transformation

**Memory Payload** (one of three types):
- Plaintext content: `{"type": "explicit", "format": "text", "content": "..."}`
- Activation state: `{"type": "activation", "format": "tensor", "injection_layer": 12, "value": "[tensor]"}`
- Parametric patch: `{"type": "parametric", "format": "lora_patch", "module": "mlp.6.down_proj", "value": "[low-rank-delta]"}`

The paper shows a concrete JSON-like schema (Figure 4) including fields for `created`, `last_used`, `source`, `model`, `usage` count, `priority`, `expires`, `access` list, `tags`, `embedding_fp`, and `storage_mode`.

### Three-Layer Architecture

**Interface Layer** -- Entry point for all memory operations:
- **MemReader**: Parses natural language into structured memory API calls
- **Memory API**: `Provenance API` (source annotation), `Update API` (content mutation), `LogQuery API` (usage trace queries)
- **Pipeline mechanism**: DAG-based operation chains (e.g., Query-Update-Archive) with transaction control. Each pipeline node passes context via MemCube.

**Operation Layer** -- Central controller:
- **MemScheduler**: Selects which memory to activate based on context. Supports pluggable strategies: LRU, semantic similarity, label-based matching. Operates at user-, task-, or organization-level.
- **MemLifecycle**: Models memory as a state machine with version rollback and freeze capabilities. Governs creation, activation, archival, deletion.
- **MemOperator**: Organizes memory via tagging, graph structures, and multi-layer partitions. Supports hybrid structural + semantic search. Frequently accessed entries are cached at an intermediate layer.

**Infrastructure Layer** -- Storage and governance:
- **MemGovernance**: Access permissions, lifecycle policies, audit trails, privacy protection, watermarking
- **MemVault**: Unified access across heterogeneous storage backends (Vector DB, Graph DB, etc.)
- **MemLoader/MemDumper**: Structured import/export for cross-platform migration
- **MemStore**: A marketplace for publishing and subscribing to memory units across models/agents

### Execution Flow (Figure 6)

1. User prompt arrives
2. MemReader parses it into structured Memory API call
3. Pipeline initiated, context passed via MemCube units
4. MemScheduler selects relevant memory (parametric, activation, or plaintext) based on access patterns and scheduling policies
5. Retrieved MemCubes injected into reasoning context
6. MemOperator organizes semantically/structurally
7. MemLifecycle governs state transitions
8. Archived memory persisted in MemVault
9. MemGovernance enforces compliance
10. MemStore enables inter-agent sharing via MemLoader/MemDumper

### Future Directions

The paper outlines three planned extensions:
1. **Cross-LLM Memory Sharing**: A Memory Interchange Protocol (MIP) for cross-model/app memory transmission
2. **Self-Evolving MemBlocks**: Memory units that self-optimize based on usage feedback
3. **Scalable Memory Marketplace**: Decentralized exchange for memory assets

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Memory should be a "first-class operational resource" | Conceptual argument; no benchmark | Philosophically sound but aspirational. The same argument has been made by Letta, Mem0, and others. |
| Three memory types (parametric, activation, plaintext) form a complete taxonomy | References to prior work on KV-cache, LoRA, RAG | Reasonable taxonomy for model-level memory. Does not address relational/graph memory as a distinct type. |
| MemCube unifies heterogeneous memory | Schema description, Figure 4 | The schema is concrete and well-specified. Whether a single abstraction can truly unify a LoRA delta and a text document is an open question. |
| Cross-type transformation pathways enable memory evolution | Conceptual description | Interesting but unvalidated. The Plaintext->Parametric path (distillation to LoRA) is technically feasible but expensive. The Activation->Plaintext path (decoding hidden states to text) is an active research problem. |
| Three-layer architecture provides closed-loop governance | Architecture diagrams | Standard layered architecture. The governance layer (access control, watermarking, audit) is more developed than most memory systems but not implemented/evaluated. |
| "Mem-training" is the next scaling paradigm after post-training | Figure 2 trend extrapolation | Speculative. No evidence provided beyond a conceptual trend line. |

**Overall evidence quality**: This is an architectural vision paper, not an empirical contribution. There are zero experiments, zero benchmarks, zero ablations. The paper's value lies entirely in its conceptual framework and system design. The "Short Version" label suggests evaluation is deferred to a longer paper.

---

## Relevance to claude-memory

### Direct Parallels

| MemOS Concept | claude-memory Equivalent | Status |
|---------------|-------------------------|--------|
| Three memory types | We operate exclusively in "plaintext memory" space (explicit text stored in Supabase). We have no parametric or activation memory levers. | N/A -- we can't modify Claude's weights or KV-cache |
| MemCube metadata (timestamps, usage count, priority, access) | Our memory schema has `created_at`, `last_accessed`, `access_count`, `priority`, `temperature` | Already implemented |
| Governance attributes (TTL, decay, access control) | Temperature-based decay, priority floors, category-based management | Partially implemented |
| Behavioral indicators driving scheduling | `access_count` and `temperature` influence `startup_load` and `recall` ranking | Partially implemented |
| MemScheduler (LRU, semantic similarity, label matching) | Hybrid search (pgvector + planned BM25), category filters, temperature-weighted ranking | In progress (RRF fusion is priority #1) |
| MemLifecycle state machine | We have soft-delete and temperature decay but no formal state machine with freeze/rollback | Gap |
| MemOperator (tagging, graph structure) | Tags/themes exist; relationship edges are priority #2 | Planned |
| MemStore (marketplace) | N/A -- single-user system | Not relevant |
| Cross-type transformation | N/A -- we cannot fine-tune Claude or inject KV-cache states | Not applicable |

### Key Differences

1. **Scope**: MemOS is designed for open-weight models where you control the full stack (weights, inference engine, KV-cache). claude-memory operates as a wrapper around a hosted API model. We will never have parametric or activation memory levers. Our entire design space is plaintext memory + prompt engineering.

2. **Multi-user/multi-agent**: MemOS heavily emphasizes access control, multi-user governance, watermarking, and cross-agent memory sharing. claude-memory is a single-user, single-agent system. The governance machinery is irrelevant to our use case.

3. **Evaluation**: MemOS provides no empirical validation. claude-memory has been in daily production use with iterative tuning based on real sessions. Our practical experience is more informative than their design document.

4. **Memory processing**: MemOS describes transformation pathways but not consolidation, reflection, or sleep-like processing. Our `/sleep` skill design (detail -> summary -> gestalt layers) addresses a gap that MemOS does not even identify as a problem.

---

## Worth Stealing (ranked)

### 1. Formal Lifecycle State Machine (Medium priority)

MemOS models memory lifecycle as an explicit state machine with states like active, frozen, archived, deleted -- plus version rollback. We currently have a simpler model (active/soft-deleted with temperature decay). A formal state machine would:
- Enable **freeze**: mark a memory as immutable (e.g., foundational identity memories that should not be consolidated or overwritten)
- Enable **rollback**: revert a memory to a previous version after a bad consolidation
- Make lifecycle transitions explicit and auditable

This connects to our planned **mutation log** (priority #6). A state machine + mutation log together give us full lifecycle auditability.

**Implementation sketch**: Add a `lifecycle_state` enum column (`active`, `frozen`, `archived`, `deleted`) to the memories table. Freeze prevents consolidation/edit. Archive removes from active retrieval but preserves for audit. Mutation log records all transitions.

### 2. MemCube-style Metadata Schema as Validation Checklist (Low priority, but useful)

The MemCube metadata categories (descriptive, governance, behavioral) provide a useful checklist for evaluating our own schema completeness:
- **Descriptive**: We have timestamps, source, category, themes. We lack `model` association (which model version created the memory) -- could be useful for detecting memories created by less capable models.
- **Governance**: We have priority and soft-delete. We lack explicit TTL/expiry dates. Temperature decay serves a similar function but is less explicit.
- **Behavioral**: We have `access_count` and `last_accessed`. We lack `context_relevance` scoring (how relevant was this memory when it was last retrieved?). This could inform better decay -- memories that are retrieved but rarely contextually useful should decay faster than memories that are retrieved and consistently useful.

**Concrete addition**: A `relevance_when_used` field that tracks not just "was this memory accessed" but "did it contribute to the response." This is hard to measure automatically but could be approximated by whether the memory was included in the final context window vs. filtered out during ranking.

### 3. Pipeline/DAG-based Operation Chains (Low priority)

The pipeline concept (Query-Update-Archive as a composable chain) is interesting for our `/sleep` skill. Sleep consolidation is inherently a multi-step pipeline: retrieve candidates -> cluster by theme -> generate summaries -> detect contradictions -> update edges -> archive details. Modeling this as a DAG with MemCube-like units flowing between stages would make the sleep process more modular and debuggable.

Not worth building pipeline infrastructure for, but worth keeping in mind as a design pattern when implementing `/sleep`.

### 4. Transformation Pathway Concept (Conceptual value only)

The idea that memory should flow between representation types based on usage patterns is elegant. We cannot do Plaintext->Parametric or Plaintext->Activation, but within our plaintext-only space we do have an analogous hierarchy:
- **Hot memories** (high temperature, frequently accessed) -> loaded into context via `startup_load` (analogous to "activation")
- **Warm memories** (moderate temperature) -> retrievable via `recall` (analogous to "plaintext")
- **Cold memories** (low temperature) -> archived, only surfaced by targeted search (analogous to "parametric" in the sense of being background knowledge)

Our temperature-based decay already implements a version of this promotion/demotion. The MemOS framing validates our approach but does not add new mechanism.

---

## Not Useful For Us

### 1. Parametric and Activation Memory Management
The entire parametric memory system (LoRA patch scheduling, module injection) and activation memory system (KV-cache manipulation, steering vectors, attention bias injection) are irrelevant. We use Claude via API. We cannot modify weights, inject activation tensors, or manipulate KV-cache. This is roughly 40% of the paper's conceptual contribution.

### 2. Multi-User Governance / Access Control
Watermarking, sensitivity tags, multi-user access permissions, compliance mechanisms -- all designed for enterprise/multi-tenant deployment. claude-memory is single-user. Not applicable.

### 3. Memory Marketplace (MemStore)
The vision of publishing and purchasing memory units in a decentralized marketplace. This is a business model, not a technical contribution. Not relevant to our use case.

### 4. Memory Interchange Protocol (MIP)
Cross-model memory sharing standard. We operate with a single model (Claude). Not relevant unless we wanted to migrate memories to a different LLM, which is not a current priority.

### 5. "Mem-training" Scaling Law Prediction
The claim that memory-centric training will be the next scaling paradigm (Figure 2) is speculative futurism with no supporting evidence. Not actionable.

### 6. MemReader (Natural Language -> Memory API)
The idea of parsing natural language to determine memory intent. We already handle this via explicit tool calls (`remember()`, `recall()`, `forget()`). Our approach is more reliable because the model explicitly decides when to invoke memory operations rather than having an intermediary try to parse intent from natural language.

---

## Impact on Implementation Priority

The MemOS paper does not change our implementation priorities. Here is the updated assessment:

| Priority | Item | MemOS Impact |
|----------|------|-------------|
| 1 | **RRF fusion** | No change. MemOS mentions "semantic similarity" and "label-based matching" as MemScheduler strategies but does not discuss RRF or hybrid retrieval. Our priority is validated by prior papers (Zep, Hindsight), not by MemOS. |
| 2 | **Relationship edges** | Slightly reinforced. MemOperator uses "graph-based structures" for organization, which aligns with our planned `memory_edges` table. But we already had strong motivation from every other system we analyzed. |
| 3 | **Graded contradiction detection** | No change. MemOS does not discuss contradiction detection at all -- a significant gap in their design. |
| 4 | **Decay floor + power-law** | No change. MemOS mentions "frequency-based decay" and "time-to-live" but provides no formulas or evaluation. Our existing designs from CortexGraph/Omega are more specific. |
| 5 | **Sleep skill** | Slightly reinforced conceptually. MemOS's transformation pathways (especially the consolidation direction) are thematically related to sleep, but they describe type-conversion (plaintext -> parametric) rather than content-consolidation (detail -> summary -> gestalt). Our sleep design remains more specific and practical. |
| 6 | **Mutation log** | Reinforced. MemOS's lifecycle state machine + version rollback concept strengthens the case for an append-only event log. Consider pairing mutation log with a formal lifecycle state machine (see "Worth Stealing" #1). |
| 7 | **Confidence scores** | No change. Not addressed by MemOS. |
| 8 | **Reference index** | No change. |
| 9 | **Multi-angle indexing** | No change. |

**Net impact**: Minor reinforcement of items #2, #5, #6. No new items added. No priority reordering warranted.

**One possible addition** (not to the main priority list, but to the "nice to have" backlog): **Lifecycle state machine** -- adding `frozen` and `archived` states alongside our existing active/deleted. This is a small schema change that pairs well with the mutation log work at priority #6. Worth bundling together when we get there.

---

## Connections

### To Prior Analyses

| Prior Paper | Connection to MemOS |
|-------------|-------------------|
| **Letta/MemGPT** | MemOS explicitly cites Letta [22] as a Stage 3 system that "implements paged context management and modular invocation" but "falls short of providing unified scheduling, lifecycle governance, and memory fusion." MemOS positions itself as the next evolution. The critique is fair but unsubstantiated -- MemOS has no empirical comparison. |
| **Mem0** | Cited [4] alongside EasyEdit as "toolkits supporting explicit memory manipulation." MemOS frames these as point solutions lacking OS-level integration. |
| **CogMem** | Not cited. CogMem's Oberauer-based three-tier model (FoA/DA/LTM) is a more rigorous cognitive grounding than MemOS's three memory types. MemOS's taxonomy is hardware-inspired (parametric/activation/plaintext); CogMem's is cognition-inspired (attention/access/long-term). For our purposes, CogMem's model maps more cleanly to our architecture. |
| **Zep** | Not cited. Zep's bi-temporal modeling and temporal trajectory tracking are more practically useful for knowledge evolution than MemOS's abstract lifecycle state machine. |
| **Generative Agents** | Not cited. The reflection/consolidation mechanism from Park et al. addresses the memory *processing* problem that MemOS ignores entirely. |
| **Continuum** | Not cited. CMA's 6 behavioral requirements provide a more concrete evaluation framework than MemOS's design principles. |
| **A-Mem** | Not cited. A-Mem's Zettelkasten linking and multi-faceted embedding address specific retrieval problems that MemOS gestures at ("multi-layer partitions") without specifying. |
| **Dynamic Cheatsheet** | Not cited. The finding that "summaries fail when replacing detail" directly challenges MemOS's implicit assumption that consolidation (plaintext -> parametric) is always beneficial. |

### Theoretical Positioning

MemOS positions itself within a three-stage evolution of LLM memory:
1. **Memory Definition and Exploration**: Taxonomies, knowledge editing, KV-cache optimization
2. **Human-like Memory**: Brain-inspired architectures (HippoRAG, Memory^3)
3. **Systematic Memory Management**: Tool-based operations + OS governance (EasyEdit, Mem0, Letta... and MemOS as the culmination)

This framing is useful for understanding where our project sits. claude-memory bridges Stages 2 and 3: we use human-inspired concepts (temperature = memory strength, consolidation = sleep, categories = memory types) but implement them as systematic tools (MCP server, Supabase backend, explicit API).

### Honest Assessment

MemOS is more vision than substance. The OS metaphor is pushed further here than anywhere else we have read, and some of the abstractions (MemCube metadata schema, lifecycle state machine, transformation pathways) are thoughtfully designed. But the complete absence of empirical validation makes it impossible to evaluate whether these abstractions actually work in practice. The paper reads like a product whitepaper for MemTensor's planned commercial offering rather than a research contribution.

The most useful papers in our research so far (Zep, CogMem, Generative Agents, Dynamic Cheatsheet) all provided empirical results that changed how we think about specific design decisions. MemOS provides a framework but no evidence. We can use it as a conceptual reference -- "have we thought about lifecycle states? access governance? transformation pathways?" -- but it does not give us new mechanisms to implement.

**Bottom line**: Read once, extract the lifecycle state machine idea, note the schema checklist value, and move on. This paper does not warrant revisiting.
