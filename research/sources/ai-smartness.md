# AI Smartness — Source Analysis

*Phase 13, 2026-02-27. Analysis of VzKtS/ai-smartness.*

## 1. Architecture Overview

**Language:** Rust 77.8%, JavaScript/TypeScript (VS Code extension), HTML
**Key dependencies:**
- ONNX runtime (local embeddings)
- SQLite (persistence)
- rusqlite (Rust SQLite bindings)
- Tauri (desktop GUI framework)
- Claude Code (primary integration target via hooks)

**Storage layout:**
```
Per-agent isolated SQLite databases
  threads          # Neurons — active reasoning streams
  bridges          # Synapses — semantic connections between threads
  shared_threads   # Published knowledge for inter-agent access
  subscriptions    # Agent-to-agent knowledge imports
  messages         # Inter-agent communication
  agent_registry   # Heartbeat-based agent discovery
```

**File structure:** Cargo workspace with 11 crates:
```
src/
  intelligence/
    gossip.rs            # Gossip propagation algorithm
    decayer.rs           # Exponential decay with orphan acceleration
    engram_retriever.rs  # 9-validator multi-signal retrieval
    memory_retriever.rs  # Simple text search fallback
    archiver.rs          # LLM-synthesized archives
    merge_evaluator.rs   # Thread merge conflict resolution
    synthesis.rs         # Information synthesis
    thread_manager.rs    # Thread lifecycle
    reactivation_decider.rs
    validators/          # Individual validator implementations
  storage/              # SQLite persistence
  processing/           # Text processing, embeddings, coherence
  mcp/                  # MCP stdio JSON-RPC server
  daemon/               # Background service (IPC, periodic tasks)
  hook/                 # Claude Code hooks (inject/capture)
  cli/                  # Command-line interface
  guardcode/            # Content rules, injection formatting
  healthguard/          # Proactive cognitive monitoring
  registry/             # Multi-agent discovery
  gui/                  # Tauri desktop UI
  bridge.rs             # ThinkBridge struct and types
  thread.rs             # Thread struct
  agent.rs              # Agent implementation
  config.rs             # Configuration
  session.rs            # Session management
```

## 2. Memory Type Implementation

**Schema:** Core memory unit is a `Thread` (the "neuron"):
- Auto-generated semantic title
- Content/body text
- Weight (0.0-1.0, subject to decay)
- Importance (0.0-1.0, influences decay half-life)
- Status (active/suspended/archived)
- `last_active`, `last_injected_at` timestamps
- Topics, concepts, labels

Threads are connected by `ThinkBridge` ("synapse") records:
```rust
pub struct ThinkBridge {
    id: String,
    source_id: String,
    target_id: String,
    relation_type: BridgeType,  // Extends, Contradicts, Depends, Replaces, ChildOf, Sibling
    reason: String,
    shared_concepts: Vec<String>,
    weight: f64,
    confidence: f64,
    status: BridgeStatus,  // Active, Weak, Invalid
    propagated_from: Option<String>,
    propagation_depth: u32,
    created_by: String,
    use_count: u32,
    created_at: DateTime<Utc>,
    last_reinforced: Option<DateTime<Utc>>,
}
```

**Types/categories:** No fixed memory type taxonomy. Threads are categorized organically via topics, concepts, and labels. The `BridgeType` enum (Extends, Contradicts, Depends, Replaces, ChildOf, Sibling) provides relationship semantics.

**Extraction:** Via Claude Code hooks — captures are processed by the daemon, which uses LLM-driven decisions to determine whether content merits a new thread or extends an existing one.

## 3. Retrieval Mechanism

**Two retrieval paths:**

**Simple path** (`MemoryRetriever`): Text search via `ThreadStorage::search()`.

**Full pipeline** (`EngramRetriever`): Three-phase multi-validator system:

1. **Hash index pre-filtering** — TopicIndex and ConceptIndex provide O(1) candidate narrowing before expensive validation.

2. **9-validator scoring** — Each candidate evaluated by:
   - V1: SemanticSimilarityValidator (embedding-based, only expensive one)
   - V2: TopicOverlap (zero-cost)
   - V3: TemporalProximity (zero-cost)
   - V4: GraphConnectivity (zero-cost)
   - V5: InjectionHistory (zero-cost)
   - V6: DecayedRelevance (zero-cost)
   - V7: LabelCoherence (zero-cost)
   - V8: FocusAlignment (zero-cost)
   - V9: ConceptCoherence (zero-cost)

3. **Consensus ranking:**
   - StrongInject (>=5/9 pass): top-of-context, full content
   - WeakInject (3-4/9 pass): bottom-of-context, condensed
   - Skip (<3/9 pass): excluded

**Context injection**: 5-layer priority system ("Engram-like" format) with layered placement based on relevance, recency, and user-defined rules.

This is notably different from RRF hybrid search. Instead of fusing ranked lists, it uses a voting/quorum model where 8 of 9 signals are free (memory lookups) and only the embedding comparison costs compute.

## 4. Standout Feature

**9-validator quorum retrieval with 8/9 zero-cost signals.** This is the most architecturally novel element. Rather than the standard approach of computing multiple similarity scores and fusing ranked lists (RRF, weighted sum), ai-smartness treats retrieval as a voting problem. Each validator independently votes pass/fail with confidence, and a quorum threshold determines injection placement. The key insight: by making 8 of 9 validators pure memory lookups (topic overlap, temporal proximity, graph connectivity, injection history, decay state, label coherence, focus alignment, concept coherence), the system gets multi-signal robustness with near-zero marginal cost per candidate. Only the embedding comparison (V1) requires compute.

## 5. Other Notable Features

- **Gossip propagation for transitive bridges**: If A-B and B-C bridges exist, the gossip algorithm synthesizes A-C with `weight = min(AB, BC) * decay_factor` and depth tracking. Configurable max depth and per-thread connection limits prevent runaway propagation. Bridges are tagged `created_by: "gossip_v2"` for traceability.

- **Orphan-accelerated decay**: Threads have importance-scaled half-lives (0.75-7.0 days), but threads not recently injected into context experience accelerated decay via a separate orphan factor: `orphan_factor = 0.5^(orphan_hours / halving_hours)`. This creates a "use it or lose it" pressure distinct from time-based decay alone. Auto-suspension at weight < 0.1.

- **Multi-agent isolation with gossip knowledge sharing**: Each agent gets an isolated database. SharedThreads use copy-on-share semantics with pull-based subscription model. Inter-agent messages discovered organically during memory injection (Layer 2.5) with zero polling. The `Contradicts` bridge type enables cross-agent disagreement tracking.

## 6. Gap Ratings

| Gap | Rating | Evidence |
|-----|--------|----------|
| Layered Memory | 55% | Threads have weight/importance tiers and context injection layers (strong/weak/skip). LLM-synthesized archives for aged content. No explicit STM/LTM distinction, but functional layering through decay + injection thresholds. |
| Multi-Angle Retrieval | 80% | 9-validator ensemble with topic, temporal, graph, injection history, decay, label, focus, concept, and embedding signals. Quorum-based fusion rather than RRF. |
| Contradiction Detection | 45% | `BridgeType::Contradicts` exists as a first-class relationship type. Bridges track contradiction at the connection level. No automated detection — relies on LLM classification during bridge creation. |
| Relationship Edges | 75% | ThinkBridge is the core architectural element. Six typed relationships (Extends, Contradicts, Depends, Replaces, ChildOf, Sibling). Weight, confidence, status, use_count, reinforcement timestamps. Gossip propagation creates transitive edges. |
| Sleep Process | 40% | Daemon auto-compacts at 0.80 context pressure. LLM-synthesized archives preserve content before removal. Orphan cleanup every 5 minutes. Not a true sleep/consolidation cycle, but continuous background maintenance. |
| Reference Index | 55% | TopicIndex and ConceptIndex with O(1) hash lookups. Shared_concepts on bridges. Auto-generated semantic titles. No explicit topic hierarchy or taxonomy. |
| Temporal Trajectories | 30% | Temporal proximity is a validator signal. Injection history tracked. But no explicit trajectory analysis, evolution tracking, or temporal clustering. |
| Confidence/UQ | 40% | Confidence field on bridges (0.0-1.0). Weight on threads subject to decay. Validator confidence scores in retrieval. No explicit uncertainty quantification or human feedback loop on confidence. |

## 7. Comparison with claude-memory

**Stronger:**
- Multi-validator quorum retrieval is architecturally more sophisticated than RRF hybrid search, with better cost profile (8/9 signals free)
- Gossip propagation creates transitive edges automatically — our adjacency expansion reads existing edges but does not synthesize new ones
- Orphan-accelerated decay is a more nuanced model than simple time-based decay — penalizes memories not actively used in context
- Multi-agent architecture with isolated databases and shared cognition is a capability we lack entirely
- Typed relationships (6 types) with bidirectional tracking and use counts are richer than our edge schema
- Hebbian reinforcement on injection (bridges strengthened when used) creates a feedback loop we approximate via recall_feedback but less directly
- ONNX local embeddings avoid per-call API costs

**Weaker:**
- No human-in-the-loop curation or feedback mechanism equivalent to recall_feedback
- No gestalt/summary layer for high-level context
- No startup_load equivalent — context injection is continuous rather than front-loaded
- Confidence is on bridges only, not on individual memories (we have per-memory confidence gradient)
- Linux-only tested (macOS/Windows "untested") vs. our cross-platform Windows operation
- No sleep consolidation in the sense of batch processing with reflection — maintenance is continuous background
- Complexity: 11 Rust crates is a significant maintenance burden vs. our single-file server
- The roadmap lists significant incomplete features (Dynamic Quota Engine, provider portability, remote GUI) — the system is ambitious but in-progress

## 8. Insights Worth Stealing

1. **Quorum-based retrieval with zero-cost validators** (effort: high, impact: high). The 9-validator voting model is a genuinely different approach to multi-signal retrieval. Rather than computing multiple scored lists and fusing them (RRF), you run cheap boolean checks (topic overlap? recent? graph-connected? recently injected?) and only invoke the expensive embedding comparison for candidates that pass enough cheap filters. This could be adapted into our recall pipeline as a pre-filter stage: before running the full RRF, check 4-5 zero-cost signals and skip candidates that fail all of them.

2. **Orphan-accelerated decay** (effort: low, impact: medium). Tracking `last_injected_at` separately from `last_active` and applying extra decay to memories never surfaced in recall is a clean mechanism for pruning memories that exist but never get used. We already have shadow_load as a penalty; orphan decay would be complementary — shadow penalizes at query time, orphan decay penalizes at the storage level over time.

3. **Gossip-propagated transitive edges** (effort: medium, impact: medium). Our adjacency expansion follows existing edges during recall but never creates new ones. The gossip algorithm's approach — if A-B and B-C exist, synthesize A-C with attenuated weight — could run during sleep as an edge synthesis step. The depth-limited propagation with `created_by: "gossip_v2"` tagging keeps provenance clean.

## 9. What's Not Worth It

- The multi-agent architecture. For a single-user system like ours, the overhead of agent isolation, shared threads, subscriptions, and inter-agent messaging adds complexity without benefit.
- The full Engram injection pipeline with 5 layers and strong/weak/skip placement. Our startup_load + on-demand recall is simpler and works well with human-in-the-loop curation. Continuous injection without human gatekeeping risks context pollution.
- The Tauri GUI and VS Code extension. Different product surface; not relevant to CLI-first architecture.
- The 11-crate Rust workspace. The engineering overhead is enormous for what is fundamentally a memory system. Single-file Python with SQLite is more maintainable for our use case.

## 10. Key Takeaway

AI Smartness is the most architecturally ambitious system in this batch and one of the more sophisticated in the entire survey. The neuron/synapse metaphor is not just branding — it manifests in real mechanisms: threads as processing units with decay and importance, bridges as typed connections with gossip propagation, and a multi-validator retrieval pipeline that treats injection as a voting problem rather than a ranking problem. The 9-validator quorum model with 8/9 zero-cost signals is the single most novel retrieval mechanism encountered in this research phase. The gossip propagation algorithm for synthesizing transitive edges is also worth studying. The main limitation is that the system optimizes for autonomous multi-agent operation rather than human-in-the-loop curation, which means it lacks the feedback loops and gestalt summarization that make our system effective for a single user working collaboratively with their AI. The Rust implementation is impressive engineering but creates a maintenance burden disproportionate to the problem space.
