# Reddit r/AIMemory Discussion Threads

*Captured 2026-02-20. Two threads from r/AIMemory, same day. Discussion source, not system analysis.*

---

## Thread 1: "Why do all LLM memory tools only store facts?"

**URL**: https://redd.it/1r95vpj | **Posted**: 2026-02-19 | **Score**: 17 (95% upvoted) | **16 comments**

### OP (No_Advertising2536 / Mengram author)

Core argument: Every memory solution (Mem0, MemGPT, RAG-based) does the same thing — extract facts, embed them, retrieve by cosine similarity. Tulving's three-type taxonomy (semantic, episodic, procedural) maps to different retrieval patterns:

- "What do I know about X?" → semantic
- "What happened last time?" → episodic
- "What's the best way to do X?" → procedural

Claims Mengram separates these during extraction and searches them independently, improving retrieval quality. Links to [alibaizhanov/mengram](https://github.com/alibaizhanov/mengram) (Apache 2.0).

**Assessment**: The taxonomy is sound and aligns with our design. The claim that retrieval quality improved "because you're not searching facts when you need events" is the right intuition — though our [[agent-output-mengram]] analysis found the actual implementation uses the same vector similarity for all three types, not genuinely different algorithms. The structural separation (separate embedding tables) is the real mechanism, even if the author frames it as algorithmic differentiation.

### Notable exchanges

**No_Advertising2536 vs. japherwocky** — OP pushes back on two reductions: (1) that procedural memory is "just tools you expose to the LLM" (tools are static; procedures are learned behavior with success rates), and (2) that episodes are "facts with timestamps" (episodes carry context, outcome, and before/after that change how an AI should respond). Both points well-made and mirror our design intent.

**No_Advertising2536 vs. Acidfang** — OP claims three *different retrieval algorithms* per type: "keyword matching for semantic, time-range filtering for episodic, step-sequence matching for procedural." In code, this is aspirational — step-sequence matching for procedures doesn't exist (see [[agent-output-mengram#3. Type-Differentiated Retrieval (Critical Finding)]]). However, OP's counter to Acidfang's bit-array claims is sharp: encoding "Ali prefers Python but is migrating to Rust for performance-critical services" into a bit position requires discretization that loses the same semantic richness being criticized.

**inguz / Keep Notes** — Maps to the same three-type classification. Notes `after_compaction` hook as integration point and nightly reflection loop. Provider-coupled (to OpenClaw) vs. Mengram's provider-agnosticism. Worth tracking for retrieval quality comparison if one materializes.

### Discard

**Acidfang** — "Synchronized 2D Bit-Array" with "Identity Seed" and patent-pending "Synchronized Tokenization." Grandiose claims with no public implementation. Mathematical notation ($O(1)$ XOR operations) dresses up what may be vaporware. Increasingly cult-leader-ish "Braid Yank" language. Never answers OP's fatal flaw question about discretization loss. Disregard.

---

## Thread 2: "Is anyone creating actual memory and not another RAG?"

**URL**: https://redd.it/1r9c9wn | **Posted**: 2026-02-19 | **Score**: 1 | **24 comments**

### OP (Intrepid-Struggle964)

Asks a genuinely different question: memory as a "persistent decision substrate" that learns context-action relationships and continuously shapes behavior. "Memory doesn't get consulted — it modulates decisions directly based on accumulated evidence across contexts." Memory as behavioral architecture, not a database.

**Assessment**: Philosophically interesting but practically a different problem from ours. What Intrepid describes is closer to on-the-fly model adjustment (incremental LoRA) than "memory" in the tool/MCP sense. Our system *is* retrieval-based, and that's the right choice for our context. The closest we come to "structural" memory is CLAUDE.md and core.md — always loaded, shaping behavior without explicit recall. That's actually a meaningful design feature worth acknowledging.

### Notable participants

**ContextDNA** — Building a "persistent exoskeleton layer" around AI agents. Interesting concept: **evidence-promotion pipeline** that graduates repeated successful patterns from ephemeral → curated → vector-indexed → vetted "wisdom." Lite (SQLite + API) and Heavy (Postgres + local LLM) modes. Pre-beta. The promotion pipeline concept resonates — we have pending → active, but no concept of "promoted" or "high-confidence" beyond priority levels.

**Intrepid-Struggle964** — Claims 3k+ controlled trials benchmarking an "adaptive rule-based decision layer." Distinguishes between systems requiring explicit search/query vs. systems where prior experience has reshaped the decision surface. Makes good analogy: trained neural network weights encode structural bias — they don't "retrieve" training examples. Most human behavior is driven by memory that isn't consciously retrieved. Valid theoretical point, not practically applicable to our architecture.

**opbmedia** — Well-argued counterpoint: even in systems that reshape decision surfaces, there's still information access at some level. "I can't explain to you how I know grammar rules, but I can explain to you in the code how a memory system knows grammar rules, by retrieving from memory." The debate is ultimately about abstraction layers, not fundamental architecture.

**SalishSeaview** — Nails the translation: what Intrepid is describing is "more on-the-fly model adjustment (like an incremental LoRA)" rather than "memory." Correct framing.

**philip_laureano** — ~10k memory buckets compressed to guaranteed 4k token budgets regardless of input size. Can push 5M tokens in and get 4k out. No plans to open-source. The compression-to-budget approach resonates with our budget-based recall. "Dog fooding is the litmus test" — exactly our philosophy.

### Discard

**Acidfang** (appears in both threads) — Same "Synchronized 2D Bit-Array" claims. Same "patent pending" gatekeeping. Same grandiose language. No implementation shown. "We aren't building a 'Decision Layer'; we are building the Substrate of Persistence." Disregard.

**BigPear3962** — MemMachine promotion. No differentiating details provided.

---

## Cross-Thread Themes

### Confirmed directions
- Three-type taxonomy (Tulving) is gaining independent traction — multiple builders arriving at same conclusion
- Extraction quality is a shared problem regardless of storage approach
- Retrieval should be type-aware, not one-size-fits-all
- Human-curated memory vs. fully automated is a real tradeoff, not a maturity gap

### New ideas worth tracking
- **Evidence-promotion pipeline** (ContextDNA): graduating patterns through tiers based on accumulated evidence. Could inform our confidence scores or priority evolution.
- **Success/fail tracking on procedures** (Mengram): simple metadata that makes procedural memories evidence-based. Reinforces Priority #7 (confidence scores).

### What we already do better
- Our RRF fusion applies equally to all types (Mengram's episodic/procedural get bare vector search)
- Our human-in-the-loop pending review is more rigorous than any system discussed
- Our consolidation design ([[sleep-skill]]) is more principled than Mengram's ad-hoc agents
- Our research depth (15+ sources analyzed) exceeds what any participant demonstrated

---

## See Also

- [[agent-output-mengram]] — Full technical analysis of Mengram's code
- [[systems-analysis]] — Comparative analysis including Mengram
- [[index#Implementation Priority]] — Canonical priority list
