# Agent Workflow Memory -- Analysis

*Generated 2026-02-19 by Opus agent reading 2409.07429v1*

---

## Paper Overview

**Paper**: Wang, Mao, Fried, Neubig (2024). Carnegie Mellon University + MIT. arXiv:2409.07429v1, Sep 11, 2024.

**Problem addressed**: LLM-based agents solving multi-step digital tasks (web navigation) struggle with long-horizon action sequences and fail to learn from past successes. Each task is solved from scratch -- agents don't extract reusable routines from their own experience trajectories. Humans, by contrast, abstract common task patterns and compose increasingly complex procedures from simpler ones.

**Core claim**: Agent Workflow Memory (AWM) induces reusable "workflows" -- abstracted sub-routines with parameterized slots -- from agent trajectories, and injects them into agent memory to guide future actions. This yields +24.6% relative step success on Mind2Web and +51.1% relative task success on WebArena. Online AWM (no training data, pure test-time learning) beats even human-engineered workflow baselines (SteP) by 7.6% relative. Workflows compose hierarchically: simple workflows become subgoals for more complex ones, creating a snowball effect.

**Scale**: 16 pages. Two benchmarks: WebArena (812 tasks, 5 websites, 4 domains) and Mind2Web (1000+ tasks, 200+ domains). Two modes: offline (induce from training examples) and online (streaming test-time induction). Ablations on representation format, environment abstraction, and action space expansion. Open-source at github.com/zorazrw/agent-workflow-memory.

---

## Architecture / Method

### Workflow Representation

Each workflow has two components:

1. **Description `d`**: NL summary of the workflow's goal (e.g., "Find a place by its name on OpenStreetMap"). Heuristically extracted from experience instructions or LM-summarized.

2. **Trajectory `(p1, p2, ...)`**: A series of steps, each containing:
   - **Environment description**: NL description of the current state (e.g., "Order {id} is shown")
   - **Reasoning**: Agent's thought process for action selection
   - **Action**: Executable program over the environment (e.g., `click('147')`, `fill('145', {location})`)

Critical design: example-specific values are abstracted into parameterized slots. "dry cat food" becomes `{product-name}`, "New York" becomes `{location}`. This is done via explicit prompting during induction -- the prompt instructs the LM to "represent non-fixed elements with descriptive variable names."

### Workflow Induction

**Formal definition**: Induction module `I` takes experience set `E = {e_i}` where each `e = (q, P^e)` is a task instruction and action trajectory. Produces workflows: `I(E) -> W = {w} = {(d_j, P^d_j)}`.

**LM-based induction** (primary method): The agent is prompted to extract "common sub-routines" from one or more experiences. Prompt instructions explicitly request:
- Finer granularity than full tasks (sub-task level)
- No similar/overlapping workflows
- At least two steps per workflow
- Variable names for non-fixed elements

The induction prompt is shown verbatim in Appendix A.1:
> "Given a list of web navigation tasks, your task is to extract the common workflows. Each given task contains a natural language instruction, and a series of actions to solve the task. You need to find the repetitive subset of actions across multiple tasks, and extract each of them out as a workflow."

**Rule-based induction** (ablation baseline): Extract action type sequences (CLICK -> TYPE -> CLICK), deduplicate by sequence, filter invalid actions. Performs comparably on WebArena (35.6% vs 35.5%) but worse on Mind2Web (43.4% vs 45.1% step SR). LM-based produces finer-grained, more abstract workflows.

### Memory Integration

Workflows are added to agent memory: `M + W -> M_w`. At action generation, the agent uses `L(q, M_w, o) -> a`. There is no retrieval mechanism described -- all workflows for a website are concatenated into the prompt context. The memory operates per-website: examples are grouped by website, workflows induced and applied within each group.

### Two Operating Modes

**Offline** (`AWM_offline`):
1. Concatenate all training examples for a website into a single prompt
2. LM induces workflows: `I(E_train) -> W_offline`
3. Apply fixed `W_offline` to all test examples

**Online** (`AWM_online`):
1. Process test queries in a streaming fashion
2. For test `q_t`, generate trajectory `{p_t}`
3. LM-based evaluator judges success: `L_eval(e_t) in {0, 1}`
4. If success: induce workflows `I(e_t) -> {w_t}`, add to memory `M^t + {w_t} -> M^{t+1}`
5. Next test uses expanded memory
6. No training data needed

**Hybrid** (`AWM_off+on`): Tested in Appendix C. Offline workflows warm-start, then online workflows accumulate. Result: *worse* than either alone. Offline and online workflows "are not fully compatible" -- offline workflows "impair the generative quality and utility efficacy of online workflows."

### Workflow as Action Space (AWMAS)

Alternative utilization: wrap each workflow into a callable high-level function (e.g., `login(username, password)`). Agent can call workflow actions alongside primitives. Calling a workflow action triggers the full pre-determined step sequence. Result: marginal gain (+1.3 step SR on Mind2Web). Agents only called workflow actions in 18.5% of tasks -- resistance to using newly-added actions. Key limitation: workflow actions execute without seeing intermediate state, so dynamic pop-ups or branching environments break them (Figure 7).

### Workflow Hierarchy / Composition

AWM builds increasingly complex workflows over time. Example from WebArena maps: "Find a place by its name" (early, simple) becomes a subgoal within "Get the zip code of a place" (later, complex). The complex workflow adopts the first few steps from the simpler one and adds new steps. This is emergent from the induction process -- not a designed composition mechanism.

---

## Key Claims & Evidence

### 1. AWM achieves SOTA on WebArena (+51.1% relative)

AWM: 35.5% overall SR vs. BrowserGym baseline: 23.5% (+12.0 absolute). Also beats SteP (human-written workflows): 35.5% vs 33.0%. Gains across all 5 websites (11.8-30.7 absolute points over BrowserGym_ax-tree). Uses fewer steps: 5.9 vs 7.9 (BrowserGym) vs 46.7 (AutoEval). All with GPT-4 (gpt-4-0613), temperature 0.0.

**Strength of evidence**: Moderate. Single model, single temperature. No error bars or confidence intervals reported. WebArena evaluation is execution-based (rigorous), not proxy-based.

### 2. Online AWM achieves rapid learning from small data

Learning curve (Figure 5): most essential workflows acquired in first ~40 examples. Gap over baseline widens to 22.5 points after 40 examples on the maps split, then stabilizes. Two phases: "rapid learning phase" (0-40 examples) and "stable inference phase" (40+).

**Strength of evidence**: Shown only on the maps split (smallest, most homogeneous). Generality of the learning curve to other splits not demonstrated.

### 3. Cross-template generalization holds

On a deduplicated WebArena subset (one example per template): AWM 33.2% vs BrowserGym 20.5% vs SteP 32.1%. Workflows generalize across different task templates, not just within-template.

**Strength of evidence**: Solid design -- explicitly controls for within-template leakage. But the 33.2% is slightly below the full 35.5%, suggesting some within-template benefit exists.

### 4. Online AWM generalizes better as domain gaps widen

On Mind2Web: as train-test gap widens (cross-task -> cross-website -> cross-domain), AWM_online's margin over AWM_offline grows. Cross-domain: AWM_online 35.5% step SR vs AWM_offline 32.6% (+2.9). AWM_online doesn't depend on training distribution, so it's immune to domain shift.

**Strength of evidence**: Consistent pattern across three generalization splits. The absolute numbers are still low (task SR: 1.7% cross-domain), reflecting benchmark difficulty.

### 5. Abstract sub-routines outperform concrete examples

AWM (+5.0 element accuracy) vs Synapse (retrieves full examples as context). Abstract workflows with parameterized slots introduce less bias on element selection than full concrete examples.

**Strength of evidence**: Head-to-head comparison on Mind2Web cross-task. Controlled for element filtering method. Compelling for the claim that abstraction helps.

### 6. Offline+online combination *hurts*

AWM_off+on scores between offline and online alone across all three Mind2Web splits. Not additive. Offline workflows impair online workflow quality. Task SR drops dramatically (cross-task: 1.6% vs 4.0% online-only or 4.8% offline-only).

**Strength of evidence**: Surprising and underexplained. The paper notes it but doesn't analyze deeply. Suggests workflow interference -- a form of negative transfer in memory.

### 7. Workflow quality metrics are favorable

Per-website: ~7.3-7.4 workflows induced. 0.08 functional overlap on WebArena (low redundancy). 0.94 utility rate (94% of test examples use at least one workflow). 0.40 coverage on Mind2Web (40% of trajectory steps covered by workflows). Quality appears high, though these are self-assessed against the model's own predictions.

---

## Relevance to claude-memory

### Procedural Memory Design (Direct)

AWM is fundamentally about **procedural memory** -- extracting reusable action sequences from experience. Our `procedural` category serves the same function but stores calibration patterns, corrections, and work heuristics rather than action trajectories. The key parallel:

- AWM workflow description `d` = our procedural memory's `content` (the "what and when to apply" text)
- AWM workflow trajectory `(p1, p2, ...)` = we don't have this. Our procedural memories are flat text, not structured step sequences.

**The gap this reveals**: Our procedural memories are purely declarative descriptions of procedures. AWM shows that structured step sequences with parameterized slots significantly outperform both (a) raw examples and (b) free-text descriptions. This suggests our procedural memories could benefit from more structured representation -- but the trade-off is that our procedures operate at much higher abstraction levels (interpersonal calibration, reasoning strategies) than web navigation actions.

### The Parameterization Insight

AWM's most distinctive contribution is variable abstraction: replacing concrete values with typed slots (`{product-name}`, `{location}`). This is what makes workflows reusable across tasks. Our procedural memories already do this naturally in prose form -- "when the user corrects a recurring tendency" rather than "when Alexis says X about Y." But we could be more explicit about it. A procedural memory like "use tilde paths for Claude Code file operations" already has an implicit parameter (`{file_path}`). Making parameters explicit could improve retrieval -- searching for memories that match the structure of the current situation, not just the content.

### The Composition Mechanism

AWM's hierarchical workflow building -- simple workflows becoming subgoals of complex ones -- is emergent, not engineered. But it's powerful: the agent recognizes that "find a place" is a substep of "get the zip code of a place." Our system has no composition mechanism. Relationship edges (Priority 2) could serve this role: a `builds_on` or `subgoal_of` edge between procedural memories would capture the same structure.

### The Negative Transfer Finding

The offline+online combination *hurting* performance is the most operationally relevant finding for us. It suggests that mixing memory sources with different quality characteristics or induction contexts can be worse than either alone. For our system: mixing human-curated high-priority memories with auto-generated lower-quality memories in the same retrieval results could produce similar interference. This validates our human-in-the-loop curation model -- keeping quality consistent.

### No Retrieval Mechanism

AWM has **no retrieval at all** for workflow selection. All workflows for a website (~7.4) are concatenated into the prompt. This works because: (a) the number is small, (b) they're pre-grouped by website. Our system faces a fundamentally different problem -- 438+ memories across diverse categories, where selection is critical. AWM is silent on the retrieval question that dominates our design.

---

## Worth Stealing (Ranked)

### 1. Parameterized Slot Abstraction for Procedural Memories (MEDIUM value, MEDIUM effort)

When storing procedural memories, explicitly identify and mark parameterized elements. Instead of:

> "When running git operations, use absolute paths starting with tilde to avoid permission errors"

Store as:

> "When running {tool_name} operations, use absolute paths starting with {path_prefix} to avoid {error_type} errors"
> Parameters: tool_name=git, path_prefix=~, error_type=permission

This makes procedural memories searchable by structure (operations + paths + error handling) rather than only by content keywords. During retrieval, a new situation involving "npm operations with relative paths causing EACCES errors" would match on structure even though the keywords differ.

**Implementation**: Add optional `parameters` JSONB column to memories table. During `remember()` for procedural category, LLM extracts parameter slots. During `recall()`, parameter-aware matching supplements vector similarity.

### 2. Sub-Routine Granularity for Workflow Extraction (MEDIUM value, design insight)

AWM deliberately induces workflows at **sub-task granularity**, not full-task level. "Search for a product on Amazon" rather than "Buy dry cat food and deliver to my address." The sub-routine is reusable; the full task is not.

For our `/sleep` consolidation: when generating summary-layer memories from detail-layer procedural memories, aim for sub-routine granularity. "How to calibrate pushback intensity" is a reusable sub-routine. "How I handled the February 3rd disagreement about the Commons essay" is a one-off episode. The sleep skill should recognize and extract the sub-routine level.

### 3. Hierarchical Workflow Composition via Edges (MEDIUM value, fits Priority 2)

AWM's emergent composition -- simple workflows becoming subgoals of complex ones -- should be captured explicitly in our relationship edge system. When a new procedural memory subsumes or extends an existing one, create a `builds_on` edge. During retrieval, if a complex procedure is retrieved, its component sub-procedures should be available for drill-down (following CogMem's summary-first, detail-on-demand pattern).

**Implementation**: When storing a new procedural memory, run similarity check against existing procedurals. If high overlap on a prefix/subset of steps, create `builds_on` edge. During retrieval, follow `builds_on` edges to expand context.

### 4. Workflow Quality Metrics for Sleep Diagnostics (LOW value, LOW effort)

AWM's quality metrics -- coverage, functional overlap, utility rate -- provide a template for evaluating our procedural memory health during `/sleep`:
- **Utility rate**: What fraction of sessions accessed at least one procedural memory? (We can compute this from access logs.)
- **Overlap**: Do any procedural memories describe redundant procedures? (Similarity check during sleep.)
- **Coverage**: What fraction of user corrections/guidance are captured by existing procedural memories? (Harder to measure, but a useful aspiration.)

### 5. Streaming Induction with Success Gating (LOW value for us, important design pattern)

AWM_online only induces workflows from *successful* trajectories, gated by an LM evaluator. Failed trajectories are discarded. Our system has a stronger version of this: human curation. But the pattern is relevant for any future auto-capture: never auto-store from unsuccessful interactions. The evaluator model adds a quality gate before memory writes.

---

## Not Useful For Us

### 1. No Retrieval Architecture

AWM concatenates all ~7 workflows per website into the prompt. There's no embedding, no similarity search, no ranking. This works because the workflow set is tiny (7.4 per website) and pre-filtered by website. Our system has 438+ memories and needs sophisticated retrieval. AWM offers zero insight on retrieval design -- and retrieval is our most important open problem (Priority 1: RRF fusion).

### 2. Web Navigation Specificity

The entire framework is designed for web navigation: HTML observations, click/type actions, accessibility trees, browser environments. The workflow format (environment description + reasoning + executable action) is tightly coupled to this domain. Our procedural memories operate at much higher abstraction levels (reasoning strategies, calibration patterns, interpersonal heuristics). The specific representation format doesn't transfer.

### 3. Rule-Based Induction

AWM's rule-based induction (extract action type sequences, deduplicate) is an interesting ablation but has no analogue for our use case. Our "actions" are not typed sequences of clicks and types -- they're nuanced judgments about when to push back, when to reframe, when to wait. Rule-based extraction of patterns from conversational interactions would require a fundamentally different approach.

### 4. Action Space Expansion (AWMAS)

Wrapping workflows as callable functions is clever for web agents but irrelevant to our setting. Our agent doesn't have a discrete action space that can be extended with new primitives. The finding that agents resist using newly-added actions (only 18.5% usage) is mildly interesting but not actionable.

### 5. The Per-Website Grouping Strategy

AWM maintains separate workflow memories per website. This is sound for web navigation (workflows are website-specific) but doesn't apply to our cross-domain memory. We need memories that work across contexts, not siloed by domain. The one lesson here is negative: their offline+online combination failed partly because mixing workflow sources produced interference. Our unified memory must handle cross-domain retrieval carefully.

---

## Impact on Implementation Priority

No changes to the ranked priority list from [[systems-analysis]]. AWM is a narrower system than the other papers reviewed -- focused on procedural memory for web navigation with no retrieval, no decay, no contradiction handling, and no consolidation beyond the initial induction step.

| Priority | Feature | AWM's Impact |
|----------|---------|-------------|
| 1 | RRF fusion | AWM has no retrieval mechanism at all. No impact. |
| 2 | Relationship edges | **Mild validation.** AWM's hierarchical composition (simple workflow -> complex workflow) is the kind of structure that `builds_on` edges would capture. But this is emergent in AWM, not an explicit design. |
| 3 | Graded contradiction | AWM has no contradiction handling. The offline+online interference finding hints at the need, but no mechanism is proposed. |
| 4 | Decay + power-law | AWM has no decay. Online workflows accumulate monotonically. The paper doesn't discuss what happens when the workflow set grows large. |
| 5 | Sleep skill | **Weak validation.** AWM's induction step (extract sub-routines from experiences) is a primitive form of consolidation. But it happens per-experience, not as a periodic batch process. The sub-routine granularity insight is useful for sleep prompt design. |
| 6 | Mutation log | AWM tracks no metadata about workflow usage over time. |
| 7 | Confidence scores | AWM's LM evaluator (success gating) is a binary confidence signal. Much coarser than what we need. |
| 8 | Reference index | AWM workflows don't cite their source experiences. |
| 9 | Multi-angle indexing | AWM has no embedding or indexing at all. |

**The one insight that should influence implementation**: parameterized slot abstraction for procedural memories. This is a novel contribution from AWM that no other reviewed paper addresses. It could be integrated into `remember()` for procedural category without major architectural changes -- an LLM pass that identifies and marks variable elements in the stored procedure.

---

## Connections

### To Dynamic Cheatsheet (Suzgun et al.)

DC and AWM both accumulate curated knowledge during inference. DC stores strategies/heuristics in a flat document; AWM stores structured workflows with parameterized steps. Key difference: DC rewrites its entire memory each step (lossy); AWM only adds new workflows (monotonically growing). AWM's approach avoids DC's abbreviation problem but lacks DC's deduplication/refinement mechanism.

DC's usage counter tracks how often strategies are reused -- AWM has no such metric despite measuring utility rate post-hoc. DC's failure on diverse domains (GPQA-Diamond) parallels AWM's per-website grouping: both struggle when memory content doesn't match the current domain.

### To Generative Agents (Park et al.)

GA's reflection process and AWM's workflow induction both extract higher-order knowledge from raw experience. GA produces reflections ("what have I learned?"); AWM produces workflows ("what reusable procedures have I identified?"). GA uses an importance trigger (cumulative importance > 150); AWM triggers on every successful experience. GA preserves provenance (citation trees); AWM does not.

The compositional hierarchy in AWM (simple workflows -> complex workflows) resembles GA's recursive reflection (observations -> reflections -> higher-order reflections). But GA's hierarchy is explicitly managed; AWM's emerges naturally from the induction prompt.

### To A-Mem (arXiv multi-faceted embedding)

A-Mem's multi-faceted embedding (+121% multi-hop improvement) addresses exactly the retrieval gap that AWM sidesteps. If AWM's workflow set grew beyond ~10 per domain, it would need retrieval -- and A-Mem's approach of embedding the same memory from multiple angles (fact, context, implication) would help match workflows to diverse query phrasings. AWM's parameterized slots could serve as additional embedding facets.

A-Mem's weakness (no audit trail) parallels AWM's: neither tracks where workflows/memories came from or how they've been used over time.

### To CogMem (Oberauer FoA/DA/LTM)

CogMem's "summary-first, detail-on-demand" retrieval pattern is what AWM would need at scale. AWM's workflow descriptions `d` function as summaries; the step trajectories are the detail. Currently both are always loaded together. At scale, you'd want to retrieve on `d` (summary) and expand to full trajectory (detail) only when the agent decides to follow that workflow.

AWM's two-component representation (description + trajectory) naturally fits CogMem's two-tier model.

### To Continuum (CMA 6 Requirements)

AWM satisfies CMA requirements partially:
- **Persistent storage**: No -- workflows exist in prompt context only
- **Selective retention**: Yes -- only successful experiences are inducted
- **Associative routing**: No -- all workflows loaded, no routing
- **Temporal chaining**: Weak -- composition is temporal (simple before complex) but not tracked
- **Consolidation**: Partial -- induction IS consolidation, but it's one-shot, not iterative
- **Cross-session**: No -- workflows are session-scoped

AWM satisfies only 1.5 of 6 CMA requirements. It's a procedural memory primitive, not a memory system.

### To Zep (bi-temporal, RRF)

Zep's entity resolution and bi-temporal modeling would address AWM's key blindspots. AWM workflows have no temporal metadata -- a workflow induced early in the stream is treated identically to one induced late, even if the early one was superseded by a better version. Zep's "valid_at" vs "recorded_at" distinction would let AWM track workflow evolution. Zep's triple retrieval via RRF would solve the retrieval problem AWM avoids by keeping workflow sets small.

### To MT-DNC (multi-context promotion)

MT-DNC's tiered memory with write protection for high-tier memories contrasts with AWM's flat, append-only workflow store. The offline+online interference finding suggests AWM *needs* tiers: offline workflows (validated against training data) and online workflows (self-assessed) should have different trust levels. MT-DNC's promotion mechanism (low-tier -> high-tier based on cross-context usage) would help AWM identify which online workflows deserve the same trust as offline ones.

---

## Summary of Key Takeaways

1. **Parameterized abstraction is the core insight.** Replacing concrete values with typed slots makes procedures reusable. This is AWM's most distinctive and transferable contribution. No other reviewed paper addresses this mechanism for procedural memory.

2. **Sub-routine granularity matters.** Inducing workflows at sub-task level (not full-task) produces more reusable, composable memory. This should inform our `/sleep` consolidation prompts for procedural memories.

3. **Mixing memory sources produces interference.** The offline+online combination *hurting* performance is a cautionary finding for any system that blends memories from different quality/induction contexts.

4. **AWM is a narrow system.** It does one thing well (procedural sub-routine extraction for web navigation) but has no retrieval, no decay, no temporal modeling, no contradiction handling, and no consolidation beyond initial induction. It validates the *concept* of procedural memory more than it informs *implementation*.

5. **Hierarchical composition emerges from good induction prompts.** AWM's prompt design -- requesting sub-routine granularity with parameterized slots -- naturally produces composable workflows. The composition is not engineered; it falls out of the representation design.

6. **Structured procedures outperform free-text descriptions.** AWM's step-by-step format with environment state + reasoning + action beats both raw examples and verbalized text descriptions. For our procedural memories: more structure may help, even at higher abstraction levels.
