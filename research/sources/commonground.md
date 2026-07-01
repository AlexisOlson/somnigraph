# CommonGround - A constitutional ledger kernel for multi-agent work coordination (not a memory system)

*Generated 2026-06-30 by Opus agent reading the repo (+ carsteneu evidence file)*

---

## Architecture

CommonGround Kernel v3r1-preview (Intelligent-Internet) is explicitly **not a memory system**. The README's own words: it is "a small constitutional ledger kernel" that "does not try to own every workflow, runtime, memory system, or orchestration framework." It preserves the minimum durable public facts (Agent identity, Turn lifecycle, work records, claim fencing, causal lineage) that independent agents need to coordinate across handoffs. Search, ranking, and content interpretation are deliberately deferred to "upper layers."

180 Python files, PostgreSQL-backed, Apache-2.0, Python 3.13+. The core kernel is four small modules in `CommonGround/kernel/`: `topology.py` (agent registration/presence), `lifecycle.py` (turn dispatch, claim tokens, reconciliation), `semantic.py` (append/list semantic records), `ledger.py` (feed pagination).

### Storage & Schema
PostgreSQL via a `TruthRepositoryPort` abstraction (`CommonGround/infra/repositories.py`). The data model (`CommonGround/service/schemas.py`, `CommonGround/contracts/models.py`) is agent-coordination rows: `AgentRef`, `TurnRef`, `ClaimTokenModel` (with `expires_at`), `AgentBirthSpecModel` (role, capabilities, grants, capacity), and provenance records (`AgentRegistrationProvenanceModel` with `kind`/`external_ref`/`payload_hash`). Actual content lives in **CardBox**, an external content-addressed blob store (git submodule `CG-Cardbox`, not vendored here). Records are pointers: `SemanticRecordSpec` is just `record_role: str` + `cardbox_ref: CardBoxRef` (`models.py:178`). No memory-unit schema, no themes, no category, no priority, no embeddings column.

### Memory Types
None in any retrieval sense. The closest analogues are **semantic records** (turn-scoped facts appended to the ledger) and **WorkMemoryReport** submissions (`WorkMemoryReportSubmissionRequest` in `schemas.py`: a manifest of `records` each with `role`, `payload`, `source_refs`, plus an optional `final_payload`). "WorkMemory" here means a durable report of what a turn produced, not a queryable memory store.

### Write Path
`SemanticKernel.append_semantic_record()` (`kernel/semantic.py`) validates the cardbox project matches the claim token, checks the box exists, then calls `truth.append_semantic_record_primitive()`. That is the entire write path: a claim-fenced append. **No extraction, no dedup, no enrichment, no salience/quality gating.** The constitution (per evidence file) states the kernel "must not interpret business fields" beyond the envelope — extraction is deliberately out of scope.

### Retrieval
**None.** There is no vector search, no BM25, no fulltext, no fusion, no reranking anywhere in the repo. `SemanticKernel.list_semantic_records()` returns records for a turn filtered by `turn_seq > after` and sliced to a `limit` — linear pagination in insertion order. `LedgerKernel.fetch_ledger_feed()` is the same: an append-only feed page after a sequence number. The single grep hit for "embedding" is a comment in `Integrations/admin_service/admission_api.py:80` explaining that the *embedding product / Admin Service* (an external product layer) owns auth — nothing to do with vector embeddings.

### Consolidation / Processing
None. No offline processing, sleep cycles, summarization, or merging.

### Lifecycle Management
Only **claim-token TTLs**: `ClaimToken.expires_at` and `ClaimRenewal` in `lifecycle.py` enforce authorization time-to-live and fencing (only one agent holds a live claim on a turn). This is concurrency/authorization lifecycle, **not content decay**. No superseding, contradiction detection, versioning of facts, archival, or forget mechanism. `LifecycleKernel.dispatch()` handles turn spawning with `DispatchAuthorityMode` (ROOT_REQUEST vs CHILD_DERIVATION) and causal `cause.kind`/`cause.id` lineage — this is a work-DAG, not a memory graph.

---

## Key Claims & Evidence

| Claim | Evidence | Assessment |
|-------|----------|------------|
| "Durable public substrate for multi-agent handoffs" | Ledger append + claim fencing implemented in kernel | Plausible for its stated goal (coordination), not memory |
| Causal lineage across turns | `DispatchAuthority` root/child modes, `cause` refs in `lifecycle.py` | Real, but it is a work-provenance DAG |
| Is a memory system | — | **False by the project's own design axiom**; no retrieval/extraction/decay exists |
| Benchmarks | None; tests are functional/integration (`tests/`) | No comparable numbers |

---

## Relevance to Somnigraph

### What CommonGround does that Somnigraph doesn't
- **Multi-agent claim fencing / concurrency control** (`lifecycle.py` claim tokens with TTL) — Somnigraph is single-user MCP and has no notion of concurrent writers competing for a record. Not a gap; a different problem.
- **Write-time causal/provenance lineage on every record** — each turn carries an explicit `cause` (root external request or parent-claim derivation), and `WorkMemoryReportRecord` carries `source_refs`. Somnigraph's `db.py` edge table records typed edges but they are *detected during sleep*, and memories do not carry structured source provenance at write time. This is a genuine contrast, though CommonGround's edges are about agent work, not memory content.

### What Somnigraph does better
Everything a memory system needs. CommonGround has no retrieval at all, so `reranker.py`, `scoring.py` (RRF+PPR), `fts.py`, `embeddings.py`, the sleep pipeline (`sleep_nrem.py`/`sleep_rem.py`), decay, feedback loop, and LoCoMo/PERMA evaluation have no counterpart. Comparison is category-mismatched: Somnigraph *is* a memory system; CommonGround explicitly is not.

---

## Worth Stealing (ranked)

### 1. Write-time causal/provenance lineage on records (Low effort, note-only)
**What**: Every CommonGround record is stamped at write time with its cause (external request vs derived-from-parent) and `source_refs`.
**Why**: Somnigraph already has typed `derivation`/`revision` edges but builds them offline during NREM sleep; a lightweight write-time `source_ref` on `remember()` would give the sleep classifier a prior rather than inferring lineage cold.
**How**: optional `source` field threaded through `tools.py::remember` into the `db.py` schema, consumed as a feature/prior by `sleep_nrem.py` edge classification. Marginal — Somnigraph's sleep-detected edges are the deliberate design choice, so this is redundant with existing capability, listed only so the idea is not lost.

---

## Not Useful For Us

### The kernel itself, claim fencing, agent topology, CardBox blob store
All of it targets multi-agent coordination and durable public work records — a different layer of the stack than a single-user memory store. There is no retrieval, ranking, extraction, consolidation, or decay to borrow.

---

## Connections

Aligns with the Phase 18 source-sweep finding indirectly: CommonGround pushes *all* interpretation (extraction, search, memory) to upper layers, betting on write-path/record discipline as the durable core — echoing the write-path-quality theme from ByteRover/agentmemory/MIRIX (`ai-memory-comparison.md`), but taken to the extreme of having no read path at all. Unlike every retrieval-oriented system in the corpus (memos, Recall, MIRIX), CommonGround is a coordination substrate a memory system would sit *on top of*, not compete with.

---

## Summary Assessment

CommonGround is a well-scoped, honestly-labeled multi-agent coordination ledger — a "constitutional kernel" for durable public work records with agent identity, turn lifecycle, claim fencing, and causal lineage on PostgreSQL. Its design axiom ("assume nothing beyond what constraints demand") makes its many absences (no search, no extraction, no decay, no memory types) principled rather than incomplete. The carsteneu evidence file is accurate and unusually candid: it states outright that CommonGround is "explicitly not a memory system" and that search is "entirely out-of-scope for v3r1."

For a memory-system survey, the correct verdict is SKIP. There is no retrieval, ranking, consolidation, or lifecycle mechanism to adopt, and no benchmark to compare against Somnigraph's 85.1% LoCoMo QA. The single transferable concept — write-time causal/provenance stamping of records — is already covered by Somnigraph's sleep-detected typed edges and is listed only for completeness. The most useful takeaway is negative and architectural: CommonGround demonstrates the *coordination layer beneath* a memory system, which is orthogonal to Somnigraph's single-user MCP design and not a gap Somnigraph needs to fill.

**Evidence cross-check**: The evidence file is consistent with the code and refreshingly honest. Its checkmarks describe coordination features (multi-agent dispatch, claim-token TTL authorization, provenance) and its ✗ marks (no search, no extraction, no layered memory, no benchmarks) all match what the code shows. Sharpest correction is not against the evidence file but against the survey table note ("multi-agent coordination ledger" — correct) and any reader who might expect memory mechanisms: there are none, by design.
