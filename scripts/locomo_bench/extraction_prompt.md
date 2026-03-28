# Multi-Layer Graph Extraction Prompt (v6 — single pass, scoped tasks)

One LLM call per conversation. Tasks are scoped by what context they
need: entities and claims use full-conversation context; segments and
cross-speaker evaluations are local per-segment. Relationship nodes
and entity facets are built by script from LLM output.

---

## Prompt

```
You are extracting structured information from a multi-session
conversation for a search index. Items are retrieved via keyword
matching and semantic similarity against natural-language questions.

This extraction has two scopes:

GLOBAL SCOPE (requires full-conversation awareness):
- Entities: identify once, merge aliases across sessions
- Claims: deduplicate across sessions; one claim per recurring fact

LOCAL SCOPE (process per topical exchange within each session):
- Segments: transform each exchange into searchable text
- Cross-speaker evaluations: detect within each segment

GLOBAL RULES:
- ACTIVE VOICE ATTRIBUTION: Always name the speaker explicitly.
  "Melanie described..." not "They discussed..."
- Every item must include evidence_turn_ids.
- Epistemic status on every claim:
  "explicit" = directly stated
  "strong_inference" = 2+ instances across conversation, or clear
  context
  "weak_inference" = single mention, plausible but limited evidence
- Session boundaries: turn ID prefixes encode session (D1:x =
  session 1, D2:x = session 2, etc.)

## Output format

```json
{
  "entities": [ ... ],
  "claims": [ ... ],
  "segments": [ ... ]
}
```

### entities

```json
{
  "name": "string (canonical name)",
  "aliases": ["string", ...],
  "type": "person | place | organization | thing",
  "grounding": "string | null"
}
```

### claims

```json
{
  "subject": "string (canonical entity name)",
  "relation": "has_trait | has_skill | has_preference | has_goal | has_experience | nickname | opinion_of | located_in | type_of | other",
  "object": "string (specific detail, under 15 words)",
  "attributed_to": "string (speaker name, or 'narrator' if inferred)",
  "polarity": "positive | negative | neutral",
  "status": "explicit | strong_inference | weak_inference",
  "time_scope": {"type": "stable | session | range | historical | recurring", "sessions": [1]},
  "evidence_turn_ids": ["D1:3", "D1:5"],
  "retrieval_text": "string — self-contained sentence, all entities named"
}
```

### segments

```json
{
  "session_number": 1,
  "turns": ["D1:1", "D1:2", "D1:3"],
  "topic": "short label",
  "surface_events": ["string", ...],
  "categories": ["string", ...],
  "implied_traits": ["string", ...],
  "cross_speaker_evaluations": ["string", ...],
  "literal_summary": "string (30-50 tokens)",
  "retrieval_text": "string (50-80 tokens)",
  "bridging_terms": ["string", ...],
  "latent_concepts": ["string", ...]
}
```

## Entity instructions (GLOBAL SCOPE)

Identify every person, place, organization, or notable thing across
the FULL conversation. Each entity appears once regardless of how
many sessions mention it.

ALIAS RULE: Merge all references to the same entity. "Mel",
"Melanie", and "my friend" are one entity with canonical name
"Melanie" and aliases ["Mel"].

GROUNDING RULE (mechanical — apply to ALL):
- Every location → country and state/region
- Every brand/product → category and what it is
- Every cultural reference → creator, genre, one-sentence desc
- Common first names, pets → null

No profile prose. Entity aggregation is handled by script.

## Claim instructions (GLOBAL SCOPE)

Extract every discrete factual or evaluative statement. Claims use
full-conversation awareness to determine epistemic status and
deduplicate recurring facts.

DEDUP RULE: For recurring facts or preferences, produce ONE claim
with time_scope type "recurring" and list all evidence_turn_ids
across sessions. Do not repeat the same claim for each session.

Claims include:
- Facts (traits, skills, preferences, goals, experiences)
- Nicknames, aliases, terms of address
- Plans, intentions, commitments
- State changes (new job, new pet, moved, new hobby)

ATTRIBUTION: First-person disclosures and third-person
characterizations have different reliability. For traits inferred
from behavior, use attributed_to: "narrator".

FREQUENCY RULE: Trait-level claims require 2+ instances across the
conversation or an explicit statement for "strong_inference".
Single casual mentions → "weak_inference".

DETAIL RULE: The object field must capture WHAT SPECIFICALLY, not
just the topic. Specific but concise — under 15 words.

OPINION_OF CLAIMS: Cross-speaker evaluations are captured in the
segments (see below). You do NOT need to extract opinion_of claims
separately here. Only extract opinion_of claims for evaluations
that span multiple sessions and are clearly a stable assessment
(e.g., a recurring theme of one speaker praising another's courage
across many sessions). Single-instance evaluations belong in their
segment's cross_speaker_evaluations field.

The retrieval_text must be a complete, self-contained sentence
resolving all pronouns and naming all entities.

## Segment instructions (LOCAL SCOPE)

Divide EACH session into topical exchanges (3-8 contiguous turns).
Process each segment locally — focus on what is in these specific
turns.

TRANSFORMATION PROTOCOL (all intermediate fields required):
1. surface_events: List concrete events, objects, statements.
2. categories: Name the category each belongs to.
3. implied_traits: Traits/preferences implied — ONLY if supported
   by enthusiasm, repetition within this segment, or explicit
   framing. Omit events with no trait implication. Empty array OK.
4. retrieval_text: Write using BOTH category vocabulary AND specific
   surface details. Active voice attribution required.

### Cross-speaker evaluations (LOCAL — extract within each segment)

For EVERY segment, scan for evaluative statements between speakers:
compliments, criticism, personality descriptions, expressions of
admiration, gratitude, or concern directed at the other speaker.

Common patterns to look for:
- Direct compliments: "you're so [trait]", "I love how you..."
- Gratitude with characterization: "thanks for always being [trait]"
- Behavioral observations: "you always [pattern]"
- Implicit evaluations: offering advice (→ views other as receptive),
  sharing vulnerabilities (→ trusts the other)

For each evaluation found, add a string to cross_speaker_evaluations
formatted as: "Observer characterizes Subject as [assessment]
(evidence: turn ID)"

Example: "Melanie characterizes Caroline as empathetic and
understanding (evidence: D1:12)"

If no evaluations are present in this segment, use an empty array.

### bridging_terms and latent_concepts

bridging_terms (5-10): Terms someone might search for NOT in the
original turns. Include hypernyms, answer-type labels ("nickname",
"career fit", "preferred vacation style"), category terms.

latent_concepts (5-10): Abstract nouns/categories implied but never
spoken in the original turns.

## The conversation

{full_conversation_text}
```

---

## Post-extraction script pipeline (zero LLM cost)

### 1. Relationship node construction

Group all cross_speaker_evaluations from segments by (observer,
subject) pair. Concatenate into a directed relationship node:

```json
{
  "node_type": "relationship",
  "observer": "Melanie",
  "subject": "Caroline",
  "relationship_type": "friendship",
  "traits_attributed": ["thoughtful", "empathetic", "driven"],
  "characterization": "concatenated evaluations as prose",
  "evidence_turn_ids": ["D1:12", "D7:4", "D13:16", "D16:18"],
  "likely_questions": ["templated from traits_attributed"]
}
```

Relationship type can be inferred from claim relations between the
two entities or set to a default.

### 2. Entity facet construction

Group claims by (entity, topic). Topic clustering options (from
cheapest to best):
- Keyword lists per hand-built taxonomy (~20 categories)
- TF-IDF on claim object fields within each entity
- Embedding clustering on claim retrieval_text (uses existing
  embedding pipeline)

For each cluster with 3+ claims, concatenate retrieval_text fields
into a facet node. Add likely_questions templated from the cluster's
bridging_terms.

### 3. Bridging terms distribution (for ablation step 0)

Map each segment's bridging_terms and latent_concepts back to its
constituent turns via the turns array. Add as BM25-indexed fields
on the original turn nodes. This allows testing vocabulary bridging
without any synthetic nodes in the pool.

### 4. Entity dedup validation

Scan entities for probable duplicates (fuzzy match on names +
aliases). Flag for manual review or auto-merge at high confidence.

### 5. Likely questions generation

For relationship nodes: template from traits_attributed.
  "What does {observer} think about {subject}?"
  "How does {observer} describe {subject}?"
  "What personality traits does {observer} attribute to {subject}?"

For entity facets: template from cluster topic + entity name.
  "What are {entity}'s {topic} activities?"
  "What does {entity} do for {topic}?"

Store as separate BM25-indexed field on each node.

### 6. Opinion_of claim validation

For each speaker pair, count cross_speaker_evaluations. If a 10+
session conversation produces <3 evaluations in either direction,
flag as potentially incomplete extraction.

---

## Implementation notes

### Retrieval architecture
- **Separate retrieval pools.** Turns and synthetic nodes searched
  independently, merged via RRF.
- **Provenance expansion.** Retrieved synthetic nodes include their
  evidence_turn_ids in reader context.
- **BM25 fields:** bridging_terms, latent_concepts, likely_questions,
  traits_attributed all indexed separately from main retrieval_text.
  Excluded from embedding input.
- **session_number** as structured metadata for temporal queries.
- **node_type** stored for future reranker feature.

### Cost
- 10 Opus calls (one per conversation, 10-21k tokens input)
- ~180k total input tokens
- Script pipeline: zero LLM cost

### Ablation plan
0. Turns + bridging_terms distributed from segments (no synthetic
   nodes — tests pure vocabulary augmentation)
1. Turns only (current baseline)
2. Turns + claims
3. Turns + claims + script-built entity facets
4. Turns + claims + facets + script-built relationship nodes
5. Turns + claims + facets + relationships + segments
6. Full (+ bridging_terms, latent_concepts in BM25)

Per-question tracking for all 18 hard failures + regression on 67
successful multi-hop questions.

**Stopping criterion:** A node type earns its place if it recovers
≥2 hard failures without causing ≥3 regressions on currently-
successful questions.
