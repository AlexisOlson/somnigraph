# Multi-Hop Retrieval Failure Analysis

*2026-03-23. Analysis of the 22 multi-hop (category 3) questions that miss at R@10 in the LoCoMo benchmark, using the Level 4 17-feature reranker.*

## Summary

Of 89 multi-hop questions, 67 hit at R@10 (75.3%). The 22 misses break into two structural groups:

- **4 near-misses** (rank 11-20): reranker or expansion improvements could recover these.
- **18 complete whiffs** (not in top 20): both bare RRF and the reranker fail identically. These are **retrieval failures**, not ranking failures.

The 18 hard misses share a root cause: **vocabulary gap between query and evidence**. 30 of 34 evidence turns have zero content-word overlap with their query (88%). Mean vector cosine similarity is 0.407 vs 0.507 for R@1 hits — the evidence turns rank mid-pack in a ~588-turn corpus, nowhere near the top 20.

## Comparison: bare RRF vs reranker

The reranker improved multi-hop from 47/89 to 67/89 R@10 — a large gain. But all 18 complete whiffs are shared between bare and reranker. The remaining failures are structurally beyond what ranking can fix.

| Metric | Bare RRF | Reranker |
|--------|----------|---------|
| R@10 | 47/89 (52.8%) | 67/89 (75.3%) |
| R@20 | 58/89 (65.2%) | 71/89 (79.8%) |

## Failure taxonomy

### Type A: Pure inference (7 questions)

The answer must be inferred from behavior; it isn't stated in the evidence. The evidence turns use completely different vocabulary than the query.

| Conv | Question | Evidence content | Vocabulary gap |
|------|----------|-----------------|----------------|
| 0 | Would Melanie prefer a national park or theme park? | Camping trips, marshmallows, meteor showers | "park" absent; inference: outdoor activities → national park |
| 2 | Does John live close to a beach or the mountains? | Nature walk photo, "appreciate nature" | "beach"/"mountains" absent; requires image understanding |
| 3 | Does Nate have friends besides Joanna? | "me and my team had a blast" (CS:GO) | "friends" absent; inference: teammates = friends |
| 3 | What nickname does Nate use for Joanna? | "Hey Jo, guess what I did?" | "nickname" absent; inference: "Jo" is the nickname |
| 6 | Was James feeling lonely before meeting Samantha? | "pets, games, travel, pizza bring happiness" | "lonely" absent; inference: no human sources of happiness |
| 6 | Did John and James study together? | "Remember this photo from elementary school?" | "study" absent; inference: elementary school → studied together |
| 7 | Why did Jolene sometimes put off doing yoga? | "planned to play console with partner" | "yoga" absent; inference: chose gaming over yoga |

**Cosine similarities**: 0.27-0.54 (mean 0.37). These evidence turns don't even *seem* related to the query in embedding space.

### Type B: Scattered weak signals (6 questions)

The answer requires synthesizing multiple turns that individually score poorly. No single turn is sufficient, and each has minimal query overlap.

| Conv | Question | Evidence | Pattern |
|------|----------|----------|---------|
| 0 | What personality traits would Melanie say Caroline has? | 3 turns: "you're so thoughtful", "you care about being real", "your drive to help" | Compliments scattered across sessions D7, D13, D16 |
| 2 | What job might Maria pursue? | 4 turns: shelter front desk, volunteering origin, giving talks, reaching out to help | Volunteering narrative across sessions D5, D11, D27, D32 |
| 3 | What alternative career for Nate after gaming? | 4 turns: turtle care, new tank, turtle diet, turtle joy | Turtle-care expertise across sessions D5, D19, D25, D28 |
| 4 | What could John do after basketball? | 3 turns: make a difference/foundation, started seminars, great leader | Leadership/service across sessions D11, D26, D27 |
| 5 | Indoor activity Andrew would enjoy + make dog happy? | 2 turns: getting into cooking + meet Toby my puppy | Cooking (D10) + dog ownership (D12), different sessions |
| 5 | Improve stress and accommodate living with dogs? | 3 turns: work piling up/stuck inside, work tough/stressful, miss outdoors/city | Work-stress-outdoors across sessions D12, D18, D21 |

**Cosine similarities**: 0.33-0.51 (mean 0.42). Slightly better than Type A because individual turns are topically adjacent, but none crosses the retrieval threshold alone.

### Type C: External knowledge required (5 questions)

The answer requires geographic, brand, or world knowledge not present in the conversation text.

| Conv | Question | Evidence says | Knowledge needed |
|------|----------|-------------|-----------------|
| 4 | Shop Tim would enjoy in NYC? | "MinaLima... created props for Harry Potter films" | MinaLima has a store in New York City |
| 5 | Which US state do Audrey and Andrew live in? | Hiking trail map image | Geographic identification from image |
| 5 | Which national park are they referring to? | "went to a beautiful national park" + trail map | Voyageurs NP matches the conversational clues |
| 7 | Which US state did Jolene visit during internship? | "yoga on top of mount Talkeetna" | Mount Talkeetna is in Alaska |
| 8 | Electronic device for Sam's fitness goals? | "my healthy road... two years, ups and downs" | Very weak connection; arguably poor ground truth |

**Cosine similarities**: 0.33-0.47 (mean 0.38). The vocabulary gap is compounded by missing world knowledge.

### Category overlap

Several questions fall into multiple categories. "Indoor activity + dog happy" is both Type B (scattered: cooking + puppy) and partially Type A (inference: cooking + dog → cook dog treats). The taxonomy is useful for designing interventions, not for strict classification.

## Vector similarity distributions

| Group | Mean cosine | Median | Min | Max |
|-------|------------|--------|-----|-----|
| R@1 hits (sample of 15) | 0.507 | 0.516 | 0.342 | 0.611 |
| Hard misses (all 34 evidence turns) | 0.407 | 0.404 | 0.274 | 0.550 |

The distributions overlap: some misses at 0.55 outscore some hits at 0.34. The difference is that hits also benefit from keyword matches via BM25 (and occasionally from speaker-name theme matching), while misses have neither keyword nor strong vector signal.

## What would make each type trivially retrievable

### For Type A — Implication extraction

A sleep pass extracts what each turn *implies*, not just what it says. The extracted implications are stored as searchable text (summary, themes, or a separate field).

Examples:
- "Melanie: camping, marshmallows, meteor showers" → *"outdoor enthusiast, enjoys camping and nature activities"* — now retrievable by "national park"
- "James: pets, games, travel, pizza bring happiness" → *"happiness sources are non-human, suggesting social isolation or loneliness"* — now retrievable by "lonely"
- "Hey Jo, guess what" → *"Nate uses nickname 'Jo' for Joanna"* — now retrievable by "nickname"
- "planned to play console with partner" → *"chose gaming over other activities like yoga"* — now retrievable by "put off yoga" if combined with context

### For Type B — Entity-scoped summaries

Aggregate all turns about a person/topic into a bridge document that makes the implicit pattern explicit:

- *"Maria volunteers at homeless shelters: front desk work, giving talks, inspired by aunt's charity work. ~1 year of involvement. Demonstrated skills: public speaking, community outreach, empathy. Potential career paths: shelter coordinator, social worker, counselor."*
- *"Nate keeps pet turtles: knows their diet (vegetables, fruits, insects), habitat requirements (clean area, proper lighting, adequate space). Finds turtle care enjoyable and peaceful. Has multiple turtles in a growing collection. Animal care skills could translate to zookeeping or veterinary work."*

### For Type C — Entity enrichment with factual grounding

When named entities appear in turns, a sleep pass resolves them against world knowledge:
- "mount Talkeetna" → enriched with *"Talkeetna, Alaska"*
- "MinaLima" → enriched with *"MinaLima design studio, locations in London and New York City"*

## Cost analysis

~5,882 turns across 10 conversations.

| Intervention | Calls | Estimated cost | Questions addressed |
|-------------|-------|---------------|-------------------|
| Implication extraction (Haiku) | 600-1200 (batched) | ~$0.50-$1.50 | Type A + B (13/18) |
| Entity-scoped summaries (Haiku) | 100-200 per conv | ~$0.50-$2.00 | Type B (6/18) |
| Entity enrichment (mini) | 500-1000 | ~$0.50-$1.00 | Type C (5/18) |
| **Total** | | **~$1.50-$4.50** | **18/18** |

One benchmark run (1977 questions × reader + judge) costs significantly more. All three interventions combined are cheaper than a single benchmark run.

## Implications for sleep pipeline design

The highest-impact single intervention is **implication/fact extraction** — it addresses 13 of 18 hard misses (Types A + B). This is essentially what EXIA GHOST does to achieve 89.94% with pure vector search and no reranker.

Entity-scoped summaries are the second lever, specifically compressing scattered evidence into a single retrievable document. These are a natural fit for the existing REM summary generation pipeline.

Type C (external knowledge) is addressable but lowest priority — 5 questions, and one (Conv 8, "electronic device") has questionable ground truth.

## All 22 R@10 misses (reference)

### Complete whiffs (18, not in top 20)

1. **Conv 0**: What personality traits might Melanie say Caroline has? — Evidence: D16:18, D13:16, D7:4
2. **Conv 2**: Does John live close to a beach or the mountains? — Evidence: D22:15
3. **Conv 2**: Would John be open to moving to another country? — Evidence: D24:3, D7:2
4. **Conv 2**: What job might Maria pursue in the future? — Evidence: D32:14, D5:8, D11:10, D27:4
5. **Conv 3**: Is it likely that Nate has friends besides Joanna? — Evidence: D1:7
6. **Conv 3**: What nickname does Nate use for Joanna? — Evidence: D7:1
7. **Conv 3**: What alternative career might Nate consider after gaming? — Evidence: D5:8, D19:3, D25:19, D28:25
8. **Conv 4**: Based on Tim's collections, what is a shop he would enjoy in NYC? — Evidence: D2:9
9. **Conv 4**: What could John do after his basketball career? — Evidence: D11:19, D26:1, D27:26
10. **Conv 5**: What indoor activity would Andrew enjoy + make his dog happy? — Evidence: D10:12, D12:1
11. **Conv 5**: What can Andrew do to improve stress and accommodate his dogs? — Evidence: D12:3, D18:1, D21:5
12. **Conv 5**: Which US state do Audrey and Andrew potentially live in? — Evidence: D11:9
13. **Conv 5**: Which national park could they be referring to? — Evidence: D5:8, D11:9
14. **Conv 6**: Was James feeling lonely before meeting Samantha? — Evidence: D9:16
15. **Conv 6**: Did John and James study together? — Evidence: D17:13
16. **Conv 7**: Why did Jolene sometimes put off doing yoga? — Evidence: D3:11, D2:30
17. **Conv 7**: Which US state did Jolene visit during her internship? — Evidence: D13:15
18. **Conv 8**: What electronic device could Evan gift Sam for fitness? — Evidence: D5:7

### Near-misses (4, rank 11-20)

1. **Conv 0**: Would Melanie prefer national park or theme park? — rank 20, evidence: D10:12, D10:14
2. **Conv 3**: What job is Joanna beginning to perform from movie scripts? — rank 11, evidence: D29:1
3. **Conv 4**: Would Tim enjoy C.S. Lewis or John Greene? — rank 20, evidence: D1:14, D1:16, D1:18
4. **Conv 4**: Good hobby related to Tim's travel dreams? — rank 16, evidence: D4:1, D6:6, D15:3, D27:37
