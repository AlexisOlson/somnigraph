# 2026-03-14 — External review integration

Four independent reviewers (Gemini, Sonnet, Opus, ChatGPT Pro) conducted blind reviews of architecture and experiments docs.

Convergent findings: feedback self-reinforcement risk, theme boost as compensation for missing graph traversal, enriched embedding degradation, selection bias in evaluation.

One confirmed bug fixed: PPR traversed contradiction-flagged edges, actively co-surfacing conflicting information (scoring.py).

Narrative corrections added to architecture.md (theme boost, feature importance, two-basin caveats; two new open problems). Methodology caveats added to experiments.md (17-query limitation, selection bias). Roadmap expanded with 8 new experiments across all tiers and 4 new open questions. P2 description updated to reflect expanded evaluation scope.

### Surprise

The contradiction edge traversal bug was invisible to internal testing — it required an external eye reading the edge schema docs alongside the PPR code to notice the omission.
