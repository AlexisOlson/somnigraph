# 2026-03-22 — Two-phase expansion fixes + new features

Fixed 3 train/eval mismatches: (1) original candidates lacked embeddings in phase 2 eval, (2) temporal feature used different regex in eval vs training, (3) theme_overlap not recomputed for expanded candidates.

Fixed entity bridge extraction stopword leak — sentence-initial words and I-contractions passed the capitalization heuristic (272 unique "entities", top hits were "hey" 196x, "anything" 174x, "can't" 159x).

Added 3 new Group H features: expansion_method_count, phase1_rrf_score, is_seed (32 features total). Expansion method analysis: 3 of 6 methods are dead (rocchio 0%, multi_query 2%, entity_focus 4%), added ablation to roadmap.

Built overnight.py orchestrator for batch experiments: forward stepwise → backward elimination → train/eval 3 feature sets × baseline/expanded × 10 conversations.

### Surprise

The entity bridge stopwords were invisible because the log only shows the first 5 alphabetically — "and, any, awesome, basketball, being" looks reasonable until you count that "hey" appeared 196 times across 954 questions.
