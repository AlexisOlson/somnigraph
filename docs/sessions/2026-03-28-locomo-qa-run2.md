# 2026-03-28 — End-to-end QA Run 2 with L5b retrieval

GPT-4.1-mini reader, batch Opus judge. Original GT: 85.6% (+0.5pp over Run 1). Corrected GT: 87.2%. Large retrieval gain (R@10 +6.7pp) produced modest QA gain — bottleneck shifted from retrieval to reader extraction. Temporal regressed -7.5pp (under investigation).

Added batch judge script (judge_batch.py), fixed run.py for multi-conversation graph building (schema init reset, dia_map persistence, conv cache clearing). Updated leaderboard and headroom notes.
