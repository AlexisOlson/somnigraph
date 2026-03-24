"""Extract all distinct capitalized words from LoCoMo conversations.

Outputs a JSON list of {word, count, examples} for review.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from config import BenchConfig
from ingest import load_locomo

cfg = BenchConfig()
conversations = load_locomo(cfg.locomo_data)

# Collect all capitalized words with context
word_counts = Counter()
word_examples = defaultdict(list)  # word -> list of (conv_idx, session, snippet)

MAX_EXAMPLES = 3

for conv_idx, conv in enumerate(conversations):
    c = conv["conversation"]
    for sess_num in range(1, 50):
        sess_key = f"session_{sess_num}"
        if sess_key not in c:
            break
        for turn in c[sess_key]:
            text = turn.get("text", "")
            words = text.split()
            prev_ends_sentence = True
            for i, w in enumerate(words):
                clean = re.sub(r'[.,!?"\';:()\[\]]+$', '', w)
                if (clean and len(clean) >= 2
                        and clean[0].isupper()
                        and not clean.isupper()
                        and not clean.startswith("I'")):
                    lower = clean.lower()
                    word_counts[lower] += 1
                    if len(word_examples[lower]) < MAX_EXAMPLES:
                        # Get surrounding context
                        start = max(0, i - 3)
                        end = min(len(words), i + 4)
                        snippet = " ".join(words[start:end])
                        word_examples[lower].append({
                            "conv": conv_idx,
                            "session": sess_num,
                            "snippet": snippet[:120],
                        })

# Sort by frequency descending
results = []
for word, count in word_counts.most_common():
    results.append({
        "word": word,
        "count": count,
        "examples": word_examples[word],
    })

out_path = Path.home() / ".somnigraph" / "benchmark" / "locomo_capitalized_words.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Found {len(results)} distinct capitalized words")
print(f"Saved to {out_path}")
print(f"\nTop 30 by frequency:")
for r in results[:30]:
    ex = r["examples"][0]["snippet"] if r["examples"] else ""
    print(f"  {r['word']:20s} {r['count']:5d}x  e.g. ...{ex}...")
