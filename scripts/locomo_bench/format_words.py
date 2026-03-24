"""Format capitalized words list compactly for review."""
import json
from pathlib import Path

src = Path.home() / ".somnigraph" / "benchmark" / "locomo_capitalized_words.json"
dst = Path.home() / ".somnigraph" / "benchmark" / "locomo_words_compact.txt"

with open(src) as f:
    data = json.load(f)

lines = []
for r in data:
    ex = r["examples"][0]["snippet"] if r["examples"] else ""
    lines.append(f'{r["word"]} ({r["count"]}x) -- {ex}')

with open(dst, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"{len(lines)} words written to {dst}")
