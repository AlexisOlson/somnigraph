"""Quick test of LLM enrichment on hard retrieval cases."""
import json
import sys
from pathlib import Path

from openai import OpenAI

key = Path.home().joinpath(".claude/secrets/openai_api_key").read_text().strip()
client = OpenAI(api_key=key)

EXTRACT_PROMPT = """Extract searchable metadata from this conversation turn. Return JSON with:
- "topics": list of 3-8 key topics/concepts (nouns, entities, activities)
- "facts": list of 1-3 factual statements that could answer future questions
- "entities": list of named entities (people, places, books, organizations)

Be thorough — include implicit topics. If someone mentions "my son got in an accident" extract ["son", "children", "family", "accident"].

Turn: {text}

JSON:"""

tests = [
    '[Caroline] The transgender stories were so inspiring! I was so happy and thankful for all the support.',
    '[Melanie] Hey Caroline, that roadtrip this past weekend was insane! We were all freaked when my son got into an accident.',
    '[Caroline] I loved "Becoming Nicole" by Amy Ellis Nutt. It is a real inspiring true story about a trans girl and her family.',
    '[Melanie] We always look forward to our family camping trip. We roast marshmallows, tell stories around the campfire.',
    '[Caroline] Researching adoption agencies - it has been a dream to have a family and give a loving home to kids who need it.',
    '[Melanie] Thanks! They were scared but we reassured them and explained their brother would be OK. They are tough kids.',
]

for text in tests:
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": EXTRACT_PROMPT.format(text=text)}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(r.choices[0].message.content)
    print(f"Turn: {text[:70]}...")
    print(f"  Topics:   {parsed.get('topics', [])}")
    print(f"  Facts:    {parsed.get('facts', [])}")
    print(f"  Entities: {parsed.get('entities', [])}")
    print()
