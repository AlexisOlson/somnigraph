import json

with open("C:/Users/Alexis/Repos/Somnigraph/scripts/locomo_bench/extractions/conv0_v6.json") as f:
    data = json.load(f)

print(f"Entities: {len(data['entities'])}")
print(f"Claims: {len(data['claims'])}")
print(f"Segments: {len(data['segments'])}")

# Session coverage
sessions = set(s["session_number"] for s in data["segments"])
claim_sessions = set()
for c in data["claims"]:
    ts = c["time_scope"]
    if isinstance(ts, dict):
        claim_sessions.update(ts.get("sessions", []))

print(f"Sessions with segments: {sorted(sessions)}")
print(f"Sessions in claims: {sorted(claim_sessions)}")

# Fatigue test
for sess in [1, 5, 10, 15, 19]:
    seg_count = len([s for s in data["segments"] if s["session_number"] == sess])
    print(f"  Session {sess:2d}: {seg_count} segments")

# Cross-speaker evaluations
total_evals = 0
eval_examples = []
for s in data["segments"]:
    evals = s.get("cross_speaker_evaluations", [])
    total_evals += len(evals)
    for e in evals:
        eval_examples.append((s["session_number"], e))

print(f"\nCross-speaker evaluations: {total_evals} total")
for sess, e in eval_examples[:10]:
    print(f"  S{sess}: {e[:120]}")

# Outdoor/nature claims
print("\n--- Outdoor/nature claims ---")
outdoor = [c for c in data["claims"] if any(t in c["retrieval_text"].lower() for t in ["camp", "outdoor", "hik", "nature", "beach", "meteor"])]
print(f"Count: {len(outdoor)}")
for c in outdoor[:5]:
    print(f"  {c['retrieval_text'][:120]}")

# Opinion_of claims
print("\n--- opinion_of claims ---")
opinions = [c for c in data["claims"] if c["relation"] == "opinion_of"]
print(f"Count: {len(opinions)}")
for c in opinions:
    print(f"  S{c['time_scope'].get('sessions', ['?'])} {c['subject']}: {c['retrieval_text'][:120]}")

# Trait words in claims
print("\n--- Claims with trait words ---")
for c in data["claims"]:
    rt = c["retrieval_text"].lower()
    if any(w in rt for w in ["thoughtful", "empathy", "driven", "authentic", "courag", "guts"]):
        print(f"  [{c['relation']}] {c['subject']}: {c['retrieval_text'][:120]}")
