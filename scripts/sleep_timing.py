"""Parse sleep progress log and compute batch timing stats."""
import sys
from pathlib import Path
import re
from datetime import datetime
from math import sqrt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from memory.constants import DATA_DIR

log = (DATA_DIR / "sleep_progress.log").read_text(encoding="utf-8")

# Extract timestamps and batch info
batch_completions = []
for line in log.strip().split("\n"):
    m = re.match(r"\[(\d{2}:\d{2}:\d{2})\]\s+Batch (\d+)/(\d+): (\d+) pairs", line)
    if m:
        t = datetime.strptime(m.group(1), "%H:%M:%S")
        batch_completions.append({
            "time": t,
            "batch": int(m.group(2)),
            "total": int(m.group(3)),
            "pairs": int(m.group(4)),
        })

# Find launch time
launch_time = None
for line in log.strip().split("\n"):
    m = re.match(r"\[(\d{2}:\d{2}:\d{2})\]\s+Launched", line)
    if m:
        launch_time = datetime.strptime(m.group(1), "%H:%M:%S")
        break

if not batch_completions or not launch_time:
    print("Not enough data in progress log")
    exit()

batch_completions.sort(key=lambda x: x["time"])
total_batches = batch_completions[0]["total"]
parallelism = 3

# Reconstruct waves: every time 3 slots fill, that's a wave boundary
# Wave ends when the 3rd completion arrives (or fewer for partial last wave)
wave_durations = []
wave_start = launch_time
completed_in_wave = 0
wave_num = 1

for i, bt in enumerate(batch_completions):
    completed_in_wave += 1
    # A wave ends when we've seen `parallelism` completions, or it's the last batch
    if completed_in_wave == parallelism or i == len(batch_completions) - 1:
        wave_end = bt["time"]
        dur = (wave_end - wave_start).total_seconds()
        wave_durations.append(dur)
        wave_start = wave_end
        completed_in_wave = 0
        wave_num += 1

n = len(wave_durations)
mean_wave = sum(wave_durations) / n
variance = sum((d - mean_wave) ** 2 for d in wave_durations) / n
std_wave = sqrt(variance)

total_seconds = (batch_completions[-1]["time"] - launch_time).total_seconds()
total_pairs = sum(bt["pairs"] for bt in batch_completions)

print(f"Batches completed: {len(batch_completions)}/{total_batches}")
print(f"Parallelism: {parallelism}")
print(f"Waves completed: {n}")
print(f"Total wall time: {total_seconds:.0f}s ({total_seconds/60:.1f} min)")
print()
print(f"Per-wave: mean={mean_wave:.0f}s ({mean_wave/60:.1f} min), std={std_wave:.0f}s ({std_wave/60:.1f} min)")
print(f"Wave durations: {[f'{d:.0f}s' for d in wave_durations]}")
print()
print(f"Total pairs: {total_pairs}")
print(f"Throughput: {total_seconds / total_pairs * parallelism:.1f}s per pair")
print()
est_waves = -(-total_batches // parallelism)
print(f"=== Estimation ===")
print(f"  deep_sleep_minutes ~= waves * {mean_wave/60:.1f} +/- {std_wave/60:.1f}")
print(f"  waves = ceil(batches / {parallelism})")
print(f"  For this run: {est_waves} waves * {mean_wave/60:.1f} = {est_waves * mean_wave / 60:.0f} min (+/- {est_waves * std_wave / 60 / sqrt(est_waves):.0f} min)")
