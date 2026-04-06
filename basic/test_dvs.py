from pathlib import Path
from metavision_core.event_io import EventsIterator

index = 1
base = Path("/home/lau/Documents/test_1")

category = "rock"   # "rock" or "paper" or "scissor"
prefix = {"rock": "r", "paper": "p", "scissor": "s"}[category]

pattern = base / f"{category}/{prefix}_{index}" / "recording_2026*.raw"

paths = sorted(pattern.parent.glob(pattern.name))
if not paths:
    raise FileNotFoundError(f"No .raw files matched: {pattern}")

mv_it = EventsIterator(str(paths[0]))

event_count = 0
first_t = None
last_t = None

# ev has 'x', 'y', 't', 'p'
for ev in mv_it:
    if first_t is None:
        first_t = ev["t"][0] 
    last_t = ev["t"][-1]
    event_count += len(ev)

print(f"total events: {event_count}")
print(f"first timestamp: {first_t} µs")
print(f"last timestamp: {last_t} µs")
print(f"duration: {(last_t - first_t) / 1e6:.2f} seconds")