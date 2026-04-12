from pathlib import Path
from metavision_core.event_io import EventsIterator
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

index = 1 # set this
category = "paper"   # set this to either "rock" or "paper" or "scissor"
prefix = {"rock": "r", "paper": "p", "scissor": "s"}[category]

base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))

pattern = base / f"{category}/{prefix}_{index}/prophesee_events.raw"

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