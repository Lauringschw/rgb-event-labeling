from pathlib import Path
from metavision_core.event_io import EventsIterator
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

def inspect_recording(category: str, index: int):
    """Print diagnostic info about event camera recording."""
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    prefix = {"rock": "r", "paper": "p", "scissor": "s"}[category]
    
    folder = base / category / f"{prefix}_{index}"
    raw_files = sorted(folder.glob("prophesee_events*.raw"))
    
    if not raw_files:
        raise FileNotFoundError(f"No .raw files in {folder}")
    
    print(f"Inspecting: {category}/{prefix}_{index}")
    print(f"File: {raw_files[0].name}\n")
    
    event_count = 0
    first_t = None
    last_t = None
    
    for ev_batch in EventsIterator(str(raw_files[0])):
        if first_t is None:
            first_t = ev_batch["t"][0]
        last_t = ev_batch["t"][-1]
        event_count += len(ev_batch)
    
    duration_s = (last_t - first_t) / 1e6
    event_rate = event_count / duration_s / 1e6  # Mev/s
    
    print(f"Total events: {event_count:,}")
    print(f"Duration: {duration_s:.3f}s")
    print(f"Event rate: {event_rate:.2f} Mev/s")
    print(f"First timestamp: {first_t} µs")
    print(f"Last timestamp: {last_t} µs")

if __name__ == "__main__":
    inspect_recording(category="paper", index=1)