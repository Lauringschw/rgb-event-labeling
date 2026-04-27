from pathlib import Path
import numpy as np
from metavision_core.event_io import EventsIterator
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / '.env')

RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR"))
DIR = os.getenv("DIR")

def count_events_in_window(folder: Path, window_ms: int = 30):
    """Count events in a time window starting from t_initial."""
    
    labels_file = folder / "labels.npy"
    raw_file = folder / "prophesee_events.raw"
    
    if not all(f.exists() for f in [labels_file, raw_file]):
        return None
    
    # Load labels
    labels = np.load(labels_file, allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']
    
    # Define time window
    t_start = t_initial
    t_end = t_initial + (window_ms * 1000)  # convert ms to microseconds
    
    # Load all events
    mv_iterator = EventsIterator(str(raw_file))
    all_events = []
    
    for events in mv_iterator:
        all_events.append(events)
    
    if len(all_events) == 0:
        return None
    
    all_events = np.concatenate(all_events)
    
    # Filter to time window
    mask = (all_events['t'] >= t_start) & (all_events['t'] < t_end)
    events_in_window = all_events[mask]
    
    return len(events_in_window)


if __name__ == "__main__":
    base = RECORDINGS_DIR / DIR
    gestures = ['rock', 'paper', 'scissor']
    
    # Test multiple window sizes
    window_sizes = [20, 30, 50, 80, 100, 120]
    
    for window_ms in window_sizes:
        print(f"\n{'='*60}")
        print(f"Window: {window_ms}ms from t_initial")
        print(f"{'='*60}")
        
        gesture_counts = {g: [] for g in gestures}
        
        for gesture in gestures:
            prefix = gesture[0]
            
            i = 1
            while True:
                folder = base / gesture / f"{prefix}_{i}"
                
                if not folder.exists():
                    break
                
                count = count_events_in_window(folder, window_ms)
                
                if count is not None:
                    gesture_counts[gesture].append(count)
                
                i += 1
        
        # Print statistics for each gesture
        for gesture in gestures:
            counts = gesture_counts[gesture]
            if len(counts) > 0:
                print(f"\n{gesture.upper()}:")
                print(f"  Recordings: {len(counts)}")
                print(f"  Min events: {min(counts):,}")
                print(f"  Max events: {max(counts):,}")
                print(f"  Mean events: {np.mean(counts):,.0f}")
                print(f"  Median events: {np.median(counts):,.0f}")
                print(f"  Std dev: {np.std(counts):,.0f}")