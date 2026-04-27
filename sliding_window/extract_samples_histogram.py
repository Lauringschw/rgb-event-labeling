from pathlib import Path
import numpy as np
from metavision_core.event_io import EventsIterator
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

# ======== Configuration ========
WINDOW_SIZE_EVENTS = 50000  # events per sample
STRIDE_EVENTS = 10000       # slide by 10k events (80% overlap)
SENSOR_HEIGHT = 720
SENSOR_WIDTH = 1280

RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR"))
DIR = os.getenv("DIR")
SLIDING_DIR = Path(os.getenv("SLIDING_DIR"))
SLIDING_DIR.mkdir(parents=True, exist_ok=True)

def events_to_histogram(events, height=SENSOR_HEIGHT, width=SENSOR_WIDTH):
    """Convert events to 2D histogram (2 channels: ON/OFF)"""
    histogram = np.zeros((2, height, width), dtype=np.float32)
    
    if len(events) == 0:
        return histogram
    
    # separate ON (p=1) and OFF (p=0) events
    on_events = events[events['p'] == 1]
    off_events = events[events['p'] == 0]
    
    # accumulate counts
    for evt in on_events:
        if 0 <= evt['y'] < height and 0 <= evt['x'] < width:
            histogram[0, evt['y'], evt['x']] += 1
    
    for evt in off_events:
        if 0 <= evt['y'] < height and 0 <= evt['x'] < width:
            histogram[1, evt['y'], evt['x']] += 1
    
    return histogram


def extract_sliding_windows(events, window_size=WINDOW_SIZE_EVENTS, stride=STRIDE_EVENTS):
    """Extract multiple samples from event stream using sliding window."""
    samples = []
    n_events = len(events)
    
    if n_events < window_size:
        print(f"    Warning: Only {n_events} events, less than window size {window_size}")
        return []
    
    # slide window from start to end
    for start_idx in range(0, n_events - window_size + 1, stride):
        end_idx = start_idx + window_size
        window_events = events[start_idx:end_idx]
        
        histogram = events_to_histogram(window_events)
        samples.append(histogram)
    
    return samples


def process_recording(folder: Path):
    """Extract sliding window samples from a single recording."""
    
    # Load required files
    labels_file = folder / "labels.npy"
    raw_file = folder / "prophesee_events.raw"
    
    if not all(f.exists() for f in [labels_file, raw_file]):
        print(f"  !! Missing required files")
        return None
    
    # Load labels
    labels = np.load(labels_file, allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']  # microseconds
    
    # Define extraction range: from t_initial to +300ms
    t_start = t_initial
    t_end = t_initial + 300_000  # 300ms in microseconds
    
    # Load events using EventsIterator
    mv_iterator = EventsIterator(str(raw_file))
    all_events = []
    
    for events in mv_iterator:
        all_events.append(events)
    
    # Concatenate all events
    if len(all_events) == 0:
        print(f"  !! No events in file")
        return None
    
    all_events = np.concatenate(all_events)
    
    # Filter events to our time range
    mask = (all_events['t'] >= t_start) & (all_events['t'] < t_end)
    events = all_events[mask]
    
    if len(events) == 0:
        print(f"  !! No events in range [{t_start}, {t_end})")
        return None
    
    # Extract sliding window samples
    samples = extract_sliding_windows(events)
    
    print(f"  - Extracted {len(samples)} samples from {len(events)} events")
    return samples


if __name__ == "__main__":
    base = RECORDINGS_DIR / DIR
    gestures = ['rock', 'paper', 'scissor']
    
    all_samples = []
    all_labels = []
    
    gesture_to_label = {'rock': 0, 'paper': 1, 'scissor': 2}
    
    total_processed = 0
    total_failed = 0
    
    for gesture in gestures:
        prefix = gesture[0]
        gesture_processed = 0
        gesture_samples = 0
        
        i = 1
        while True:
            folder = base / gesture / f"{prefix}_{i}"
            
            if not folder.exists():
                break
            
            print(f"\n{gesture}/{prefix}_{i}")
            
            samples = process_recording(folder)
            
            if samples is not None and len(samples) > 0:
                # Add all samples from this recording
                label = gesture_to_label[gesture]
                for sample in samples:
                    all_samples.append(sample)
                    all_labels.append(label)
                
                gesture_samples += len(samples)
                gesture_processed += 1
                total_processed += 1
            else:
                total_failed += 1
            
            i += 1
        
        print(f"\n{gesture.upper()}: {gesture_processed} recordings, {gesture_samples} samples")
    
    # Save consolidated dataset
    output_data = SLIDING_DIR / "histogram_data.npy"
    output_labels = SLIDING_DIR / "histogram_labels.npy"
    
    np.save(output_data, np.array(all_samples))
    np.save(output_labels, np.array(all_labels))
    
    print(f"\n{'='*50}")
    print(f"TOTAL: {total_processed} recordings → {len(all_samples)} samples")
    print(f"Failed: {total_failed} recordings")
    print(f"\nSaved to:")
    print(f"  - {output_data}")
    print(f"  - {output_labels}")