from pathlib import Path
import numpy as np
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import MostRecentTimestampBuffer
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

# == configs =====================================================================
WINDOW_DURATION_US = 50_000   # 50ms time window
STRIDE_DURATION_US = 25_000   # 50% overlap (25ms stride)
SENSOR_HEIGHT = 360
SENSOR_WIDTH  = 640
EXTRACTION_RANGE_US = 300_000   # 300 ms total extraction window
BATCH_SIZE         = 500        # samples per batch file
MAX_RECORDINGS_PER_GESTURE = 320

# == paths =====================================================================
RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR"))
DIR            = os.getenv("DIR")
SLIDING_DIR    = Path(os.getenv("SLIDING_DIR_TIME_TS"))  # NEW: time surface output
SLIDING_DIR.mkdir(parents=True, exist_ok=True)

GESTURE_TO_LABEL = {'rock': 0, 'paper': 1, 'scissor': 2}


# == representation using Metavision SDK ========================================

def events_to_timesurface(events, height=SENSOR_HEIGHT, width=SENSOR_WIDTH,
                          orig_height=720, orig_width=1280):
    """
    Convert events to time surface using Metavision SDK's MostRecentTimestampBuffer.
    Time surface = timestamp of most recent event at each pixel, normalized to [0,1].
    """
    if len(events) == 0:
        return np.zeros((1, height, width), dtype=np.float32)

    # Downsample coordinates
    # CRITICAL: Cast to int32 BEFORE multiplication to avoid uint16 overflow
    x = (events['x'].astype(np.int32) * width  // orig_width)
    y = (events['y'].astype(np.int32) * height // orig_height)
    
    # Filter valid coordinates
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    
    # Create downsampled event array with proper dtype
    downsampled = np.zeros(np.sum(valid), dtype=[
        ('x', np.int16),
        ('y', np.int16),
        ('p', np.int16),
        ('t', np.int64)
    ])
    
    downsampled['x'] = x[valid]
    downsampled['y'] = y[valid]
    downsampled['p'] = events['p'][valid]
    downsampled['t'] = events['t'][valid]
    
    if len(downsampled) == 0:
        return np.zeros((1, height, width), dtype=np.float32)
    
    # Use Metavision SDK's MostRecentTimestampBuffer for time surface
    ts_buffer = MostRecentTimestampBuffer(height, width)
    
    # Generate time surface from events
    ts_buffer.generate(downsampled)
    
    # Get numpy array: shape (height, width), dtype int64 (timestamps in microseconds)
    time_surface = ts_buffer.numpy().copy()
    
    # Normalize to [0, 1]
    # Subtract minimum timestamp (oldest event) and divide by range
    t_min = downsampled['t'].min()
    t_max = downsampled['t'].max()
    
    if t_max > t_min:
        # Normalize: (t - t_min) / (t_max - t_min)
        time_surface = (time_surface.astype(np.float32) - t_min) / (t_max - t_min)
    else:
        # All events at same timestamp
        time_surface = np.ones_like(time_surface, dtype=np.float32)
    
    # Pixels with no events remain 0
    mask = (time_surface > 0)
    result = np.zeros((1, height, width), dtype=np.float32)
    result[0][mask] = time_surface[mask]
    
    return result


# == TIME-BASED sliding window =================================================

def extract_time_windows(events, t_start_us, t_end_us):
    """
    Slide a FIXED-TIME window over the event stream.
    Window duration: WINDOW_DURATION_US (50ms)
    Stride:          STRIDE_DURATION_US (25ms = 50% overlap)
    
    Returns list of (1, H, W) time surface arrays.
    """
    samples = []
    
    if len(events) == 0:
        return samples
    
    # Sliding window over TIME (not event count)
    current_t = t_start_us
    
    while current_t + WINDOW_DURATION_US <= t_end_us:
        window_end = current_t + WINDOW_DURATION_US
        
        # Select events in time window
        mask = (events['t'] >= current_t) & (events['t'] < window_end)
        window_events = events[mask]
        
        # Skip if too few events (optional threshold)
        if len(window_events) < 100:  # minimum 100 events
            print(f"      Warning: only {len(window_events)} events in [{current_t}, {window_end}), skipping")
            current_t += STRIDE_DURATION_US
            continue
        
        time_surface = events_to_timesurface(window_events)
        samples.append(time_surface)
        
        # Slide forward by stride
        current_t += STRIDE_DURATION_US
    
    return samples


# == per-recording processing ==================================================

def process_recording(folder: Path):
    """
    Load a single recording, extract TIME-BASED sliding-window time surface samples.
    Returns list of (1, H, W) arrays, or None on failure.
    """
    labels_file = folder / "labels.npy"
    raw_file    = folder / "prophesee_events.raw"

    if not labels_file.exists() or not raw_file.exists():
        print(f"  !! Missing files in {folder.name}")
        return None

    labels    = np.load(labels_file, allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']

    t_start = t_initial
    t_end   = t_initial + EXTRACTION_RANGE_US

    # Load all events using Metavision SDK
    mv_iterator = EventsIterator(str(raw_file))
    chunks = [ev for ev in mv_iterator]
    if not chunks:
        print(f"  !! No events in {folder.name}")
        return None

    all_events = np.concatenate(chunks)

    # Filter to extraction window
    mask   = (all_events['t'] >= t_start) & (all_events['t'] < t_end)
    events = all_events[mask]

    if len(events) == 0:
        print(f"  !! No events in [{t_start}, {t_end}) for {folder.name}")
        return None

    # Extract TIME-BASED windows
    samples = extract_time_windows(events, t_start, t_end)
    
    n_events = len(events)
    duration_ms = (events['t'][-1] - events['t'][0]) / 1000.0
    
    print(f"  -> {len(samples)} samples from {n_events} events ({duration_ms:.1f}ms)")
    return samples


# == batch helpers =============================================================

def save_batch(batch_samples, batch_labels, batch_rec_ids, batch_num):
    np.save(SLIDING_DIR / f"timesurface_time_data_batch_{batch_num}.npy",
            np.array(batch_samples, dtype=np.float32))
    np.save(SLIDING_DIR / f"timesurface_time_labels_batch_{batch_num}.npy",
            np.array(batch_labels, dtype=np.int64))
    np.save(SLIDING_DIR / f"timesurface_time_recids_batch_{batch_num}.npy",
            np.array(batch_rec_ids, dtype=np.int64))
    print(f"  [batch {batch_num}] saved {len(batch_samples)} samples")


# == main ======================================================================

if __name__ == "__main__":
    base = RECORDINGS_DIR / DIR

    batch_samples = []
    batch_labels  = []
    batch_rec_ids = []
    batch_num     = 0

    total_processed = 0
    total_failed    = 0
    total_samples   = 0

    recording_id = 0

    print("=" * 60)
    print("TIME SURFACE EXTRACTION (50ms windows, 50% overlap)")
    print("Using Metavision SDK: MostRecentTimestampBuffer")
    print("=" * 60)
    print(f"Window duration : {WINDOW_DURATION_US / 1000:.1f} ms")
    print(f"Stride          : {STRIDE_DURATION_US / 1000:.1f} ms")
    print(f"Overlap         : {100 * (1 - STRIDE_DURATION_US / WINDOW_DURATION_US):.0f}%")
    print(f"Output dir      : {SLIDING_DIR}\n")

    for gesture in GESTURE_TO_LABEL:
        prefix           = gesture[0]
        label            = GESTURE_TO_LABEL[gesture]
        gesture_samples  = 0
        gesture_ok       = 0

        for i in range(1, MAX_RECORDINGS_PER_GESTURE + 1):
            folder = base / gesture / f"{prefix}_{i}"
            if not folder.exists():
                break

            print(f"\n{gesture}/{prefix}_{i}  (rec_id={recording_id})")
            samples = process_recording(folder)

            if samples:
                for s in samples:
                    batch_samples.append(s)
                    batch_labels.append(label)
                    batch_rec_ids.append(recording_id)

                    if len(batch_samples) >= BATCH_SIZE:
                        save_batch(batch_samples, batch_labels, batch_rec_ids, batch_num)
                        batch_samples, batch_labels, batch_rec_ids = [], [], []
                        batch_num += 1

                gesture_samples += len(samples)
                total_samples   += len(samples)
                gesture_ok      += 1
                total_processed += 1
            else:
                total_failed += 1

            recording_id += 1

        print(f"\n{gesture.upper()}: {gesture_ok} recordings, {gesture_samples} samples")

    # Flush remaining samples
    if batch_samples:
        save_batch(batch_samples, batch_labels, batch_rec_ids, batch_num)

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_processed} recordings -> {total_samples} samples")
    print(f"Failed: {total_failed} recordings")
    
    print(f"Batches saved to: {SLIDING_DIR}")
    print(f"Next step: run merge_timesurface_time.py")