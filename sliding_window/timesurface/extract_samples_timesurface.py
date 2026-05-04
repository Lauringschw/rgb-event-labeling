from pathlib import Path
import numpy as np
from metavision_core.event_io import EventsIterator
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

# == configs ===================================================================
WINDOW_SIZE_EVENTS  = 20_000
STRIDE_EVENTS       = 4_000        # 80% overlap
SENSOR_HEIGHT       = 360
SENSOR_WIDTH        = 640
ORIG_HEIGHT         = 720
ORIG_WIDTH          = 1280
EXTRACTION_RANGE_US = 300_000      # 300ms
BATCH_SIZE          = 500
MAX_RECORDINGS_PER_GESTURE = 270   # match histogram run

RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR"))
DIR            = os.getenv("DIR")
SLIDING_DIR    = Path(os.getenv("SLIDING_DIR"))
SLIDING_DIR.mkdir(parents=True, exist_ok=True)

GESTURE_TO_LABEL = {'rock': 0, 'paper': 1, 'scissor': 2}


# == representation ============================================================

def events_to_timesurface(events, height=SENSOR_HEIGHT, width=SENSOR_WIDTH,
                           orig_height=ORIG_HEIGHT, orig_width=ORIG_WIDTH):
    """
    Single-channel time surface: (1, height, width)
    Each pixel stores the normalized timestamp of its most recent event.
    Values in [0, 1]. Pixels with no events = 0.
    Following Lagorce et al. 2016 (HOTS).
    """
    surface = np.zeros((1, height, width), dtype=np.float32)
    if len(events) == 0:
        return surface

    x = (events['x'].astype(np.int32) * width  // orig_width)
    y = (events['y'].astype(np.int32) * height // orig_height)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y = x[valid], y[valid]
    t = events['t'][valid].astype(np.float64)

    if len(t) == 0:
        return surface

    t_min, t_max = t.min(), t.max()
    if t_max == t_min:
        t_norm = np.ones(len(t), dtype=np.float32)
    else:
        t_norm = ((t - t_min) / (t_max - t_min)).astype(np.float32)

    # assign most recent event timestamp per pixel
    # iterate in chronological order so later events overwrite earlier ones
    order = np.argsort(t)
    x_sorted = x[order]
    y_sorted = y[order]
    t_sorted = t_norm[order]

    surface[0, y_sorted, x_sorted] = t_sorted
    return surface


# == sliding window ============================================================

def extract_sliding_windows(events):
    samples = []
    n = len(events)
    if n < WINDOW_SIZE_EVENTS:
        print(f"    Warning: only {n} events — skipping")
        return []
    for start in range(0, n - WINDOW_SIZE_EVENTS + 1, STRIDE_EVENTS):
        window = events[start : start + WINDOW_SIZE_EVENTS]
        samples.append(events_to_timesurface(window))
    return samples


# == per-recording processing ==================================================

def process_recording(folder: Path):
    labels_file = folder / "labels.npy"
    raw_file    = folder / "prophesee_events.raw"

    if not labels_file.exists() or not raw_file.exists():
        print(f"  !! Missing files in {folder.name}")
        return None

    labels    = np.load(labels_file, allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']
    t_start   = t_initial
    t_end     = t_initial + EXTRACTION_RANGE_US

    mv_iterator = EventsIterator(str(raw_file))
    chunks = [ev for ev in mv_iterator]
    if not chunks:
        print(f"  !! No events in {folder.name}")
        return None

    all_events = np.concatenate(chunks)
    mask   = (all_events['t'] >= t_start) & (all_events['t'] < t_end)
    events = all_events[mask]

    if len(events) == 0:
        print(f"  !! No events in range for {folder.name}")
        return None

    samples = extract_sliding_windows(events)
    print(f"  -> {len(samples)} samples from {len(events)} events")
    return samples


# == batch helpers =============================================================

def save_batch(batch_samples, batch_labels, batch_rec_ids, batch_num):
    np.save(SLIDING_DIR / f"timesurface_data_batch_{batch_num}.npy",
            np.array(batch_samples, dtype=np.float32))
    np.save(SLIDING_DIR / f"timesurface_labels_batch_{batch_num}.npy",
            np.array(batch_labels, dtype=np.int64))
    np.save(SLIDING_DIR / f"timesurface_recids_batch_{batch_num}.npy",
            np.array(batch_rec_ids, dtype=np.int64))
    print(f"  [batch {batch_num}] saved {len(batch_samples)} samples")


# == main ======================================================================

if __name__ == "__main__":
    base = RECORDINGS_DIR / DIR

    batch_samples, batch_labels, batch_rec_ids = [], [], []
    batch_num       = 0
    total_processed = 0
    total_failed    = 0
    total_samples   = 0
    recording_id    = 0

    for gesture in GESTURE_TO_LABEL:
        prefix          = gesture[0]
        label           = GESTURE_TO_LABEL[gesture]
        gesture_samples = 0
        gesture_ok      = 0

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

    if batch_samples:
        save_batch(batch_samples, batch_labels, batch_rec_ids, batch_num)

    print(f"\n{'='*50}")
    print(f"TOTAL: {total_processed} recordings -> {total_samples} samples")
    print(f"Failed: {total_failed} recordings")
    print(f"Batches saved to: {SLIDING_DIR}")
    print(f"Next step: run merge_timesurface.py")