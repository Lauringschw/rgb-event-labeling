from pathlib import Path
import numpy as np
from metavision_core.event_io import EventsIterator
from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'sliding_window'))
from dataset_loader_histogram import HistogramDataset

load_dotenv(Path(__file__).parent.parent.parent / '.env')

# == paths =====================================================================
RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR"))
DIR            = os.getenv("DIR")
SLIDING_DIR    = Path(os.getenv("SLIDING_DIR"))
TEST_DIR       = Path(os.getenv("TEST_DIR", str(SLIDING_DIR / "test_samples")))
TEST_DIR.mkdir(parents=True, exist_ok=True)

# == sensor config =============================================================
SENSOR_HEIGHT = 360
SENSOR_WIDTH  = 640

# == recording config ==========================================================
MAX_RECORDINGS_PER_GESTURE = 270 

# == RQ configs ================================================================
RQ1_DURATIONS_MS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RQ2_OFFSETS_MS  = [0, 20, 40, 60, 80, 100]
RQ2_DURATION_MS = 30

GESTURE_TO_LABEL = {'rock': 0, 'paper': 1, 'scissor': 2}


# == representation ============================================================

def events_to_histogram(events, height=SENSOR_HEIGHT, width=SENSOR_WIDTH,
                        orig_height=720, orig_width=1280):
    histogram = np.zeros((2, height, width), dtype=np.float32)
    if len(events) == 0:
        return histogram

    x = (events['x'].astype(np.int32) * width  // orig_width)
    y = (events['y'].astype(np.int32) * height // orig_height)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y = x[valid], y[valid]
    p    = events['p'][valid]

    on_mask  = p == 1
    off_mask = ~on_mask
    np.add.at(histogram[0], (y[on_mask],  x[on_mask]),  1)
    np.add.at(histogram[1], (y[off_mask], x[off_mask]), 1)
    return histogram


# == event loading =============================================================

def load_recording_events(folder: Path):
    """
    Load all events and t_initial from a recording folder.
    Returns (all_events, t_initial_us) or (None, None) on failure.
    """
    labels_file = folder / "labels.npy"
    raw_file    = folder / "prophesee_events.raw"

    if not labels_file.exists() or not raw_file.exists():
        print(f"  !! Missing files in {folder.name}")
        return None, None

    labels    = np.load(labels_file, allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']

    mv_iterator = EventsIterator(str(raw_file))
    chunks = [ev for ev in mv_iterator]
    if not chunks:
        print(f"  !! No events in {folder.name}")
        return None, None

    all_events = np.concatenate(chunks)
    return all_events, t_initial


def extract_fixed_time_window(all_events, t_start_us, duration_ms):
    """
    Extract events in [t_start_us, t_start_us + duration_ms*1000).
    Returns histogram array (2, H, W).
    """
    t_end_us = t_start_us + duration_ms * 1_000
    mask     = (all_events['t'] >= t_start_us) & (all_events['t'] < t_end_us)
    return events_to_histogram(all_events[mask])


# == get test recording IDs ====================================================

def get_test_recording_ids():
    """
    Re-derive test recording IDs using the same deterministic split
    as dataset_loader_histogram.py (seeds 42 & 123, 70/10/20 split).
    Requires histogram_recording_ids.npy to exist.
    """
    dataset = HistogramDataset()
    data    = dataset.load_samples()
    split   = dataset.get_split(data, test_size=0.20, val_size=0.10)
    return split['recs_test']


# == main extraction ===========================================================

def extract_rq1_rq2(test_rec_ids):
    """
    For each test recording:
      RQ1: extract 10 fixed-time windows of varying duration from t_initial
      RQ2: extract 6 fixed-30ms windows at varying offsets from t_initial
    """
    base = RECORDINGS_DIR / DIR

    # accumulators
    rq1_data, rq1_labels, rq1_durations, rq1_recids = [], [], [], []
    rq2_data, rq2_labels, rq2_offsets,  rq2_recids  = [], [], [], []

    # we need a mapping: recording_id -> (gesture, folder)
    # reconstruct in same order as extract_samples_histogram.py
    recording_id = 0
    rec_id_to_info = {}

    for gesture in GESTURE_TO_LABEL:
        prefix = gesture[0]
        i = 1
        while i <= MAX_RECORDINGS_PER_GESTURE:  # add this cap
            folder = base / gesture / f"{prefix}_{i}"
            if not folder.exists():
                break
            rec_id_to_info[recording_id] = (gesture, folder)
            recording_id += 1
            i += 1


    print(f"\nTotal recordings mapped: {recording_id}")
    print(f"Test recordings: {len(test_rec_ids)}\n")

    for idx, rec_id in enumerate(sorted(test_rec_ids)):
        if rec_id not in rec_id_to_info:
            print(f"  !! rec_id {rec_id} not found — skipping")
            continue

        gesture, folder = rec_id_to_info[rec_id]
        label = GESTURE_TO_LABEL[gesture]
        print(f"[{idx+1}/{len(test_rec_ids)}] {gesture}/{folder.name}  (rec_id={rec_id})")

        all_events, t_initial = load_recording_events(folder)
        if all_events is None:
            print(f"  !! Skipping")
            continue

        # == RQ1: vary duration, fixed start = t_initial ======================
        for dur_ms in RQ1_DURATIONS_MS:
            hist = extract_fixed_time_window(all_events, t_initial, dur_ms)
            rq1_data.append(hist)
            rq1_labels.append(label)
            rq1_durations.append(dur_ms)
            rq1_recids.append(rec_id)

        # == RQ2: fixed 30ms duration, vary start offset =======================
        for offset_ms in RQ2_OFFSETS_MS:
            t_start = t_initial + offset_ms * 1_000
            hist    = extract_fixed_time_window(all_events, t_start, RQ2_DURATION_MS)
            rq2_data.append(hist)
            rq2_labels.append(label)
            rq2_offsets.append(offset_ms)
            rq2_recids.append(rec_id)

        print(f"  => RQ1: {len(RQ1_DURATIONS_MS)} samples | RQ2: {len(RQ2_OFFSETS_MS)} samples")

    return (
        np.array(rq1_data,     dtype=np.float32),
        np.array(rq1_labels,   dtype=np.int64),
        np.array(rq1_durations,dtype=np.int64),
        np.array(rq1_recids,   dtype=np.int64),
        np.array(rq2_data,     dtype=np.float32),
        np.array(rq2_labels,   dtype=np.int64),
        np.array(rq2_offsets,  dtype=np.int64),
        np.array(rq2_recids,   dtype=np.int64),
    )


# == entry point ===============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST SAMPLE EXTRACTION — Histogram (RQ1 + RQ2 + RQ3)")
    print("=" * 60)

    # get test recording IDs from deterministic split
    print("\nDeriving test recording IDs from training split ")
    test_rec_ids = get_test_recording_ids()
    print(f"Test set: {len(test_rec_ids)} recordings")

    # extract
    (rq1_data, rq1_labels, rq1_durations, rq1_recids,
     rq2_data, rq2_labels, rq2_offsets,  rq2_recids) = extract_rq1_rq2(test_rec_ids)

    # save RQ1
    np.save(TEST_DIR / "rq1_data.npy",           rq1_data)
    np.save(TEST_DIR / "rq1_labels.npy",          rq1_labels)
    np.save(TEST_DIR / "rq1_durations_ms.npy",    rq1_durations)
    np.save(TEST_DIR / "rq1_recording_ids.npy",   rq1_recids)

    # save RQ2
    np.save(TEST_DIR / "rq2_data.npy",            rq2_data)
    np.save(TEST_DIR / "rq2_labels.npy",           rq2_labels)
    np.save(TEST_DIR / "rq2_offsets_ms.npy",       rq2_offsets)
    np.save(TEST_DIR / "rq2_recording_ids.npy",    rq2_recids)

    print(f"\n{'='*60}")
    print("SAVED")
    print(f"{'='*60}")
    print(f"RQ1: {rq1_data.shape}  ->  {TEST_DIR / 'rq1_data.npy'}")
    print(f"RQ2: {rq2_data.shape}  ->  {TEST_DIR / 'rq2_data.npy'}")
    print(f"\nNext step: run evaluate_histogram.py")