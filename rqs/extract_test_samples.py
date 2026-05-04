import argparse
from pathlib import Path
import numpy as np
from metavision_core.event_io import EventsIterator
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / '.env')

RECORDINGS_DIR             = Path(os.getenv("RECORDINGS_DIR"))
DIR                        = os.getenv("DIR")
OUTPUT_DIR                 = Path(os.getenv("OUTPUT_DIR"))

SENSOR_HEIGHT              = 360
SENSOR_WIDTH               = 640
ORIG_HEIGHT                = 720
ORIG_WIDTH                 = 1280
N_BINS                     = 5    # voxel grid bins
MAX_RECORDINGS_PER_GESTURE = 270

RQ1_DURATIONS_MS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RQ2_OFFSETS_MS   = [0, 20, 40, 60, 80, 100]
RQ2_DURATION_MS  = 30

GESTURE_TO_LABEL = {'rock': 0, 'paper': 1, 'scissor': 2}


# == representations ===========================================================

def to_histogram(events):
    out = np.zeros((2, SENSOR_HEIGHT, SENSOR_WIDTH), dtype=np.float32)
    if len(events) == 0:
        return out
    x = (events['x'].astype(np.int32) * SENSOR_WIDTH  // ORIG_WIDTH)
    y = (events['y'].astype(np.int32) * SENSOR_HEIGHT // ORIG_HEIGHT)
    v = (x >= 0) & (x < SENSOR_WIDTH) & (y >= 0) & (y < SENSOR_HEIGHT)
    x, y, p = x[v], y[v], events['p'][v]
    np.add.at(out[0], (y[p == 1], x[p == 1]), 1)
    np.add.at(out[1], (y[p == 0], x[p == 0]), 1)
    return out


def to_voxel(events):
    out = np.zeros((N_BINS, SENSOR_HEIGHT, SENSOR_WIDTH), dtype=np.float32)
    if len(events) == 0:
        return out
    x = (events['x'].astype(np.int32) * SENSOR_WIDTH  // ORIG_WIDTH)
    y = (events['y'].astype(np.int32) * SENSOR_HEIGHT // ORIG_HEIGHT)
    v = (x >= 0) & (x < SENSOR_WIDTH) & (y >= 0) & (y < SENSOR_HEIGHT)
    x, y = x[v], y[v]
    t = events['t'][v].astype(np.float64)
    p = events['p'][v]
    if len(t) == 0:
        return out
    t_min, t_max = t.min(), t.max()
    t_norm  = np.zeros(len(t)) if t_max == t_min else (t - t_min) / (t_max - t_min)
    bin_idx = np.clip((t_norm * N_BINS).astype(np.int32), 0, N_BINS - 1)
    weights = np.where(p == 1, 1.0, -1.0).astype(np.float32)
    np.add.at(out, (bin_idx, y, x), weights)
    return out


def to_timesurface(events):
    out = np.zeros((1, SENSOR_HEIGHT, SENSOR_WIDTH), dtype=np.float32)
    if len(events) == 0:
        return out
    x = (events['x'].astype(np.int32) * SENSOR_WIDTH  // ORIG_WIDTH)
    y = (events['y'].astype(np.int32) * SENSOR_HEIGHT // ORIG_HEIGHT)
    v = (x >= 0) & (x < SENSOR_WIDTH) & (y >= 0) & (y < SENSOR_HEIGHT)
    x, y = x[v], y[v]
    t = events['t'][v].astype(np.float64)
    if len(t) == 0:
        return out
    t_min, t_max = t.min(), t.max()
    t_norm = np.ones(len(t)) if t_max == t_min else (t - t_min) / (t_max - t_min)
    order  = np.argsort(t)
    out[0, y[order], x[order]] = t_norm[order].astype(np.float32)
    return out


REPR_FN = {
    'histogram':   to_histogram,
    'voxel':       to_voxel,
    'timesurface': to_timesurface,
}


# == helpers ===================================================================

def build_rec_id_map():
    base    = RECORDINGS_DIR / DIR
    mapping = {}
    rec_id  = 0
    for gesture in GESTURE_TO_LABEL:
        prefix = gesture[0]
        for i in range(1, MAX_RECORDINGS_PER_GESTURE + 1):
            folder = base / gesture / f"{prefix}_{i}"
            if not folder.exists():
                break
            mapping[rec_id] = (gesture, folder)
            rec_id += 1
    return mapping


def load_recording(folder: Path):
    labels_file = folder / "labels.npy"
    raw_file    = folder / "prophesee_events.raw"
    if not labels_file.exists() or not raw_file.exists():
        return None, None
    labels    = np.load(labels_file, allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']
    chunks = [ev for ev in EventsIterator(str(raw_file))]
    if not chunks:
        return None, None
    return np.concatenate(chunks), t_initial


def extract_window(all_events, t_start_us, duration_ms, repr_fn):
    t_end_us = t_start_us + duration_ms * 1_000
    mask = (all_events['t'] >= t_start_us) & (all_events['t'] < t_end_us)
    return repr_fn(all_events[mask])


# == main ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repr', required=True,
                        choices=['histogram', 'voxel', 'timesurface'],
                        help='Event representation to extract')
    args = parser.parse_args()

    repr_fn  = REPR_FN[args.repr]
    TEST_DIR = OUTPUT_DIR / "test_samples" / args.repr
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(f"TEST EXTRACTION — {args.repr} (RQ1 + RQ2)")
    print("=" * 55)

    test_ids_path = OUTPUT_DIR / "test_recording_ids.npy"
    if not test_ids_path.exists():
        raise FileNotFoundError(
            f"Missing: {test_ids_path}\n"
            f"Run extract_samples_histogram_fixed.py first to generate split.")

    test_rec_ids   = np.load(test_ids_path)
    rec_id_to_info = build_rec_id_map()
    print(f"Test recordings: {len(test_rec_ids)}\n")

    rq1_data, rq1_labels, rq1_durations, rq1_recids = [], [], [], []
    rq2_data, rq2_labels, rq2_offsets,  rq2_recids  = [], [], [], []

    for idx, rec_id in enumerate(sorted(test_rec_ids)):
        if rec_id not in rec_id_to_info:
            print(f"  !! rec_id {rec_id} not found")
            continue

        gesture, folder = rec_id_to_info[rec_id]
        label = GESTURE_TO_LABEL[gesture]
        print(f"[{idx+1}/{len(test_rec_ids)}] {gesture}/{folder.name}")

        all_events, t_initial = load_recording(folder)
        if all_events is None:
            continue

        for dur_ms in RQ1_DURATIONS_MS:
            rq1_data.append(extract_window(all_events, t_initial, dur_ms, repr_fn))
            rq1_labels.append(label)
            rq1_durations.append(dur_ms)
            rq1_recids.append(rec_id)

        for off_ms in RQ2_OFFSETS_MS:
            t_start = t_initial + off_ms * 1_000
            rq2_data.append(extract_window(all_events, t_start, RQ2_DURATION_MS, repr_fn))
            rq2_labels.append(label)
            rq2_offsets.append(off_ms)
            rq2_recids.append(rec_id)

        print(f"  RQ1: {len(RQ1_DURATIONS_MS)} | RQ2: {len(RQ2_OFFSETS_MS)}")

    np.save(TEST_DIR / "rq1_data.npy",         np.array(rq1_data,     dtype=np.float32))
    np.save(TEST_DIR / "rq1_labels.npy",        np.array(rq1_labels,   dtype=np.int64))
    np.save(TEST_DIR / "rq1_durations_ms.npy",  np.array(rq1_durations,dtype=np.int64))
    np.save(TEST_DIR / "rq1_recording_ids.npy", np.array(rq1_recids,   dtype=np.int64))
    np.save(TEST_DIR / "rq2_data.npy",          np.array(rq2_data,    dtype=np.float32))
    np.save(TEST_DIR / "rq2_labels.npy",         np.array(rq2_labels,  dtype=np.int64))
    np.save(TEST_DIR / "rq2_offsets_ms.npy",     np.array(rq2_offsets, dtype=np.int64))
    np.save(TEST_DIR / "rq2_recording_ids.npy",  np.array(rq2_recids,  dtype=np.int64))

    print(f"\nSaved to {TEST_DIR}")
    print(f"RQ1: {np.array(rq1_data).shape}")
    print(f"RQ2: {np.array(rq2_data).shape}")
    print(f"Next: python3 evaluate.py --repr {args.repr}")