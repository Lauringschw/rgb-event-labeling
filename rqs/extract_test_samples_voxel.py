from pathlib import Path
import numpy as np
from metavision_core.event_io import EventsIterator
from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'sliding_window'))
from dataset_loader_voxel import VoxelDataset

load_dotenv(Path(__file__).parent.parent / '.env')

RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR"))
DIR            = os.getenv("DIR")
SLIDING_DIR    = Path(os.getenv("SLIDING_DIR"))
TEST_DIR       = Path(os.getenv("TEST_DIR", str(SLIDING_DIR / "test_samples"))) / "voxel"
TEST_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_HEIGHT = 360
SENSOR_WIDTH  = 640
ORIG_HEIGHT   = 720
ORIG_WIDTH    = 1280
N_BINS        = 5

RQ1_DURATIONS_MS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RQ2_OFFSETS_MS   = [0, 20, 40, 60, 80, 100]
RQ2_DURATION_MS  = 30

MAX_RECORDINGS_PER_GESTURE = 270  # must match extract_samples_voxel.py

GESTURE_TO_LABEL = {'rock': 0, 'paper': 1, 'scissor': 2}


# == representation ============================================================

def events_to_voxel(events, height=SENSOR_HEIGHT, width=SENSOR_WIDTH,
                    orig_height=ORIG_HEIGHT, orig_width=ORIG_WIDTH, n_bins=N_BINS):
    voxel = np.zeros((n_bins, height, width), dtype=np.float32)
    if len(events) == 0:
        return voxel

    x = (events['x'].astype(np.int32) * width  // orig_width)
    y = (events['y'].astype(np.int32) * height // orig_height)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y = x[valid], y[valid]
    t = events['t'][valid].astype(np.float64)
    p = events['p'][valid]

    if len(t) == 0:
        return voxel

    t_min, t_max = t.min(), t.max()
    if t_max == t_min:
        bin_idx = np.zeros(len(t), dtype=np.int32)
    else:
        t_norm  = (t - t_min) / (t_max - t_min)
        bin_idx = np.clip((t_norm * n_bins).astype(np.int32), 0, n_bins - 1)

    weights = np.where(p == 1, 1.0, -1.0).astype(np.float32)
    np.add.at(voxel, (bin_idx, y, x), weights)
    return voxel


# == helpers ===================================================================

def load_recording_events(folder: Path):
    labels_file = folder / "labels.npy"
    raw_file    = folder / "prophesee_events.raw"
    if not labels_file.exists() or not raw_file.exists():
        print(f"  !! Missing files in {folder.name}")
        return None, None
    labels    = np.load(labels_file, allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']
    chunks = [ev for ev in EventsIterator(str(raw_file))]
    if not chunks:
        print(f"  !! No events in {folder.name}")
        return None, None
    return np.concatenate(chunks), t_initial


def extract_window(all_events, t_start_us, duration_ms):
    t_end_us = t_start_us + duration_ms * 1_000
    mask = (all_events['t'] >= t_start_us) & (all_events['t'] < t_end_us)
    return events_to_voxel(all_events[mask])


def get_test_recording_ids():
    dataset = VoxelDataset()
    data    = dataset.load_samples()
    split   = dataset.get_split(data, test_size=0.20, val_size=0.10)
    return split['recs_test']


def build_rec_id_map():
    base = RECORDINGS_DIR / DIR
    rec_id_to_info = {}
    recording_id = 0
    for gesture in GESTURE_TO_LABEL:
        prefix = gesture[0]
        for i in range(1, MAX_RECORDINGS_PER_GESTURE + 1):
            folder = base / gesture / f"{prefix}_{i}"
            if not folder.exists():
                break
            rec_id_to_info[recording_id] = (gesture, folder)
            recording_id += 1
    return rec_id_to_info


# == main ======================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("TEST SAMPLE EXTRACTION — Voxel (RQ1 + RQ2)")
    print("=" * 55)

    test_rec_ids   = get_test_recording_ids()
    rec_id_to_info = build_rec_id_map()
    print(f"Test recordings: {len(test_rec_ids)}\n")

    rq1_data, rq1_labels, rq1_durations, rq1_recids = [], [], [], []
    rq2_data, rq2_labels, rq2_offsets,  rq2_recids  = [], [], [], []

    for idx, rec_id in enumerate(sorted(test_rec_ids)):
        if rec_id not in rec_id_to_info:
            print(f"  !! rec_id {rec_id} not found — skipping")
            continue

        gesture, folder = rec_id_to_info[rec_id]
        label = GESTURE_TO_LABEL[gesture]
        print(f"[{idx+1}/{len(test_rec_ids)}] {gesture}/{folder.name}")

        all_events, t_initial = load_recording_events(folder)
        if all_events is None:
            continue

        for dur_ms in RQ1_DURATIONS_MS:
            rq1_data.append(extract_window(all_events, t_initial, dur_ms))
            rq1_labels.append(label)
            rq1_durations.append(dur_ms)
            rq1_recids.append(rec_id)

        for off_ms in RQ2_OFFSETS_MS:
            t_start = t_initial + off_ms * 1_000
            rq2_data.append(extract_window(all_events, t_start, RQ2_DURATION_MS))
            rq2_labels.append(label)
            rq2_offsets.append(off_ms)
            rq2_recids.append(rec_id)

        print(f"  ✓ RQ1: {len(RQ1_DURATIONS_MS)} | RQ2: {len(RQ2_OFFSETS_MS)}")

    np.save(TEST_DIR / "rq1_data.npy",          np.array(rq1_data,     dtype=np.float32))
    np.save(TEST_DIR / "rq1_labels.npy",         np.array(rq1_labels,   dtype=np.int64))
    np.save(TEST_DIR / "rq1_durations_ms.npy",   np.array(rq1_durations,dtype=np.int64))
    np.save(TEST_DIR / "rq1_recording_ids.npy",  np.array(rq1_recids,   dtype=np.int64))

    np.save(TEST_DIR / "rq2_data.npy",           np.array(rq2_data,    dtype=np.float32))
    np.save(TEST_DIR / "rq2_labels.npy",          np.array(rq2_labels,  dtype=np.int64))
    np.save(TEST_DIR / "rq2_offsets_ms.npy",      np.array(rq2_offsets, dtype=np.int64))
    np.save(TEST_DIR / "rq2_recording_ids.npy",   np.array(rq2_recids,  dtype=np.int64))

    print(f"\nSaved to {TEST_DIR}")
    print(f"RQ1: {np.array(rq1_data).shape}")
    print(f"RQ2: {np.array(rq2_data).shape}")
    print("RQ3: reuses rq1_data.npy at duration=30ms")