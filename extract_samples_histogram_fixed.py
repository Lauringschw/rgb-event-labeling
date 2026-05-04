from pathlib import Path
import numpy as np
from metavision_core.event_io import EventsIterator
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / '.env')

# == configs ===================================================================
WINDOW_MS                  = 50
TRAIN_OFFSETS_MS           = list(range(0, 101, 10))  # 0,10,...,100ms — 11 offsets
SENSOR_HEIGHT              = 360
SENSOR_WIDTH               = 640
ORIG_HEIGHT                = 720
ORIG_WIDTH                 = 1280
BATCH_SIZE                 = 500
MAX_RECORDINGS_PER_GESTURE = 270

RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR"))
DIR            = os.getenv("DIR")
OUTPUT_DIR     = Path(os.getenv("OUTPUT_DIR"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GESTURE_TO_LABEL = {'rock': 0, 'paper': 1, 'scissor': 2}


# == representation ============================================================

def events_to_histogram(events, height=SENSOR_HEIGHT, width=SENSOR_WIDTH,
                        orig_height=ORIG_HEIGHT, orig_width=ORIG_WIDTH):
    histogram = np.zeros((2, height, width), dtype=np.float32)
    if len(events) == 0:
        return histogram
    x = (events['x'].astype(np.int32) * width  // orig_width)
    y = (events['y'].astype(np.int32) * height // orig_height)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y = x[valid], y[valid]
    p    = events['p'][valid]
    np.add.at(histogram[0], (y[p == 1], x[p == 1]), 1)
    np.add.at(histogram[1], (y[p == 0], x[p == 0]), 1)
    return histogram


def extract_fixed_window(all_events, t_start_us, duration_ms):
    t_end_us = t_start_us + duration_ms * 1_000
    mask = (all_events['t'] >= t_start_us) & (all_events['t'] < t_end_us)
    return events_to_histogram(all_events[mask])


# == split =====================================================================

def get_splits():
    """
    Derive 70/10/20 recording-level split directly from folder structure.
    No dependency on any previously extracted dataset.
    Seeds: 42 (test), 123 (val).
    """
    base = RECORDINGS_DIR / DIR
    all_ids    = []
    all_labels = []
    rec_id     = 0

    for gesture in GESTURE_TO_LABEL:
        prefix = gesture[0]
        label  = GESTURE_TO_LABEL[gesture]
        for i in range(1, MAX_RECORDINGS_PER_GESTURE + 1):
            folder = base / gesture / f"{prefix}_{i}"
            if not folder.exists():
                break
            all_ids.append(rec_id)
            all_labels.append(label)
            rec_id += 1

    all_ids    = np.array(all_ids)
    all_labels = np.array(all_labels)

    recs_temp, recs_test, _, _ = train_test_split(
        all_ids, all_labels,
        test_size=0.20, random_state=42, stratify=all_labels)

    temp_labels = np.array([all_labels[all_ids == r][0] for r in recs_temp])

    recs_train, recs_val, _, _ = train_test_split(
        recs_temp, temp_labels,
        test_size=0.125, random_state=123, stratify=temp_labels)

    return recs_train, recs_val, recs_test


def build_rec_id_map():
    base = RECORDINGS_DIR / DIR
    mapping  = {}
    rec_id   = 0
    for gesture in GESTURE_TO_LABEL:
        prefix = gesture[0]
        for i in range(1, MAX_RECORDINGS_PER_GESTURE + 1):
            folder = base / gesture / f"{prefix}_{i}"
            if not folder.exists():
                break
            mapping[rec_id] = (gesture, folder)
            rec_id += 1
    return mapping


# == event loading =============================================================

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


# == batch helpers =============================================================

def save_batch(data, labels, recids, batch_num, prefix):
    np.save(OUTPUT_DIR / f"{prefix}_data_batch_{batch_num}.npy",   np.array(data,   dtype=np.float32))
    np.save(OUTPUT_DIR / f"{prefix}_labels_batch_{batch_num}.npy", np.array(labels, dtype=np.int64))
    np.save(OUTPUT_DIR / f"{prefix}_recids_batch_{batch_num}.npy", np.array(recids, dtype=np.int64))
    print(f"  [batch {batch_num}] {len(data)} samples")


def merge(prefix):
    data_files  = sorted(OUTPUT_DIR.glob(f"{prefix}_data_batch_*.npy"),   key=lambda p: int(p.stem.split('_')[-1]))
    label_files = sorted(OUTPUT_DIR.glob(f"{prefix}_labels_batch_*.npy"), key=lambda p: int(p.stem.split('_')[-1]))
    recid_files = sorted(OUTPUT_DIR.glob(f"{prefix}_recids_batch_*.npy"), key=lambda p: int(p.stem.split('_')[-1]))
    if not data_files:
        return
    d = np.concatenate([np.load(f) for f in data_files])
    l = np.concatenate([np.load(f) for f in label_files])
    r = np.concatenate([np.load(f) for f in recid_files])
    np.save(OUTPUT_DIR / f"{prefix}_data.npy",   d)
    np.save(OUTPUT_DIR / f"{prefix}_labels.npy", l)
    np.save(OUTPUT_DIR / f"{prefix}_recids.npy", r)
    print(f"  {prefix}: {len(d):,} samples | rock={np.sum(l==0)}, paper={np.sum(l==1)}, scissor={np.sum(l==2)}")
    for f in data_files + label_files + recid_files:
        f.unlink()


# == extract ===================================================================

def extract_split(rec_ids, mapping, prefix):
    print(f"\n--- {prefix} ({len(rec_ids)} recordings) ---")
    batch_data, batch_labels, batch_recids = [], [], []
    batch_num = 0
    ok = 0
    fail = 0

    for rec_id in sorted(rec_ids):
        if rec_id not in mapping:
            fail += 1
            continue
        gesture, folder = mapping[rec_id]
        label = GESTURE_TO_LABEL[gesture]
        all_events, t_initial = load_recording(folder)
        if all_events is None:
            fail += 1
            continue
        for offset_ms in TRAIN_OFFSETS_MS:
            t_start = t_initial + offset_ms * 1_000
            batch_data.append(extract_fixed_window(all_events, t_start, WINDOW_MS))
            batch_labels.append(label)
            batch_recids.append(rec_id)
            if len(batch_data) >= BATCH_SIZE:
                save_batch(batch_data, batch_labels, batch_recids, batch_num, prefix)
                batch_data, batch_labels, batch_recids = [], [], []
                batch_num += 1
        ok += 1
        if ok % 50 == 0:
            print(f"  {ok}/{len(rec_ids)} done")

    if batch_data:
        save_batch(batch_data, batch_labels, batch_recids, batch_num, prefix)
    merge(prefix)
    print(f"  ok={ok}, failed={fail}")


# == main ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FIXED-TIME EXTRACTION — Histogram")
    print(f"Window: {WINDOW_MS}ms | Offsets: {TRAIN_OFFSETS_MS}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    recs_train, recs_val, recs_test = get_splits()
    mapping = build_rec_id_map()

    print(f"\nSplit: train={len(recs_train)}, val={len(recs_val)}, test={len(recs_test)}")

    # save test IDs for evaluation scripts to use
    np.save(OUTPUT_DIR / "test_recording_ids.npy", recs_test)
    print(f"Test IDs saved → {OUTPUT_DIR}/test_recording_ids.npy")

    extract_split(recs_train, mapping, "train")
    extract_split(recs_val,   mapping, "val")

    print(f"\nDone. Next: run train_histogram_fixed.py")
