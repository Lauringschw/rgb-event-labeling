from pathlib import Path
import numpy as np
from metavision_core.event_io import EventsIterator
from dotenv import load_dotenv
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'sliding_window'))
from dataset_loader_histogram import HistogramDataset

load_dotenv(Path(__file__).parent.parent / '.env')

# == configs ===================================================================
WINDOW_MS           = 50           # fixed window duration for training
TRAIN_OFFSETS_MS    = list(range(0, 101, 10))  # 0,10,...,100ms — 11 per recording
SENSOR_HEIGHT       = 360
SENSOR_WIDTH        = 640
ORIG_HEIGHT         = 720
ORIG_WIDTH          = 1280
BATCH_SIZE          = 500
MAX_RECORDINGS_PER_GESTURE = 270   # same as thesis baseline

RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR"))
DIR            = os.getenv("DIR")
SLIDING_DIR    = Path(os.getenv("SLIDING_DIR"))

# save to separate subfolder so thesis pipeline is untouched
FIXED_DIR = Path(os.getenv("SLIDING_DIR_T7")) / "histogram_fixed"
FIXED_DIR.mkdir(parents=True, exist_ok=True)

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
    on_mask  = p == 1
    off_mask = ~on_mask
    np.add.at(histogram[0], (y[on_mask],  x[on_mask]),  1)
    np.add.at(histogram[1], (y[off_mask], x[off_mask]), 1)
    return histogram


def extract_fixed_window(all_events, t_start_us, duration_ms):
    t_end_us = t_start_us + duration_ms * 1_000
    mask = (all_events['t'] >= t_start_us) & (all_events['t'] < t_end_us)
    return events_to_histogram(all_events[mask])


# == get train recording ids ===================================================

def get_train_recording_ids():
    """Re-derive train recording IDs using same split as thesis baseline."""
    dataset = HistogramDataset()
    data    = dataset.load_samples()
    split   = dataset.get_split(data, test_size=0.20, val_size=0.10)
    return split['recs_train'], split['recs_val']


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


# == per-recording processing ==================================================

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
        return None, None
    return np.concatenate(chunks), t_initial


# == batch helpers =============================================================

def save_batch(batch_data, batch_labels, batch_rec_ids, batch_num, prefix="train"):
    np.save(FIXED_DIR / f"histogram_fixed_{prefix}_data_batch_{batch_num}.npy",
            np.array(batch_data,    dtype=np.float32))
    np.save(FIXED_DIR / f"histogram_fixed_{prefix}_labels_batch_{batch_num}.npy",
            np.array(batch_labels,  dtype=np.int64))
    np.save(FIXED_DIR / f"histogram_fixed_{prefix}_recids_batch_{batch_num}.npy",
            np.array(batch_rec_ids, dtype=np.int64))
    print(f"  [batch {batch_num}] saved {len(batch_data)} samples")


def merge_and_save(prefix="train"):
    print(f"\nMerging {prefix} batches ...")
    data_files  = sorted(FIXED_DIR.glob(f"histogram_fixed_{prefix}_data_batch_*.npy"),
                         key=lambda p: int(p.stem.split('_')[-1]))
    label_files = sorted(FIXED_DIR.glob(f"histogram_fixed_{prefix}_labels_batch_*.npy"),
                         key=lambda p: int(p.stem.split('_')[-1]))
    recid_files = sorted(FIXED_DIR.glob(f"histogram_fixed_{prefix}_recids_batch_*.npy"),
                         key=lambda p: int(p.stem.split('_')[-1]))
    if not data_files:
        print(f"No {prefix} batches found.")
        return
    all_data   = np.concatenate([np.load(f) for f in data_files])
    all_labels = np.concatenate([np.load(f) for f in label_files])
    all_recids = np.concatenate([np.load(f) for f in recid_files])
    np.save(FIXED_DIR / f"histogram_fixed_{prefix}_data.npy",   all_data)
    np.save(FIXED_DIR / f"histogram_fixed_{prefix}_labels.npy",  all_labels)
    np.save(FIXED_DIR / f"histogram_fixed_{prefix}_recids.npy",  all_recids)
    print(f"  {prefix}: {len(all_data):,} samples saved")
    print(f"  rock={np.sum(all_labels==0)}, paper={np.sum(all_labels==1)}, scissor={np.sum(all_labels==2)}")
    for f in data_files + label_files + recid_files:
        f.unlink()


# == main ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FIXED-TIME TRAINING EXTRACTION — Histogram")
    print(f"Window: {WINDOW_MS}ms | Offsets: {TRAIN_OFFSETS_MS}")
    print("=" * 60)

    print("\nDeriving train/val splits from thesis baseline ...")
    recs_train, recs_val = get_train_recording_ids()
    rec_id_to_info = build_rec_id_map()
    print(f"Train recordings: {len(recs_train)}")
    print(f"Val recordings:   {len(recs_val)}")

    for split_name, rec_ids in [("train", recs_train), ("val", recs_val)]:
        print(f"\n--- Extracting {split_name} set ---")
        batch_data, batch_labels, batch_rec_ids = [], [], []
        batch_num  = 0
        processed  = 0
        failed     = 0

        for rec_id in sorted(rec_ids):
            if rec_id not in rec_id_to_info:
                failed += 1
                continue

            gesture, folder = rec_id_to_info[rec_id]
            label = GESTURE_TO_LABEL[gesture]

            all_events, t_initial = load_recording_events(folder)
            if all_events is None:
                failed += 1
                continue

            for offset_ms in TRAIN_OFFSETS_MS:
                t_start = t_initial + offset_ms * 1_000
                hist    = extract_fixed_window(all_events, t_start, WINDOW_MS)
                batch_data.append(hist)
                batch_labels.append(label)
                batch_rec_ids.append(rec_id)

                if len(batch_data) >= BATCH_SIZE:
                    save_batch(batch_data, batch_labels, batch_rec_ids, batch_num, split_name)
                    batch_data, batch_labels, batch_rec_ids = [], [], []
                    batch_num += 1

            processed += 1
            if processed % 50 == 0:
                print(f"  {processed}/{len(rec_ids)} recordings processed")

        if batch_data:
            save_batch(batch_data, batch_labels, batch_rec_ids, batch_num, split_name)

        merge_and_save(split_name)
        print(f"{split_name.upper()}: {processed} ok, {failed} failed")

    print(f"\nAll files saved to: {FIXED_DIR}")
    print("Next step: run train_histogram_fixed.py")