from pathlib import Path
import numpy as np
import os
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

SLIDING_DIR_T7_TIME = Path(os.getenv("SLIDING_DIR_T7_TIME"))

GESTURE_TO_LABEL = {'rock': 0, 'paper': 1, 'scissor': 2}
LABEL_TO_GESTURE = {v: k for k, v in GESTURE_TO_LABEL.items()}


class HistogramTimeDataset:

    def load_samples(self):
        """
        Load time-based data, labels, and recording_ids from disk.
        Returns dict with keys: data, labels, recording_ids
        """
        data_path   = SLIDING_DIR_T7_TIME / "histogram_time_data.npy"
        labels_path = SLIDING_DIR_T7_TIME / "histogram_time_labels.npy"
        recids_path = SLIDING_DIR_T7_TIME / "histogram_time_recording_ids.npy"

        for p in [data_path, labels_path, recids_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing: {p}\n"
                    f"Run extract_samples_histogram_time.py first."
                )

        data        = np.load(data_path, mmap_mode='r')
        labels      = np.load(labels_path)
        recording_ids = np.load(recids_path)

        print(f"Loaded {len(data)} time-based samples from {SLIDING_DIR_T7_TIME}")
        print(f"Data shape: {data.shape}")
        print(f"Unique recordings: {len(np.unique(recording_ids))}")
        for idx, name in LABEL_TO_GESTURE.items():
            print(f"  {name}: {np.sum(labels == idx)} samples")

        return {
            'data':          data,
            'labels':        labels,
            'recording_ids': recording_ids,
        }

    def get_split(self, dataset, test_size=0.20, val_size=0.10):
        """
        Recording-level stratified split -> 70 / 10 / 20 train / val / test.
        Seeds: 42 for test split, 123 for val split.
        """
        data          = dataset['data']
        labels        = dataset['labels']
        recording_ids = dataset['recording_ids']

        # Build per-recording label
        unique_recs = np.unique(recording_ids)
        rec_labels  = np.array([
            labels[recording_ids == r][0] for r in unique_recs
        ])

        # Split 1: test set (20%)
        recs_temp, recs_test, _, _ = train_test_split(
            unique_recs, rec_labels,
            test_size=test_size,
            random_state=42,
            stratify=rec_labels,
        )

        rec_labels_temp = np.array([
            labels[recording_ids == r][0] for r in recs_temp
        ])

        # Split 2: val set (10% of total)
        adjusted_val = val_size / (1.0 - test_size)
        recs_train, recs_val, _, _ = train_test_split(
            recs_temp, rec_labels_temp,
            test_size=adjusted_val,
            random_state=123,
            stratify=rec_labels_temp,
        )

        # Assign samples to sets
        def mask_for(rec_set):
            return np.isin(recording_ids, rec_set)

        train_mask = mask_for(recs_train)
        val_mask   = mask_for(recs_val)
        test_mask  = mask_for(recs_test)

        n_train = int(np.sum(train_mask))
        n_val   = int(np.sum(val_mask))
        n_test  = int(np.sum(test_mask))

        print(f"\nRecording-level split:")
        print(f"  train: {len(recs_train)} recordings -> {n_train} samples")
        print(f"  val:   {len(recs_val)}   recordings -> {n_val} samples")
        print(f"  test:  {len(recs_test)}  recordings -> {n_test} samples")

        return {
            'recs_train': recs_train,
            'recs_val':   recs_val,
            'recs_test':  recs_test,
        }