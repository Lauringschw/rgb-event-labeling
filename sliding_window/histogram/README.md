# sliding_window/histogram

Four scripts for sliding-window histogram dataset creation and CNN training.

## Order of execution

```
1. extract_samples_histogram.py
2. merge_histogram.py
3. train_histogram.py
```

---

## extract_samples_histogram.py

Extracts sliding-window 2D histogram samples from all recordings.

**Sliding window**  
N = 20,000 events, stride S = 4,000 events (80% overlap)

**Extraction range**  
t_initial -> t_initial + 300ms

**Output** (batch files saved to `SLIDING_DIR`)

```
histogram_data_batch_0.npy          float32  (500, 2, 720, 1280)
histogram_labels_batch_0.npy        int64    (500,)
histogram_recids_batch_0.npy        int64    (500,)
...
histogram_data_batch_N.npy
histogram_labels_batch_N.npy
histogram_recids_batch_N.npy
```

Each batch = 500 samples. Recording ID tracks which recording each sample came from (used for split).

---

## merge_histogram.py

Reads histogram batch files from `SLIDING_DIR`, writes merged dataset to `SLIDING_DIR_T7` using memory-mapped files. Deletes each batch immediately after writing to keep peak disk usage at: merged_so_far + one_batch (~300MB overhead).

**Output** (saved to `SLIDING_DIR_T7`)

```
histogram_data.npy              float32  (N, 2, 720, 1280)
histogram_labels.npy            int64    (N,)
histogram_recording_ids.npy     int64    (N,)
```

All batch files deleted after merge.

---

## dataset_loader_histogram.py

Loads the consolidated histogram dataset and produces recording-level train/val/test splits (70/10/20) to prevent data leakage from sliding windows.

**Split strategy**

1. Get unique (recording_id, gesture_label) pairs
2. Split recording IDs into test vs rest (stratified by gesture)
3. Split rest into train vs val (stratified by gesture)
4. Assign every sample to the set its recording belongs to

**Seeds**  
42 (test split), 123 (val split)

**Returns**

```python
{
    'X_train': ndarray, 'y_train': ndarray,
    'X_val':   ndarray, 'y_val':   ndarray,
    'X_test':  ndarray, 'y_test':  ndarray,
    'recs_train': ndarray,
    'recs_val':   ndarray,
    'recs_test':  ndarray,
}
```

---

## train_histogram.py

Trains a CNN on 2-channel 2D histogram event representations.

**Architecture**

```
Block 1: Conv2d(2->32, k=5, p=2) -> ReLU -> MaxPool2d(2×2)
Block 2: Conv2d(32->64, k=3, p=1) -> ReLU -> MaxPool2d(2×2)
Block 3: Conv2d(64->128, k=3, p=1) -> ReLU -> MaxPool2d(2×2)
Classifier: Flatten -> Linear(128×90×160->256) -> ReLU -> Dropout(0.5) -> Linear(256->3)
```

**Training config**

```
Optimizer  : Adam, lr=0.001
Loss       : CrossEntropyLoss
Batch size : 32
Max epochs : 50
Early stop : patience=10 epochs (no val-acc improvement)
Split      : recording-level 70/10/20, seeds 42 & 123
Best model : saved on best validation accuracy
```

**Output** (saved to `SLIDING_DIR_T7`)

```
model_histogram_best.pth
histogram_training_metrics.txt
```

---

## .env variables required

```
RECORDINGS_DIR=   path to folder containing gesture subfolders
DIR=              subfolder name e.g. trial2
SLIDING_DIR=      where batch files live
SLIDING_DIR_T7=   where merged dataset + model weights are saved
```
