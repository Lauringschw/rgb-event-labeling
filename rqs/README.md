# rqs/histogram

Two scripts for evaluating the trained histogram CNN on all three research questions.

## Order of execution

```
1. extract_test_samples_histogram.py
2. evaluate_histogram.py
```

---

## extract_test_samples_histogram.py

Extracts fixed-time-window test samples from raw recordings for RQ1 and RQ2.

**Input**

- Raw recordings from `RECORDINGS_DIR/DIR/`
- `histogram_recording_ids.npy` from training (to derive test split)

**How test recordings are identified**
Re-derives the recording-level split using the same seeds as training (42, 123)
and 70/10/20 proportions. Guarantees zero overlap with training data.

**RQ1 — Window Length Effect**
For each test recording: 10 windows of varying duration starting at t_initial.
Durations: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ms

**RQ2 — Temporal Landmark Effect**
For each test recording: 6 windows of fixed 30ms duration at varying offsets from t_initial.
Offsets: 0, 20, 40, 60, 80, 100 ms

**RQ3**
No extraction needed. evaluate_histogram.py reuses the RQ1 30ms window.

**Output** (saved to `TEST_DIR`)

```
rq1_data.npy              float32  (N_test×10, 2, 720, 1280)
rq1_labels.npy            int64    (N_test×10,)
rq1_durations_ms.npy      int64    (N_test×10,)   which duration each row is
rq1_recording_ids.npy     int64    (N_test×10,)

rq2_data.npy              float32  (N_test×6, 2, 720, 1280)
rq2_labels.npy            int64    (N_test×6,)
rq2_offsets_ms.npy        int64    (N_test×6,)    which offset each row is
rq2_recording_ids.npy     int64    (N_test×6,)
```

---

## evaluate_histogram.py

Loads the trained model and test samples, runs inference, reports accuracy.

**Input**

- `model_histogram_best.pth` from `SLIDING_DIR_T7`
- Test sample files from `TEST_DIR` (produced by extract script above)

**RQ1 output**
Accuracy per window duration (10 values). Reported overall and per class.

**RQ2 output**
Accuracy per temporal offset (6 values). Reported overall and per class.

**RQ3 output**
Histogram accuracy at τ=0, Δt=30ms, sliced from RQ1 results.
Saved separately for cross-representation comparison after voxel and
time surface models are also evaluated.

**Output** (saved to `TEST_DIR/results/`)

```
rq1_accuracies.npy            float64  (10,)   one value per duration
rq1_durations_ms.npy          int64    (10,)
rq1_results.txt               human-readable table

rq2_accuracies.npy            float64  (6,)    one value per offset
rq2_offsets_ms.npy            int64    (6,)
rq2_results.txt               human-readable table

rq3_histogram_acc_30ms.npy    float64  (1,)    for RQ3 comparison
rq3_histogram_result.txt
```

---

## .env variables required

```
RECORDINGS_DIR=   path to folder containing gesture subfolders
DIR=              subfolder name e.g. trial2
SLIDING_DIR=      where batch files and merged training data live
SLIDING_DIR_T7=   where model weights are saved
TEST_DIR=         where test samples and results are saved
```
