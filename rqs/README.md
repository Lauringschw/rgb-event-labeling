# rqs

Two scripts for extracting test samples and evaluating trained CNNs on all three research questions, for any event representation.

## Order of execution

```
1. extract_test_samples.py --repr <representation>
2. evaluate.py --repr <representation>
```

Supported representations: `histogram`, `voxel`, `timesurface`

---

## extract_test_samples.py

Extracts fixed-time-window test samples from raw recordings for RQ1 and RQ2.

**Input**

- Raw recordings from `RECORDINGS_DIR/DIR/`
- `test_recording_ids.npy` from `OUTPUT_DIR/` (saved by extraction pipeline to define the split)

**How test recordings are identified**
Loads the test recording IDs saved during training extraction. Guarantees zero overlap with training data.

**RQ1 — Window Length Effect**
For each test recording: 10 windows of varying duration starting at t_initial.
Durations: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ms

**RQ2 — Temporal Landmark Effect**
For each test recording: 6 windows of fixed 30ms duration at varying offsets from t_initial.
Offsets: 0, 20, 40, 60, 80, 100 ms

**RQ3**
No extra extraction needed. evaluate.py reuses the RQ1 30ms window.

**Output** (saved to `OUTPUT_DIR/test_samples/<repr>/`)

```
rq1_data.npy              float32  (N_test×10, C, 360, 640)
rq1_labels.npy            int64    (N_test×10,)
rq1_durations_ms.npy      int64    (N_test×10,)
rq1_recording_ids.npy     int64    (N_test×10,)

rq2_data.npy              float32  (N_test×6, C, 360, 640)
rq2_labels.npy            int64    (N_test×6,)
rq2_offsets_ms.npy        int64    (N_test×6,)
rq2_recording_ids.npy     int64    (N_test×6,)
```

Where C = 2 (histogram), 5 (voxel), 1 (timesurface).

---

## evaluate.py

Loads the trained model and test samples, runs inference, reports accuracy.

**Input**

- Model weights from `OUTPUT_DIR/` — filename depends on representation
- Test sample files from `OUTPUT_DIR/test_samples/<repr>/`

**RQ1 output**
Accuracy per window duration (10 values). Reported overall and per class.

**RQ2 output**
Accuracy per temporal offset (6 values). Reported overall and per class.

**RQ3 output**
Accuracy at τ=0, Δt=30ms, sliced from RQ1 results. Saved for cross-representation comparison.

**Output** (saved to `OUTPUT_DIR/results/<repr>/`)

```
rq1_accuracies.npy              float64  (10,)
rq1_durations_ms.npy            int64    (10,)
rq1_results.txt

rq2_accuracies.npy              float64  (6,)
rq2_offsets_ms.npy              int64    (6,)
rq2_results.txt

rq3_<repr>_acc_30ms.npy         float64  (1,)
rq3_<repr>_result.txt
```

---

## .env variables required

```
RECORDINGS_DIR=   path to drive containing gesture recording folders
DIR=              subfolder name e.g. trial2
OUTPUT_DIR=       all outputs go here (train data, model weights, test samples, results)
```
