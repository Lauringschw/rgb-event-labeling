Author: **Laurin Gschwenter**
Year: **2026**
Project: **Bachelor Thesis**

# Research Questions

### RQ1: Window Length Comparison

**Question:** How much temporal context is needed for accurate gesture recognition?

**Method:** Extract events starting at t_initial (gesture onset) with three different durations:

- 20ms window: `[t_initial → t_initial + 20ms]`
- 30ms window: `[t_initial → t_initial + 30ms]`
- 50ms window: `[t_initial → t_initial + 50ms]`

Train separate models on each window length. Compare accuracy vs temporal context trade-off.

### RQ2: Temporal Landmark Comparison

**Question:** Can we recognize gestures before they're fully formed?

**Method:** Extract 50ms event windows starting at four different time points:

- t_initial (0ms after onset): `[t_initial → t_initial + 50ms]`
- t_early (+50ms): `[t_initial + 50ms → t_initial + 100ms]`
- t_mid (+100ms): `[t_initial + 100ms → t_initial + 150ms]`
- t_late (+200ms): `[t_initial + 200ms → t_initial + 250ms]`

Train separate models on each landmark. Compare accuracy vs. prediction latency trade-off.

### RQ3: Event Representation Comparison

**Question:** Which event encoding best captures gesture motion patterns?

**Method:** Encode the same event window (`[t_initial -> t_initial + 50ms]`) in three formats:

- **Histogram:** 2D spatial event count (where motion occurred)
- **Voxel grid:** 3D spatiotemporal bins (motion over time)
- **Time surface:** 2D event recency map (when pixels last fired)

Train separate models (2D CNN for histogram/time_surface, 3D CNN for voxel_grid). Compare which representation yields the highest accuracy.

# Environment Configuration

Set paths in `.env` to match your local folder structure.

```env
THESIS_DIR=/path/to/your/thesis
BACKUP_DIR=/path/to/your/thesis/backup
LABELS_DIR=/path/to/your/thesis/labels
MODEL_DIR=/path/to/your/thesis/models
RECORDINGS_DIR=/path/to/your/thesis/recordings
RESULTS_DIR=/path/to/your/thesis/results
SAMPLES_DIR=/path/to/your/thesis/samples

DIR=test_2
```

# Workflow

1. **Data collection** → record 3,000 gestures (1,000 per class: rock, paper, scissors)
2. **Sync extraction** → run `rqs_shared/extract_sync_timestamp.py` on each recording to generate RGB<->DVS timestamp mapping
3. **Labeling** → run `rqs_shared/label_tool.py` on each recording to manually mark GO signal and `t_initial` (gesture onset)
4. **Sample generation** → run `rq1_window_length/extract_samples_rq1.py`, `rq2_temporal_landmark/extract_samples_rq2.py`, and `rq3_representation/extract_samples_rq3.py` on all recordings to create training data
5. **Training** → run `rq1_window_length/train_model_rq1.py`, `rq2_temporal_landmark/train_model_rq2.py`, and `rq3_representation/train_model_rq3.py` to train CNNs and evaluate results

# Table of Files

| File                                           | Purpose                                                                                            | Run when?                                       |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `rqs_shared/extract_sync_timestamp.py`         | Extract RGB<->DVS sync timestamps from trigger channel                                             | Once per recording (before labeling)            |
| `rqs_shared/label_tool.py`                     | Manual labeling GUI to mark GO signal and `t_initial`                                              | Once per recording (after sync extraction)      |
| `rq1_window_length/extract_samples_rq1.py`     | Generate event samples with different window lengths (20ms, 30ms, 50ms)                            | Once per recording (after labeling)             |
| `rq2_temporal_landmark/extract_samples_rq2.py` | Generate event samples at different temporal landmarks (`t_initial`, `t_early`, `t_mid`, `t_late`) | Once per recording (after labeling)             |
| `rq3_representation/extract_samples_rq3.py`    | Generate event samples in different representations (histogram, voxel_grid, time_surface)          | Once per recording (after labeling)             |
| `rqs_shared/dataset_loader.py`                 | Load and split generated samples into train/val/test sets                                          | Imported by training scripts (not run directly) |
| `rq1_window_length/train_model_rq1.py`         | Train CNNs to compare window lengths -> answers RQ1                                                | After all RQ1 samples generated                 |
| `rq2_temporal_landmark/train_model_rq2.py`     | Train CNNs to compare temporal landmarks -> answers RQ2                                            | After all RQ2 samples generated                 |
| `rq3_representation/train_model_rq3.py`        | Train CNNs to compare event representations -> answers RQ3                                         | After all RQ3 samples generated                 |

---

# Output Files

| Generated File                | Contains                                                                    | Created by                                     |
| ----------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------- |
| `basler_frame_timestamps.npy` | RGB frame timestamps in DVS time (sync mapping)                             | `rqs_shared/extract_sync_timestamp.py`         |
| `labels.npy`                  | GO frame, GO timestamp, `t_initial` frame, `t_initial` timestamp            | `rqs_shared/label_tool.py`                     |
| `event_samples_rq1.npy`       | 3 event frames per recording (20ms, 30ms, 50ms windows)                     | `rq1_window_length/extract_samples_rq1.py`     |
| `event_samples_rq2.npy`       | 4 event frames per recording (`t_initial`, `t_early`, `t_mid`, `t_late`)    | `rq2_temporal_landmark/extract_samples_rq2.py` |
| `event_samples_rq3.npy`       | 3 event representations per recording (histogram, voxel_grid, time_surface) | `rq3_representation/extract_samples_rq3.py`    |
| `model_*.pth`                 | Trained CNN model weights                                                   | `rq*_*/train_model_rq*.py`                     |
| `rq*_results.npy`             | Test accuracy, confusion matrices, metrics                                  | `rq*_*/train_model_rq*.py`                     |
