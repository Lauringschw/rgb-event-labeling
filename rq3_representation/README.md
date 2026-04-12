# RQ3: Event Representation Comparison

## Goal

Test which event representation gives the best gesture classification performance.

Representations compared:

1. histogram (2D)
2. voxel_grid (3D)
3. time_surface (2D)

Fixed extraction setting:

1. Landmark: t_initial
2. Window length: 50 ms

## End-to-end flow

```text
labels + events
    -> extract_samples_rq3.py
    -> event_samples_rq3.npy (histogram, voxel_grid, time_surface)
    -> train_model_rq3.py
    -> 3 trained models + rq3_results.npy
```

## Step 1: Build RQ3 samples

Script: extract_samples_rq3.py

What it reads per recording:

1. labels.npy for t_initial timestamp
2. prophesee_events.raw for DVS events

What it creates:

1. event_samples_rq3.npy with three encodings of the same event window:
1. histogram: 2D event-count map
1. voxel_grid: 3D time-binned representation (5 bins)
1. time_surface: 2D latest-event recency map

How it behaves:

1. Iterates all gesture folders and recording subfolders
2. Extracts events from [t_initial, t_initial + 50 ms)
3. Converts that event subset into the three representations

Run:

```bash
python extract_samples_rq3.py
```

## Step 2: Train and compare representations

Script: train_model_rq3.py

What it does:

1. Loads RQ3 samples through GestureDataset
2. For each representation:
3. Builds stratified train/val/test splits
4. Selects model type by input structure
5. Trains and keeps the best checkpoint by validation accuracy
6. Evaluates test accuracy
7. Prints confusion matrix and classification report
8. Saves model weights and summary results

Model assignment:

1. histogram -> 2D CNN
2. voxel_grid -> 3D CNN
3. time_surface -> 2D CNN

Run:

```bash
python train_model_rq3.py
```

## Outputs

| Artifact              | Location              | Purpose                                         |
| --------------------- | --------------------- | ----------------------------------------------- |
| event_samples_rq3.npy | each recording folder | three encoded versions of the same event window |
| rq3_histogram.pth     | MODEL_DIR             | best model for histogram representation         |
| rq3_voxel_grid.pth    | MODEL_DIR             | best model for voxel grid representation        |
| rq3_time_surface.pth  | MODEL_DIR             | best model for time surface representation      |
| rq3_results.npy       | RESULTS_DIR           | representation-level evaluation summary         |

## Recommended run order

1. Ensure recordings are labeled (labels.npy present)
2. Run sample extraction
3. Run representation training
4. Compare test accuracy across the three encodings

## RQ3 answer criterion

RQ3 is answered by the final test-accuracy comparison:

1. Higher test accuracy indicates the more effective event representation for this dataset
2. The saved results file contains all representation-level metrics together
