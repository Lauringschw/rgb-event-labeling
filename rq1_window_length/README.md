# RQ1: Window Length Comparison

## Goal

Test whether event window length affects gesture recognition accuracy.

Window settings compared:

1. 20 ms
2. 30 ms
3. 50 ms

## End-to-end flow

```text
labels + events
      -> extract_samples_rq1.py
      -> event_samples_rq1.npy (20ms, 30ms, 50ms)
      -> train_model_rq1.py
      -> 3 trained models + rq1_results.npy
```

## Step 1: Build RQ1 samples

Script: extract_samples_rq1.py

What it reads per recording:

1. labels.npy for the t_initial timestamp
2. prophesee_events.raw for DVS events

What it creates:

1. event_samples_rq1.npy
2. Contains one 2D event-count frame for each window: 20ms, 30ms, 50ms

How it behaves:

1. Uses only the t_initial landmark
2. Iterates all gesture folders and recording subfolders
3. Skips folders missing labels.npy and reports them

Run:

```bash
python extract_samples_rq1.py
```

## Step 2: Train and compare models

Script: train_model_rq1.py

What it does:

1. Loads RQ1 samples using GestureDataset
2. For each window length:
   1. Builds stratified train/val/test splits
   2. Trains a CNN model
   3. Keeps the best checkpoint by validation accuracy
   4. Evaluates on test set
3. Prints confusion matrix and classification report
4. Saves model weights and summary results

Run:

```bash
python train_model_rq1.py
```

## Outputs

| Artifact              | Location              | Purpose                           |
| --------------------- | --------------------- | --------------------------------- |
| event_samples_rq1.npy | each recording folder | extracted frames for 20/30/50 ms  |
| rq1_20ms.pth          | MODEL_DIR             | best model for 20 ms              |
| rq1_30ms.pth          | MODEL_DIR             | best model for 30 ms              |
| rq1_50ms.pth          | MODEL_DIR             | best model for 50 ms              |
| rq1_results.npy       | RESULTS_DIR           | evaluation summary across windows |

## Recommended run order

1. Ensure recordings are labeled and contain labels.npy
2. Run extraction
3. Run training
4. Compare test accuracy across the three windows

## RQ1 answer criterion

RQ1 is answered by the final test-accuracy comparison:

1. Higher test accuracy indicates the better temporal window for this dataset
2. The saved results file keeps all window-level metrics together
