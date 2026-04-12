# RQ2: Temporal Landmark Comparison

## Goal

Test how early a gesture can be recognized after t_initial while maintaining strong accuracy.

Temporal landmarks compared:

1. t_initial (0 ms)
2. t_early (+50 ms)
3. t_mid (+100 ms)
4. t_late (+200 ms)

Fixed window size for all landmarks: 50 ms.

## End-to-end flow

```text
labels + events
      -> extract_samples_rq2.py
      -> event_samples_rq2.npy (t_initial, t_early, t_mid, t_late)
      -> train_model_rq2.py
      -> 4 trained models + rq2_results.npy
```

## Step 1: Build RQ2 samples

Script: extract_samples_rq2.py

What it reads per recording:

1. labels.npy for t_initial timestamp
2. prophesee_events.raw for DVS events

Landmark windows extracted:

1. t_initial: [t_initial, t_initial + 50 ms)
2. t_early: [t_initial + 50 ms, t_initial + 100 ms)
3. t_mid: [t_initial + 100 ms, t_initial + 150 ms)
4. t_late: [t_initial + 200 ms, t_initial + 250 ms)

What it creates:

1. event_samples_rq2.npy
2. Contains one 2D event-count frame per landmark

How it behaves:

1. Iterates all gesture folders and recording subfolders
2. Reads full event stream and slices by landmark time windows

Run:

```bash
python extract_samples_rq2.py
```

## Step 2: Train and compare landmarks

Script: train_model_rq2.py

What it does:

1. Loads RQ2 samples through GestureDataset
2. For each landmark:
   1. Builds stratified train/val/test splits
   2. Trains a CNN model
   3. Selects best checkpoint by validation accuracy
   4. Evaluates test accuracy
3. Adds latency metadata for each landmark
4. Prints confusion matrix and classification report
5. Saves model weights and summary results

Run:

```bash
python train_model_rq2.py
```

## Outputs

| Artifact              | Location              | Purpose                                     |
| --------------------- | --------------------- | ------------------------------------------- |
| event_samples_rq2.npy | each recording folder | extracted frames at four temporal landmarks |
| rq2_t_initial.pth     | MODEL_DIR             | best model at 0 ms latency                  |
| rq2_t_early.pth       | MODEL_DIR             | best model at +50 ms latency                |
| rq2_t_mid.pth         | MODEL_DIR             | best model at +100 ms latency               |
| rq2_t_late.pth        | MODEL_DIR             | best model at +200 ms latency               |
| rq2_results.npy       | RESULTS_DIR           | landmark-level accuracy + latency summary   |

## Recommended run order

1. Ensure recordings are labeled (labels.npy present)
2. Run sample extraction
3. Run landmark training
4. Compare latency-accuracy trade-off across all four landmarks

## RQ2 answer criterion

RQ2 is answered by combining latency and test accuracy:

1. Earlier landmarks are faster but may have less gesture information
2. Later landmarks are slower but often more stable
3. The preferred operating point is the earliest landmark with acceptable accuracy
