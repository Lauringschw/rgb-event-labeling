# RQ2: Temporal Landmark Comparison

**Research Question:** Can we predict gestures before they're fully formed (early prediction) or must we wait until completion?

## **extract_samples_rq2.py**

Generates training samples by extracting **50ms event windows** starting at four different timepoints after gesture onset.

**How it works:**
Each landmark captures a different temporal slice of the gesture:

- **t_initial** (0ms after onset): events from `[t_initial → t_initial + 50ms]`
- **t_early** (+50ms): events from `[t_initial + 50ms → t_initial + 100ms]`
- **t_mid** (+100ms): events from `[t_initial + 100ms → t_initial + 150ms]`
- **t_late** (+200ms): events from `[t_initial + 200ms → t_initial + 250ms]`

**Input:**

- `labels.npy` (from manual labeling)
- `recording_*.raw` (DVS event file)

**Output:**

- `event_samples_rq2.npy` containing:
  ```python
  {
      't_initial': (720, 1280) event frame,  # earliest prediction (0ms latency)
      't_early':   (720, 1280) event frame,  # early prediction (+50ms latency)
      't_mid':     (720, 1280) event frame,  # mid prediction (+100ms latency)
      't_late':    (720, 1280) event frame   # late prediction (+200ms latency)
  }
  ```

**Run once per recording after labeling:**

```bash
python extract_samples_rq2.py
```

## **train_model_rq2.py**

Trains four separate CNN models (one per temporal landmark) and compares the accuracy vs prediction speed trade-off.

**What it does:**

1. Loads all `event_samples_rq2.npy` files
2. Splits into train/val/test (70%/10%/20%)
3. Trains separate CNN for each landmark
4. Measures prediction latency (time after gesture onset)
5. Evaluates accuracy on test set
6. Saves models and results

**Output:**

- `model_t_initial.pth`, `model_t_early.pth`, `model_t_mid.pth`, `model_t_late.pth`
- `rq2_results.npy` (accuracy + latency comparison)

**Run after all samples are generated:**

```bash
python train_model_rq2.py
```

**Answers RQ2:** Determines the earliest point at which gestures can be reliably recognized — balancing prediction speed (t_initial = fastest) with accuracy (t_late = most information).
