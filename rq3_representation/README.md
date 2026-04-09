# RQ3: Event Representation Comparison

**Research Question:** Which event encoding (2D histogram vs 3D voxel grid vs time surface) captures gesture dynamics most effectively?

## **extract_samples_rq3.py**

Generates training samples by encoding the same events (t_initial + 50ms window) in **three different representations**.

**Input:**

- `labels.npy` (from manual labeling)
- `recording_*.raw` (DVS event file)

**Output:**

- `event_samples_rq3.npy` containing:
  ```python
  {
      'histogram':    (720, 1280) 2D event count,
      'voxel_grid':   (5, 720, 1280) 3D temporal bins,
      'time_surface': (720, 1280) 2D recency map
  }
  ```

**Run once per recording after labeling:**

```bash
python extract_samples_rq3.py
```

## **train_model_rq3.py**

Trains three models with different architectures (2D CNN for histogram/time_surface, 3D CNN for voxel_grid) and compares accuracy.

**What it does:**

1. Loads all `event_samples_rq3.npy` files
2. Splits into train/val/test (70%/10%/20%)
3. Trains models for each representation:
   - **histogram** → 2D CNN (spatial event count)
   - **voxel_grid** → 3D CNN (temporal + spatial bins)
   - **time_surface** → 2D CNN (event recency)
4. Evaluates on test set
5. Saves models and results

**Output:**

- `model_histogram.pth`, `model_voxel_grid.pth`, `model_time_surface.pth`
- `rq3_results.npy` (accuracy comparison)

**Run after all samples are generated:**

```bash
python train_model_rq3.py
```

**Answers RQ3:** Compares accuracy across event representations → identifies optimal encoding for gesture classification with event cameras.
