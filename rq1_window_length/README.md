# RQ1: Window Length Comparison

**Research Question:** Does event window length (20ms vs 30ms vs 50ms) affect gesture classification accuracy?

## **extract_samples_rq1.py**

Generates training samples by extracting events at **t_initial** with three different window lengths.

**Input:**

- `labels.npy` (from manual labeling)
- `recording_*.raw` (DVS event file)

**Output:**

- `event_samples_rq1.npy` containing:
  ```python
  {
      '20ms': (720, 1280) event frame,
      '30ms': (720, 1280) event frame,
      '50ms': (720, 1280) event frame
  }
  ```

**Run once per recording after labeling:**

```bash
python extract_samples_rq1.py
```

## **train_model_rq1.py**

Trains three separate CNN models (one per window length) and compares accuracy.

**What it does:**

1. Loads all `event_samples_rq1.npy` files
2. Splits into train/val/test (70%/10%/20%)
3. Trains CNN for each window: 20ms, 30ms, 50ms
4. Evaluates on test set
5. Saves models and results

**Output:**

- `model_20ms.pth`, `model_30ms.pth`, `model_50ms.pth`
- `rq1_results.npy` (accuracy comparison)

**Run after all samples are generated:**

```bash
python train_model_rq1.py
```

**Answers RQ1:** Compares test accuracy across window lengths → identifies optimal temporal window for low-latency gesture recognition.
