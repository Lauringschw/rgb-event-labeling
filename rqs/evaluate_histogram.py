import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

# == paths =====================================================================
SLIDING_DIR_T7 = Path(os.getenv("SLIDING_DIR_T7"))
TEST_DIR       = Path(os.getenv("TEST_DIR",
                                str(Path(os.getenv("SLIDING_DIR")) / "test_samples")))
RESULTS_DIR    = TEST_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# == RQ configs ================================================================
RQ1_DURATIONS_MS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RQ2_OFFSETS_MS   = [0, 20, 40, 60, 80, 100]
RQ2_DURATION_MS  = 30
LABEL_TO_GESTURE = {0: 'rock', 1: 'paper', 2: 'scissor'}


# == model (must match train_histogram.py exactly) =============================

class HistogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 45 * 80, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# == device ====================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# == inference =================================================================

def run_inference(model, data, device, batch_size=32):
    """
    Run model inference on data array.
    Returns predicted labels as numpy array.
    """
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(data))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds   = []
    with torch.no_grad():
        for (X_batch,) in loader:
            out = model(X_batch.to(device))
            preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def accuracy(preds, labels):
    return 100.0 * np.mean(preds == labels)


def per_class_accuracy(preds, labels):
    return {
        LABEL_TO_GESTURE[i]: 100.0 * np.mean(preds[labels == i] == i)
        for i in range(3)
    }


# == RQ1 evaluation ============================================================

def evaluate_rq1(model, device):
    print("\n" + "=" * 55)
    print("RQ1 — Window Length Effect")
    print("=" * 55)

    data      = np.load(TEST_DIR / "rq1_data.npy")
    labels    = np.load(TEST_DIR / "rq1_labels.npy")
    durations = np.load(TEST_DIR / "rq1_durations_ms.npy")

    results = {}   # duration_ms -> overall accuracy
    lines   = []

    header = f"{'Duration':>10}  {'Overall':>8}  {'Rock':>8}  {'Paper':>8}  {'Scissor':>8}"
    print(header)
    print("-" * len(header))
    lines.append(header)

    for dur in RQ1_DURATIONS_MS:
        mask  = durations == dur
        X     = data[mask]
        y     = labels[mask]
        preds = run_inference(model, X, device)

        overall = accuracy(preds, y)
        per_cls = per_class_accuracy(preds, y)
        results[dur] = overall

        line = (f"{dur:>8}ms  {overall:>7.2f}%"
                f"  {per_cls['rock']:>7.2f}%"
                f"  {per_cls['paper']:>7.2f}%"
                f"  {per_cls['scissor']:>7.2f}%")
        print(line)
        lines.append(line)

    # save
    rq1_acc = np.array([results[d] for d in RQ1_DURATIONS_MS])
    np.save(RESULTS_DIR / "rq1_accuracies.npy",    rq1_acc)
    np.save(RESULTS_DIR / "rq1_durations_ms.npy",  np.array(RQ1_DURATIONS_MS))

    with open(RESULTS_DIR / "rq1_results.txt", 'w') as f:
        f.write("RQ1 — Window Length Effect (Histogram)\n")
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved -> {RESULTS_DIR / 'rq1_accuracies.npy'}")
    return results


# == RQ2 evaluation ============================================================

def evaluate_rq2(model, device):
    print("\n" + "=" * 55)
    print("RQ2 — Temporal Landmark Effect")
    print("=" * 55)

    data    = np.load(TEST_DIR / "rq2_data.npy")
    labels  = np.load(TEST_DIR / "rq2_labels.npy")
    offsets = np.load(TEST_DIR / "rq2_offsets_ms.npy")

    results = {}   # offset_ms -> overall accuracy
    lines   = []

    header = (f"{'Offset':>10}  {'Window':>14}  "
              f"{'Overall':>8}  {'Rock':>8}  {'Paper':>8}  {'Scissor':>8}")
    print(header)
    print("-" * len(header))
    lines.append(header)

    for off in RQ2_OFFSETS_MS:
        mask  = offsets == off
        X     = data[mask]
        y     = labels[mask]
        preds = run_inference(model, X, device)

        overall = accuracy(preds, y)
        per_cls = per_class_accuracy(preds, y)
        results[off] = overall

        window_str = f"t+{off}–{off+RQ2_DURATION_MS}ms"
        line = (f"{off:>8}ms  {window_str:>14}"
                f"  {overall:>7.2f}%"
                f"  {per_cls['rock']:>7.2f}%"
                f"  {per_cls['paper']:>7.2f}%"
                f"  {per_cls['scissor']:>7.2f}%")
        print(line)
        lines.append(line)

    # save
    rq2_acc = np.array([results[o] for o in RQ2_OFFSETS_MS])
    np.save(RESULTS_DIR / "rq2_accuracies.npy",   rq2_acc)
    np.save(RESULTS_DIR / "rq2_offsets_ms.npy",   np.array(RQ2_OFFSETS_MS))

    with open(RESULTS_DIR / "rq2_results.txt", 'w') as f:
        f.write("RQ2 — Temporal Landmark Effect (Histogram)\n")
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved -> {RESULTS_DIR / 'rq2_accuracies.npy'}")
    return results


# == RQ3 evaluation ============================================================

def evaluate_rq3(rq1_results):
    """
    RQ3 uses histogram accuracy at τ=0, Δt=30ms — sliced from RQ1 results.
    The actual RQ3 comparison (histogram vs voxel vs time surface) happens
    in a separate script after all three representations are evaluated.
    """
    print("\n" + "=" * 55)
    print("RQ3 — Representation Comparison (Histogram contribution)")
    print("=" * 55)

    acc_30ms = rq1_results[30]
    print(f"Histogram accuracy at τ=0, Δt=30ms: {acc_30ms:.2f}%")
    print("(Full RQ3 comparison after voxel + time surface evaluation)")

    # save for cross-representation comparison
    np.save(RESULTS_DIR / "rq3_histogram_acc_30ms.npy", np.array([acc_30ms]))

    with open(RESULTS_DIR / "rq3_histogram_result.txt", 'w') as f:
        f.write("RQ3 — Histogram accuracy at τ=0, Δt=30ms\n")
        f.write(f"{acc_30ms:.4f}\n")

    print(f"Saved -> {RESULTS_DIR / 'rq3_histogram_acc_30ms.npy'}")


# == entry point ===============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("EVALUATION — Histogram CNN (RQ1 + RQ2 + RQ3)")
    print("=" * 55)

    # == device & model ========================================================
    device = get_device()
    print(f"\nDevice: {device}")

    model_path = SLIDING_DIR_T7 / "model_histogram_best.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run train_histogram.py first."
        )

    model = HistogramCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}\n")

    # == check test samples exist ==============================================
    for fname in ["rq1_data.npy", "rq1_labels.npy", "rq1_durations_ms.npy",
                  "rq2_data.npy", "rq2_labels.npy", "rq2_offsets_ms.npy"]:
        if not (TEST_DIR / fname).exists():
            raise FileNotFoundError(
                f"Missing: {TEST_DIR / fname}\n"
                f"Run extract_test_samples_histogram.py first."
            )

    # == evaluate ==============================================================
    rq1_results = evaluate_rq1(model, device)
    rq2_results = evaluate_rq2(model, device)
    evaluate_rq3(rq1_results)

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETE")
    print("=" * 55)
    print(f"Results saved to: {RESULTS_DIR}")