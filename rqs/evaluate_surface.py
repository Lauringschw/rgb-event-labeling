import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

SLIDING_DIR_T7 = Path(os.getenv("SLIDING_DIR_T7")) / "timesurface"
SLIDING_DIR    = Path(os.getenv("SLIDING_DIR"))
TEST_DIR       = Path(os.getenv("TEST_DIR", str(SLIDING_DIR / "test_samples"))) / "timesurface"
RESULTS_DIR    = TEST_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RQ1_DURATIONS_MS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RQ2_OFFSETS_MS   = [0, 20, 40, 60, 80, 100]
RQ2_DURATION_MS  = 30
LABEL_TO_GESTURE = {0: 'rock', 1: 'paper', 2: 'scissor'}


# == model =====================================================================

class TimeSurfaceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
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
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].copy().astype(np.float32)
            for j in range(len(batch)):
                max_val = batch[j].max()
                if max_val > 0:
                    batch[j] = batch[j] / max_val
            out = model(torch.from_numpy(batch).to(device))
            preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def accuracy(preds, labels):
    return 100.0 * np.mean(preds == labels)


def per_class_accuracy(preds, labels):
    return {LABEL_TO_GESTURE[i]: 100.0 * np.mean(preds[labels == i] == i)
            for i in range(3)}


# == RQ1 =======================================================================

def evaluate_rq1(model, device):
    print("\n" + "=" * 55)
    print("RQ1 — Window Length Effect")
    print("=" * 55)

    data      = np.load(TEST_DIR / "rq1_data.npy")
    labels    = np.load(TEST_DIR / "rq1_labels.npy")
    durations = np.load(TEST_DIR / "rq1_durations_ms.npy")

    results = {}
    lines   = ["RQ1 — Window Length Effect (Time Surface)"]
    header  = f"{'Duration':>10}  {'Overall':>8}  {'Rock':>8}  {'Paper':>8}  {'Scissor':>8}"
    print(header)
    print("-" * len(header))
    lines.append(header)

    for dur in RQ1_DURATIONS_MS:
        mask    = durations == dur
        preds   = run_inference(model, data[mask], device)
        overall = accuracy(preds, labels[mask])
        pc      = per_class_accuracy(preds, labels[mask])
        results[dur] = overall
        line = (f"{dur:>8}ms  {overall:>7.2f}%"
                f"  {pc['rock']:>7.2f}%  {pc['paper']:>7.2f}%  {pc['scissor']:>7.2f}%")
        print(line)
        lines.append(line)

    np.save(RESULTS_DIR / "rq1_accuracies.npy",  np.array([results[d] for d in RQ1_DURATIONS_MS]))
    np.save(RESULTS_DIR / "rq1_durations_ms.npy", np.array(RQ1_DURATIONS_MS))
    (RESULTS_DIR / "rq1_results.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved -> {RESULTS_DIR}/rq1_*")
    return results


# == RQ2 =======================================================================

def evaluate_rq2(model, device):
    print("\n" + "=" * 55)
    print("RQ2 — Temporal Landmark Effect")
    print("=" * 55)

    data    = np.load(TEST_DIR / "rq2_data.npy")
    labels  = np.load(TEST_DIR / "rq2_labels.npy")
    offsets = np.load(TEST_DIR / "rq2_offsets_ms.npy")

    results = {}
    lines   = ["RQ2 — Temporal Landmark Effect (Time Surface)"]
    header  = f"{'Offset':>10}  {'Window':>14}  {'Overall':>8}  {'Rock':>8}  {'Paper':>8}  {'Scissor':>8}"
    print(header)
    print("-" * len(header))
    lines.append(header)

    for off in RQ2_OFFSETS_MS:
        mask    = offsets == off
        preds   = run_inference(model, data[mask], device)
        overall = accuracy(preds, labels[mask])
        pc      = per_class_accuracy(preds, labels[mask])
        results[off] = overall
        line = (f"{off:>8}ms  {f't+{off}-{off+RQ2_DURATION_MS}ms':>14}"
                f"  {overall:>7.2f}%"
                f"  {pc['rock']:>7.2f}%  {pc['paper']:>7.2f}%  {pc['scissor']:>7.2f}%")
        print(line)
        lines.append(line)

    np.save(RESULTS_DIR / "rq2_accuracies.npy",  np.array([results[o] for o in RQ2_OFFSETS_MS]))
    np.save(RESULTS_DIR / "rq2_offsets_ms.npy",  np.array(RQ2_OFFSETS_MS))
    (RESULTS_DIR / "rq2_results.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved -> {RESULTS_DIR}/rq2_*")
    return results


# == RQ3 =======================================================================

def evaluate_rq3(rq1_results):
    print("\n" + "=" * 55)
    print("RQ3 — Representation Comparison (Time Surface contribution)")
    print("=" * 55)

    acc_30ms = rq1_results[30]
    print(f"Time surface accuracy at tau=0, dt=30ms: {acc_30ms:.2f}%")

    np.save(RESULTS_DIR / "rq3_timesurface_acc_30ms.npy", np.array([acc_30ms]))
    (RESULTS_DIR / "rq3_timesurface_result.txt").write_text(
        f"RQ3 — Time surface accuracy at tau=0, dt=30ms\n{acc_30ms:.4f}\n")
    print(f"Saved -> {RESULTS_DIR}/rq3_timesurface_*")


# == entry point ===============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("EVALUATION — Time Surface CNN (RQ1 + RQ2 + RQ3)")
    print("=" * 55)

    device = get_device()
    print(f"\nDevice: {device}")

    model_path = SLIDING_DIR_T7 / "model_timesurface_best.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\nRun train_timesurface.py first.")

    model = TimeSurfaceCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded from {model_path}\n")

    for fname in ["rq1_data.npy", "rq1_labels.npy", "rq1_durations_ms.npy",
                  "rq2_data.npy", "rq2_labels.npy", "rq2_offsets_ms.npy"]:
        if not (TEST_DIR / fname).exists():
            raise FileNotFoundError(
                f"Missing: {TEST_DIR / fname}\n"
                f"Run extract_test_samples_timesurface.py first.")

    rq1_results = evaluate_rq1(model, device)
    rq2_results = evaluate_rq2(model, device)
    evaluate_rq3(rq1_results)

    print("\n" + "=" * 55)
    print("COMPLETE — results saved to:", RESULTS_DIR)
    print("=" * 55)