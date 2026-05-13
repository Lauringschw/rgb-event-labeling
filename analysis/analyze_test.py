"""
statistical_tests.py

Computes:
  1. Wilson 95% CIs on all RQ1 accuracies (per representation x window size)
  2. McNemar's test for RQ3: pairwise representation comparisons at 30ms
  3. Cochran's Q for RQ2: accuracy across offsets per (representation, window size)

RQ2 file layout:
  rq2_data.npy   shape (N * n_offsets, C, H, W)
  rq2_labels.npy shape (N * n_offsets,)
  Offsets stacked in order [0, 20, 40, 60, 80, 100] ms, N samples each.

Usage:
    python3 statistical_tests.py

Output:
  ./stats_output/wilson_cis.csv
  ./stats_output/mcnemar_rq3.csv
  ./stats_output/cochran_rq2.csv
  ./stats_output/summary.txt
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import os
import csv
from itertools import combinations
from dotenv import load_dotenv
from torchvision import models
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2

load_dotenv(Path(__file__).parent.parent / '.env')

SLIDING_BASE = Path(os.getenv("SLIDING_BASE"))

# == config ====================================================================

WINDOW_SIZES     = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
OFFSETS          = [0, 20, 40, 60, 80, 100]
N_BINS           = 5
REPR_NAMES       = ['histogram', 'voxel', 'timesurface']
LABEL_TO_GESTURE = {0: 'rock', 1: 'paper', 2: 'scissor'}

OUTPUT_DIR = Path('./stats_output')
OUTPUT_DIR.mkdir(exist_ok=True)

BASE_DIRS = {
    'histogram':   SLIDING_BASE / 'histogram',
    'voxel':       SLIDING_BASE / 'voxel',
    'timesurface': SLIDING_BASE / 'timesurface',
}

# == models ====================================================================

class ResNet18Histogram(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x): return self.resnet(x)

class ResNet18Voxel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(N_BINS, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x): return self.resnet(x)

class ResNet18TimeSurface(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x): return self.resnet(x)

MODEL_CLS = {
    'histogram':   ResNet18Histogram,
    'voxel':       ResNet18Voxel,
    'timesurface': ResNet18TimeSurface,
}

# == helpers ===================================================================

def get_device():
    if torch.cuda.is_available():         return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def get_model_path(repr_name, window_ms, base_dir):
    if repr_name == 'histogram':
        return base_dir / f"{window_ms}ms" / "merged" / f"model_histogram_{window_ms}ms_best.pth"
    return base_dir / f"{window_ms}ms" / "merged" / f"model_{repr_name}_{window_ms}ms_best.pth"

def normalize(batch, repr_name):
    for j in range(len(batch)):
        max_val = np.abs(batch[j]).max() if repr_name == 'voxel' else batch[j].max()
        if max_val > 0:
            batch[j] /= max_val
    return batch

def run_inference(model, data, device, repr_name, batch_size=32):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = normalize(data[i:i+batch_size].copy().astype(np.float32), repr_name)
            out   = model(torch.from_numpy(batch).to(device))
            preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(preds)

def load_model(repr_name, window_ms, device):
    base_dir   = BASE_DIRS[repr_name]
    model_path = get_model_path(repr_name, window_ms, base_dir)
    if not model_path.exists():
        return None
    model = MODEL_CLS[repr_name]().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    return model

def load_rq1(repr_name, window_ms):
    test_dir = BASE_DIRS[repr_name] / f"{window_ms}ms" / "test_samples"
    needed   = ["rq1_data.npy", "rq1_labels.npy", "rq1_recording_ids.npy"]
    if not all((test_dir / f).exists() for f in needed):
        return None, None, None
    return (
        np.load(test_dir / "rq1_data.npy"),
        np.load(test_dir / "rq1_labels.npy"),
        np.load(test_dir / "rq1_recording_ids.npy"),
    )

def load_rq2_all_offsets(repr_name, window_ms):
    """
    rq2_data.npy  shape: (N * n_offsets, C, H, W)
    rq2_labels.npy shape: (N * n_offsets,)
    Offsets stacked in order [0, 20, 40, 60, 80, 100] ms, N samples each.
    Returns list of (data, labels) per offset, or None if files missing.
    """
    test_dir = BASE_DIRS[repr_name] / f"{window_ms}ms" / "test_samples"
    data_f   = test_dir / "rq2_data.npy"
    label_f  = test_dir / "rq2_labels.npy"
    if not data_f.exists() or not label_f.exists():
        return None
    data       = np.load(data_f)
    labels     = np.load(label_f)
    n_offsets  = len(OFFSETS)
    chunk_size = len(data) // n_offsets
    return [
        (data[i * chunk_size:(i + 1) * chunk_size],
         labels[i * chunk_size:(i + 1) * chunk_size])
        for i in range(n_offsets)
    ]

# == statistical functions =====================================================

def wilson_ci(n_correct, n_total, alpha=0.05):
    if n_total == 0:
        return float('nan'), float('nan'), float('nan')
    acc    = n_correct / n_total
    lo, hi = proportion_confint(n_correct, n_total, alpha=alpha, method='wilson')
    return acc, lo, hi

def mcnemar_test(correct_a, correct_b):
    b01    = int(np.sum(~correct_a &  correct_b))
    b10    = int(np.sum( correct_a & ~correct_b))
    table  = np.array([[0, b01], [b10, 0]])
    result = mcnemar(table, exact=True, correction=False)
    return result.pvalue, b10, b01

def cochran_q_test(correct_matrix):
    """
    correct_matrix: (n_conditions, n_samples) int array.
    Returns (Q, p, df).
    """
    k, n        = correct_matrix.shape
    L_i         = correct_matrix.sum(axis=0)
    G_j         = correct_matrix.sum(axis=1)
    total       = correct_matrix.sum()
    numerator   = (k - 1) * (k * np.sum(G_j**2) - total**2)
    denominator = k * total - np.sum(L_i**2)
    if denominator == 0:
        return float('nan'), float('nan'), k - 1
    Q  = numerator / denominator
    df = k - 1
    p  = chi2.sf(Q, df)
    return Q, p, df

# == 1. Wilson CIs =============================================================

def compute_wilson_cis(device):
    print("=" * 60)
    print("1. WILSON 95% CIs — RQ1 accuracies")
    print("=" * 60)

    rows = []
    for repr_name in REPR_NAMES:
        for window_ms in WINDOW_SIZES:
            data, labels, _ = load_rq1(repr_name, window_ms)
            if data is None:
                print(f"  SKIP {repr_name} {window_ms}ms — files missing")
                continue
            model = load_model(repr_name, window_ms, device)
            if model is None:
                print(f"  SKIP {repr_name} {window_ms}ms — model missing")
                continue

            preds       = run_inference(model, data, device, repr_name)
            n_correct   = int(np.sum(preds == labels))
            n_total     = len(labels)
            acc, lo, hi = wilson_ci(n_correct, n_total)

            print(f"  {repr_name:12s} {window_ms:>3}ms  "
                  f"acc={acc*100:.2f}%  95%CI=[{lo*100:.2f}%, {hi*100:.2f}%]  "
                  f"({n_correct}/{n_total})")

            rows.append({
                'representation': repr_name,
                'window_ms':      window_ms,
                'n_correct':      n_correct,
                'n_total':        n_total,
                'accuracy':       round(acc * 100, 4),
                'ci_lower':       round(lo  * 100, 4),
                'ci_upper':       round(hi  * 100, 4),
            })

    out = OUTPUT_DIR / 'wilson_cis.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {out}\n")
    return rows

# == 2. McNemar RQ3 ============================================================

def compute_mcnemar_rq3(device):
    print("=" * 60)
    print("2. McNEMAR'S TEST — RQ3 pairwise at 30ms (offset=0)")
    print("=" * 60)

    window_ms = 30
    correct   = {}

    for repr_name in REPR_NAMES:
        data, labels, _ = load_rq1(repr_name, window_ms)
        if data is None:
            print(f"  SKIP {repr_name} — files missing")
            continue
        model = load_model(repr_name, window_ms, device)
        if model is None:
            print(f"  SKIP {repr_name} — model missing")
            continue
        preds              = run_inference(model, data, device, repr_name)
        correct[repr_name] = (preds == labels)
        print(f"  {repr_name:12s}  n={len(labels)}  acc={correct[repr_name].mean()*100:.2f}%")

    rows = []
    print()
    for a, b in combinations(REPR_NAMES, 2):
        if a not in correct or b not in correct:
            continue
        p, b10, b01 = mcnemar_test(correct[a], correct[b])
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"  {a} vs {b}:")
        print(f"    discordant pairs: {a}correct+{b}wrong={b10}  {a}wrong+{b}correct={b01}")
        print(f"    McNemar p={p:.4f}  {sig}")
        rows.append({
            'repr_A':            a,
            'repr_B':            b,
            'n_A_right_B_wrong': b10,
            'n_A_wrong_B_right': b01,
            'p_value':           round(p, 6),
            'significant_0.05':  p < 0.05,
        })

    out = OUTPUT_DIR / 'mcnemar_rq3.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {out}\n")
    return rows

# == 3. Cochran's Q RQ2 ========================================================

def compute_cochran_rq2(device):
    print("=" * 60)
    print("3. COCHRAN'S Q — RQ2 offset effect per (representation, window size)")
    print("=" * 60)

    rows = []
    for repr_name in REPR_NAMES:
        for window_ms in WINDOW_SIZES:
            model = load_model(repr_name, window_ms, device)
            if model is None:
                print(f"  SKIP {repr_name} {window_ms}ms — model missing")
                continue

            chunks = load_rq2_all_offsets(repr_name, window_ms)
            if chunks is None:
                print(f"  SKIP {repr_name} {window_ms}ms — rq2 files missing")
                continue

            offset_corrects = []
            for data, labels in chunks:
                preds = run_inference(model, data, device, repr_name)
                offset_corrects.append((preds == labels).astype(int))

            mat      = np.array(offset_corrects)  # (n_offsets, chunk_size)
            Q, p, df = cochran_q_test(mat)
            sig      = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            accs     = [c.mean() * 100 for c in offset_corrects]
            acc_str  = "  ".join(f"t{o}={a:.1f}%" for o, a in zip(OFFSETS, accs))

            print(f"  {repr_name:12s} {window_ms:>3}ms  Q={Q:.2f}  p={p:.4f}  {sig}")
            print(f"    {acc_str}")

            rows.append({
                'representation':   repr_name,
                'window_ms':        window_ms,
                'Q_statistic':      round(Q, 4),
                'df':               df,
                'p_value':          round(p, 6),
                'significant_0.05': p < 0.05,
                **{f'acc_tau{o}ms': round(offset_corrects[i].mean() * 100, 2)
                   for i, o in enumerate(OFFSETS)},
            })

    if rows:
        out = OUTPUT_DIR / 'cochran_rq2.csv'
        with open(out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Saved: {out}\n")
    else:
        print("  No rows — check rq2_data.npy/rq2_labels.npy exist in test_samples/\n")
    return rows

# == 4. Summary ================================================================

def write_summary(wilson_rows, mcnemar_rows, cochran_rows):
    out   = OUTPUT_DIR / 'summary.txt'
    lines = ["STATISTICAL ANALYSIS SUMMARY", "=" * 60]

    lines.append("\n--- Wilson 95% CIs (key conditions) ---")
    for r in wilson_rows:
        if r['window_ms'] in [10, 30, 100]:
            lines.append(
                f"  {r['representation']:12s} {r['window_ms']:>3}ms  "
                f"{r['accuracy']:.2f}% [{r['ci_lower']:.2f}%, {r['ci_upper']:.2f}%]"
            )

    lines.append("\n--- McNemar RQ3 (30ms, offset=0) ---")
    for r in mcnemar_rows:
        sig = "significant" if r['significant_0.05'] else "not significant"
        lines.append(f"  {r['repr_A']} vs {r['repr_B']}: p={r['p_value']:.4f} ({sig})")

    lines.append("\n--- Cochran's Q RQ2 (offset effect, 30ms model) ---")
    for r in cochran_rows:
        if r['window_ms'] == 30:
            sig = "significant" if r['significant_0.05'] else "not significant"
            lines.append(
                f"  {r['representation']:12s}: Q={r['Q_statistic']:.2f}  "
                f"p={r['p_value']:.4f} ({sig})"
            )

    lines.append("\n--- Cochran's Q: summary across all models ---")
    if cochran_rows:
        sig_count = sum(1 for r in cochran_rows if r['significant_0.05'])
        lines.append(
            f"  {sig_count}/{len(cochran_rows)} (repr x window) combinations "
            f"significant at alpha=0.05"
        )
    else:
        lines.append("  No Cochran results computed.")

    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Summary saved: {out}")

# == main ======================================================================

if __name__ == '__main__':
    device = get_device()
    print(f"Device      : {device}")
    print(f"SLIDING_BASE: {SLIDING_BASE}")
    print(f"Output      : {OUTPUT_DIR.resolve()}\n")

    wilson_rows  = compute_wilson_cis(device)
    mcnemar_rows = compute_mcnemar_rq3(device)
    cochran_rows = compute_cochran_rq2(device)
    write_summary(wilson_rows, mcnemar_rows, cochran_rows)

    print("\n" + "=" * 60)
    print(f"COMPLETE — results in {OUTPUT_DIR.resolve()}")
    print("=" * 60)