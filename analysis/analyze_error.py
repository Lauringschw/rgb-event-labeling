"""
analyze_errors.py

Computes:
  1. Confusion matrices per representation per window size (RQ1, offset=0)
  2. Cross-representation error overlap across all window sizes
  3. Per-recording error consistency at 30ms (RQ3 baseline)

Usage:
    python3 analyze_errors.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from dotenv import load_dotenv
from torchvision import models

load_dotenv(Path(__file__).parent.parent / '.env')

SLIDING_BASE = Path(os.getenv("SLIDING_BASE"))

# == config ====================================================================

WINDOW_SIZES     = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_BINS           = 5
LABEL_TO_GESTURE = {0: 'rock', 1: 'paper', 2: 'scissor'}
REPR_NAMES       = ['histogram', 'voxel', 'timesurface']

OUTPUT_DIR = Path('./analysis_output')
OUTPUT_DIR.mkdir(exist_ok=True)

# If histogram/voxel/timesurface live under different base dirs, change here.
# By default all three use SLIDING_BASE (same as evaluate.py).
BASE_DIRS = {
    'histogram':   SLIDING_BASE / 'histogram',
    'voxel':       SLIDING_BASE / 'voxel',
    'timesurface': SLIDING_BASE / 'timesurface',
}


# == models (identical to evaluate.py) =========================================

class ResNet18Histogram(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        return self.resnet(x)


class ResNet18Voxel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(N_BINS, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        return self.resnet(x)


class ResNet18TimeSurface(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        return self.resnet(x)


MODEL_CLS = {
    'histogram':   ResNet18Histogram,
    'voxel':       ResNet18Voxel,
    'timesurface': ResNet18TimeSurface,
}


def get_model_path(repr_name: str, window_ms: int, base_dir: Path) -> Path:
    if repr_name == 'histogram':
        return base_dir / f"{window_ms}ms" / "merged" / f"model_histogram_{window_ms}ms_best.pth"
    else:
        return base_dir / f"{window_ms}ms" / "merged" / f"model_{repr_name}_{window_ms}ms_best.pth"


# == normalisation (identical to evaluate.py) ==================================

def normalize(batch, repr_name):
    for j in range(len(batch)):
        max_val = np.abs(batch[j]).max() if repr_name == 'voxel' else batch[j].max()
        if max_val > 0:
            batch[j] /= max_val
    return batch


# == device (identical to evaluate.py) =========================================

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# == inference (identical to evaluate.py) ======================================

def run_inference(model, data, device, repr_name, batch_size=32):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = normalize(data[i:i+batch_size].copy().astype(np.float32), repr_name)
            out   = model(torch.from_numpy(batch).to(device))
            preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# == load model + run rq1 inference ============================================

def get_rq1_preds(repr_name, window_ms, device):
    base_dir = BASE_DIRS[repr_name]
    test_dir = base_dir / f"{window_ms}ms" / "test_samples"
    model_path = get_model_path(repr_name, window_ms, base_dir)

    if not model_path.exists():
        print(f"  !! Model not found: {model_path}")
        return None, None, None

    for fname in ["rq1_data.npy", "rq1_labels.npy", "rq1_recording_ids.npy"]:
        if not (test_dir / fname).exists():
            print(f"  !! Missing: {test_dir / fname}")
            return None, None, None

    model = MODEL_CLS[repr_name]().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data    = np.load(test_dir / "rq1_data.npy")
    labels  = np.load(test_dir / "rq1_labels.npy")
    rec_ids = np.load(test_dir / "rq1_recording_ids.npy")

    preds = run_inference(model, data, device, repr_name)
    return preds, labels, rec_ids


# == 1. confusion matrices =====================================================

def plot_confusion_matrices(device):
    print("=" * 55)
    print("CONFUSION MATRICES")
    print("=" * 55)

    for repr_name in REPR_NAMES:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'Confusion Matrices — {repr_name}', fontsize=14, fontweight='bold')

        for idx, window_ms in enumerate(WINDOW_SIZES):
            ax = axes[idx // 5][idx % 5]
            print(f"  {repr_name} {window_ms}ms...", end=' ', flush=True)

            preds, labels, _ = get_rq1_preds(repr_name, window_ms, device)
            if preds is None:
                ax.set_title(f'{window_ms}ms — missing')
                ax.axis('off')
                print("skipped")
                continue

            cm = np.zeros((3, 3), dtype=int)
            for t, p in zip(labels, preds):
                cm[t][p] += 1
            acc = 100.0 * np.mean(preds == labels)

            ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(['R', 'P', 'S'], fontsize=9)
            ax.set_yticklabels(['R', 'P', 'S'], fontsize=9)
            ax.set_title(f'{window_ms}ms  ({acc:.1f}%)', fontsize=9)

            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=11,
                            color='white' if cm[i, j] > cm.max() * 0.5 else 'black')

            if idx % 5 == 0:
                ax.set_ylabel('True', fontsize=8)
            if idx >= 5:
                ax.set_xlabel('Predicted', fontsize=8)

            print(f"acc={acc:.1f}%")

        plt.tight_layout()
        out_path = OUTPUT_DIR / f'confusion_matrices_{repr_name}.pdf'
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}\n")


# == 2. cross-representation error overlap =====================================

def analyze_error_overlap(device):
    print("=" * 55)
    print("CROSS-REPRESENTATION ERROR OVERLAP")
    print("=" * 55)

    wrong_sets = {}  # (repr_name, window_ms) -> set of misclassified rec_ids

    for repr_name in REPR_NAMES:
        for window_ms in WINDOW_SIZES:
            print(f"  {repr_name} {window_ms}ms...", end=' ', flush=True)
            preds, labels, rec_ids = get_rq1_preds(repr_name, window_ms, device)
            if preds is None:
                print("skipped")
                continue
            wrong_mask = preds != labels
            wrong_sets[(repr_name, window_ms)] = set(rec_ids[wrong_mask].tolist())
            print(f"{wrong_mask.sum()} errors")

    header = (f"{'Window':>8}  {'H':>5}  {'V':>5}  {'T':>5}  "
              f"{'H∩V':>5}  {'H∩T':>5}  {'V∩T':>5}  {'H∩V∩T':>7}")
    print(f"\n{header}")
    print("-" * len(header))

    overlap_data = []
    for window_ms in WINDOW_SIZES:
        h = wrong_sets.get(('histogram',   window_ms), set())
        v = wrong_sets.get(('voxel',       window_ms), set())
        t = wrong_sets.get(('timesurface', window_ms), set())

        h_only = h - v - t
        v_only = v - h - t
        t_only = t - h - v
        hv     = (h & v) - t
        ht     = (h & t) - v
        vt     = (v & t) - h
        hvt    = h & v & t

        print(f"{window_ms:>6}ms  {len(h):>5}  {len(v):>5}  {len(t):>5}  "
              f"{len(hv):>5}  {len(ht):>5}  {len(vt):>5}  {len(hvt):>7}")

        overlap_data.append(dict(
            window_ms=window_ms,
            h=len(h), v=len(v), t=len(t),
            h_only=len(h_only), v_only=len(v_only), t_only=len(t_only),
            hv=len(hv), ht=len(ht), vt=len(vt), hvt=len(hvt),
            shared_h=len(hvt)/len(h) if h else 0,
            shared_v=len(hvt)/len(v) if v else 0,
            shared_t=len(hvt)/len(t) if t else 0,
        ))

    # save table
    txt_path = OUTPUT_DIR / 'error_overlap.txt'
    with open(txt_path, 'w') as f:
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')
        for d in overlap_data:
            f.write(f"{d['window_ms']:>6}ms  {d['h']:>5}  {d['v']:>5}  {d['t']:>5}  "
                    f"{d['hv']:>5}  {d['ht']:>5}  {d['vt']:>5}  {d['hvt']:>7}\n")
    print(f"\nSaved: {txt_path}")

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    windows = [d['window_ms'] for d in overlap_data]
    x = np.arange(len(windows))
    w = 0.6

    ax = axes[0]
    b0 = np.zeros(len(overlap_data))
    for key, label, color in [
        ('hvt',    'All 3',       '#d62728'),
        ('hv',     'Hist+Voxel',  '#ff7f0e'),
        ('ht',     'Hist+TS',     '#9467bd'),
        ('vt',     'Voxel+TS',    '#8c564b'),
        ('h_only', 'Hist only',   '#1f77b4'),
        ('v_only', 'Voxel only',  '#2ca02c'),
        ('t_only', 'TS only',     '#e377c2'),
    ]:
        vals = np.array([d[key] for d in overlap_data], dtype=float)
        ax.bar(x, vals, w, bottom=b0, label=label, color=color)
        b0 += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f'{w}ms' for w in windows])
    ax.set_xlabel('Window size')
    ax.set_ylabel('Misclassified recordings')
    ax.set_title('Error overlap across representations (RQ1, offset=0)')
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(windows, [d['shared_h']*100 for d in overlap_data], 'o-',
             label='Histogram',    color='#1f77b4')
    ax2.plot(windows, [d['shared_v']*100 for d in overlap_data], 's-',
             label='Voxel',        color='#2ca02c')
    ax2.plot(windows, [d['shared_t']*100 for d in overlap_data], '^-',
             label='Time Surface', color='#e377c2')
    ax2.set_xticks(windows)
    ax2.set_xticklabels([f'{w}ms' for w in windows])
    ax2.set_xlabel('Window size')
    ax2.set_ylabel('% of errors shared by all 3')
    ax2.set_title('Fraction of errors shared across all 3 representations')
    ax2.set_ylim(0, 100)
    ax2.legend()

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'error_overlap.pdf'
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# == 3. per-recording consistency at 30ms ======================================

def analyze_recording_consistency(device):
    print("\n" + "=" * 55)
    print("PER-RECORDING CONSISTENCY — 30ms models")
    print("=" * 55)

    window_ms    = 30
    error_counts = {}  # rec_id -> int (0–3)
    rec_labels   = {}  # rec_id -> true label
    all_rec_ids  = set()

    for repr_name in REPR_NAMES:
        preds, labels, rec_ids = get_rq1_preds(repr_name, window_ms, device)
        if preds is None:
            continue
        for pred, label, rid in zip(preds, labels, rec_ids):
            rid = int(rid)
            all_rec_ids.add(rid)
            rec_labels[rid] = int(label)
            if pred != label:
                error_counts[rid] = error_counts.get(rid, 0) + 1

    for n in range(4):
        count = sum(1 for r in all_rec_ids if error_counts.get(r, 0) == n)
        print(f"  Wrong by {n}/3 representations: {count} recordings")

    print()
    for gesture_id, gesture_name in LABEL_TO_GESTURE.items():
        recs = [r for r in all_rec_ids if rec_labels.get(r) == gesture_id]
        hard = [r for r in recs if error_counts.get(r, 0) == 3]
        print(f"  {gesture_name}: {len(hard)}/{len(recs)} recordings wrong by all 3")

    txt_path = OUTPUT_DIR / 'recording_consistency_30ms.txt'
    with open(txt_path, 'w') as f:
        f.write("rec_id,gesture,n_representations_wrong\n")
        for rid in sorted(all_rec_ids):
            gesture = LABEL_TO_GESTURE.get(rec_labels.get(rid), '?')
            f.write(f"{rid},{gesture},{error_counts.get(rid, 0)}\n")
    print(f"\nSaved: {txt_path}")


# == main ======================================================================

if __name__ == '__main__':
    device = get_device()
    print(f"Device      : {device}")
    print(f"SLIDING_BASE: {SLIDING_BASE}")
    print(f"Output      : {OUTPUT_DIR.resolve()}\n")

    plot_confusion_matrices(device)
    analyze_error_overlap(device)
    analyze_recording_consistency(device)

    print("\n" + "=" * 55)
    print(f"COMPLETE — results in {OUTPUT_DIR.resolve()}")
    print("=" * 55)