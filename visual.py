"""
visualize_resolutions.py

Shows how a single event recording looks as a 2D histogram
at different spatial resolutions side by side.

Usage: python3 visualize_resolutions.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from metavision_core.event_io import EventsIterator
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / '.env')

RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR"))
DIR            = os.getenv("DIR")

# ── pick one recording to visualize ──────────────────────────────────────────
# change this to any recording you want to inspect
GESTURE   = "paper"
REC_INDEX = 1
PREFIX    = GESTURE[0]

RESOLUTIONS = [
    (720, 1280, "Full res\n720×1280"),
    (360, 640,  "Half res\n360×640"),
    (180, 320,  "Quarter res\n180×320"),
    (90,  160,  "Eighth res\n90×160"),
]

ORIG_H = 720
ORIG_W = 1280

WINDOW_SIZE = 20_000   # events per window
EXTRACTION_RANGE_US = 300_000


def events_to_histogram(events, height, width):
    histogram = np.zeros((2, height, width), dtype=np.float32)
    if len(events) == 0:
        return histogram
    x = (events['x'] * width  // ORIG_W).astype(np.int32)
    y = (events['y'] * height // ORIG_H).astype(np.int32)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, p = x[valid], y[valid], events['p'][valid]
    on_mask  = p == 1
    off_mask = ~on_mask
    np.add.at(histogram[0], (y[on_mask],  x[on_mask]),  1)
    np.add.at(histogram[1], (y[off_mask], x[off_mask]), 1)
    return histogram


def load_events(folder: Path):
    labels_file = folder / "labels.npy"
    raw_file    = folder / "prophesee_events.raw"
    labels      = np.load(labels_file, allow_pickle=True).item()
    t_initial   = labels['t_initial_time_us']
    chunks = [ev for ev in EventsIterator(str(raw_file))]
    all_events = np.concatenate(chunks)
    mask   = (all_events['t'] >= t_initial) & \
             (all_events['t'] <  t_initial + EXTRACTION_RANGE_US)
    return all_events[mask]


def histogram_to_rgb(hist):
    """Convert 2-channel (ON/OFF) histogram to RGB for display."""
    on  = hist[0]
    off = hist[1]
    # normalize each channel independently
    on  = on  / (on.max()  + 1e-6)
    off = off / (off.max() + 1e-6)
    rgb = np.zeros((*on.shape, 3), dtype=np.float32)
    rgb[..., 0] = off   # red   = OFF events
    rgb[..., 1] = on    # green = ON events
    return np.clip(rgb, 0, 1)


if __name__ == "__main__":
    folder = RECORDINGS_DIR / DIR / GESTURE / f"{PREFIX}_{REC_INDEX}"
    print(f"Loading: {folder}")

    events = load_events(folder)
    print(f"Events in range: {len(events):,}")

    # use first full window
    window = events
    print(f"Window: {len(window):,} events\n")

    # ── plot ──────────────────────────────────────────────────────────────────
    n_res   = len(RESOLUTIONS)
    n_times = 3  # t_initial, t_initial+50ms, t_initial+100ms (if enough events)

    fig, axes = plt.subplots(2, n_res, figsize=(n_res * 4, 8))
    fig.suptitle(
        f"{GESTURE}/{PREFIX}_{REC_INDEX} — 2D Histogram at different resolutions\n"
        f"Green = ON events, Red = OFF events",
        fontsize=13
    )

    for col, (H, W, label) in enumerate(RESOLUTIONS):
        hist = events_to_histogram(window, H, W)
        rgb  = histogram_to_rgb(hist)

        # top row: combined (ON+OFF)
        axes[0, col].imshow(rgb, aspect='auto')
        axes[0, col].set_title(label, fontsize=11)
        axes[0, col].axis('off')
        axes[0, col].text(0.02, 0.02, f"{H}×{W}",
                          transform=axes[0, col].transAxes,
                          color='white', fontsize=8,
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        # bottom row: ON events only (grayscale)
        on_norm = hist[0] / (hist[0].max() + 1e-6)
        axes[1, col].imshow(on_norm, cmap='hot', aspect='auto')
        axes[1, col].set_title(f"ON only", fontsize=10)
        axes[1, col].axis('off')
        axes[1, col].text(0.02, 0.02,
                          f"max={int(hist[0].max())} events/px",
                          transform=axes[1, col].transAxes,
                          color='white', fontsize=7,
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    plt.tight_layout()

    # save
    out_path = Path("resolution_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")
    plt.show()