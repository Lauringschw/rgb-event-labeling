import numpy as np
import matplotlib.pyplot as plt
from metavision_core.event_io import EventsIterator
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / '.env')

GESTURES = ["rock", "paper", "scissor"]
COLORS   = {'rock': '#e74c3c', 'paper': '#3498db', 'scissor': '#2ecc71'}
RECORDING_NUM = 150  # Check recording 150 for each gesture


def load_labels(folder: Path):
    p = folder / 'labels.npy'
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True).item()


def load_events(folder: Path):
    raw = folder / 'prophesee_events.raw'
    if not raw.exists():
        return None
    ev_list = []
    for evs in EventsIterator(str(raw)):
        ev_list.append(evs)
    return np.concatenate(ev_list)


def to_histogram_frame(events, height=360, width=640):
    """Create 2D event histogram for visualization (downsampled to 360x640)"""
    if len(events) == 0:
        return np.zeros((height, width), dtype=np.float32)
    
    # Downsample coordinates
    x = (events['x'].astype(np.int32) * width  // 1280)
    y = (events['y'].astype(np.int32) * height // 720)
    
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    
    frame = np.zeros((height, width), dtype=np.float32)
    on_events = events[valid & (events['p'] == 1)]
    
    if len(on_events) > 0:
        np.add.at(frame, (y[valid & (events['p'] == 1)], x[valid & (events['p'] == 1)]), 1)
    
    # Normalize for visualization
    if frame.max() > 0:
        frame = frame / frame.max()
    
    return frame


def visualize_300ms_extraction(base: Path, output_dir: Path):
    """
    Visualize what's happening in the 300ms extraction window:
    1. Event count over time (0-500ms)
    2. Visual histogram of events at different time points
    3. Cumulative event capture percentage
    """
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # Time bins for event density
    BIN_MS = 10
    TOTAL_WINDOW_MS = 500
    EXTRACTION_WINDOW_MS = 300
    bin_us = BIN_MS * 1000
    n_bins = TOTAL_WINDOW_MS // BIN_MS
    
    # Time points to visualize (0ms, 100ms, 200ms, 300ms)
    viz_times_ms = [0, 100, 200, 300]
    
    results = {}
    
    for col, gesture in enumerate(GESTURES):
        prefix = gesture[0]
        folder = base / gesture / f"{prefix}_{RECORDING_NUM}"
        
        if not folder.exists():
            print(f"!! {gesture}/{prefix}_{RECORDING_NUM} does not exist")
            continue
        
        labels = load_labels(folder)
        events = load_events(folder)
        
        if labels is None or events is None:
            print(f"!! Could not load {folder.name}")
            continue
        
        t_initial = labels['t_initial_time_us']
        color = COLORS[gesture]
        
        # === ROW 0: Event density over time ===================================
        ax_density = fig.add_subplot(gs[0, col])
        
        mask_500ms = (events['t'] >= t_initial) & (events['t'] < t_initial + TOTAL_WINDOW_MS * 1000)
        ev_500ms = events[mask_500ms]
        
        counts = np.zeros(n_bins, dtype=np.int32)
        for i in range(n_bins):
            lo = t_initial + i * bin_us
            hi = lo + bin_us
            counts[i] = np.sum((ev_500ms['t'] >= lo) & (ev_500ms['t'] < hi))
        
        centers = np.arange(n_bins) * BIN_MS + BIN_MS / 2
        ax_density.bar(centers, counts, width=BIN_MS*0.8, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_density.axvspan(0, EXTRACTION_WINDOW_MS, color='green', alpha=0.15, label='300ms extraction')
        ax_density.axvline(EXTRACTION_WINDOW_MS, color='red', linestyle='--', linewidth=2, label='Cutoff')
        
        # Mark visualization points
        for vt in viz_times_ms:
            if vt <= EXTRACTION_WINDOW_MS:
                ax_density.axvline(vt, color='orange', linestyle=':', alpha=0.5, linewidth=1)
        
        ax_density.set_xlabel('Time from t_initial (ms)', fontsize=10)
        ax_density.set_ylabel(f'Events per {BIN_MS}ms', fontsize=10)
        ax_density.set_title(f'{gesture.capitalize()} - Event Density', fontsize=12, fontweight='bold')
        ax_density.legend(fontsize=8, loc='upper right')
        ax_density.grid(True, alpha=0.3, axis='y')
        ax_density.set_xlim([0, TOTAL_WINDOW_MS])
        
        # === ROWS 1-3: Visual histograms at different time points ============
        for row_idx, viz_time_ms in enumerate(viz_times_ms, start=1):
            ax_hist = fig.add_subplot(gs[row_idx, col])
            
            # Extract events from t_initial to t_initial + viz_time_ms
            mask_viz = (events['t'] >= t_initial) & (events['t'] < t_initial + viz_time_ms * 1000)
            ev_viz = events[mask_viz]
            
            frame = to_histogram_frame(ev_viz)
            n_events = len(ev_viz)
            
            im = ax_hist.imshow(frame, cmap='hot', vmin=0, vmax=1, aspect='auto')
            ax_hist.axis('off')
            
            # Label
            label_color = 'green' if viz_time_ms <= EXTRACTION_WINDOW_MS else 'red'
            label_text = f'{viz_time_ms}ms: {n_events:,} events'
            if viz_time_ms == EXTRACTION_WINDOW_MS:
                label_text += ' ✓'
            elif viz_time_ms > EXTRACTION_WINDOW_MS:
                label_text += ' ✗'
            
            ax_hist.text(0.02, 0.98, label_text,
                        transform=ax_hist.transAxes, fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=label_color, alpha=0.7),
                        color='white', fontweight='bold')
            
            if col == 2 and row_idx == 2:  # Add colorbar only once
                cbar = plt.colorbar(im, ax=ax_hist, fraction=0.046, pad=0.04)
                cbar.set_label('Normalized intensity', fontsize=8)
        
        # === Calculate statistics =============================================
        mask_300ms = (events['t'] >= t_initial) & (events['t'] < t_initial + EXTRACTION_WINDOW_MS * 1000)
        ev_300ms = events[mask_300ms]
        
        total_events_500ms = len(ev_500ms)
        events_300ms = len(ev_300ms)
        pct_captured = (events_300ms / total_events_500ms * 100) if total_events_500ms > 0 else 0
        
        results[gesture] = {
            'total_500ms': total_events_500ms,
            'captured_300ms': events_300ms,
            'pct': pct_captured
        }
        
        print(f"{gesture.capitalize():<8} | 0-500ms: {total_events_500ms:>7,} events | "
              f"0-300ms: {events_300ms:>7,} events ({pct_captured:>5.1f}%)")
    
    # Overall title
    fig.suptitle(f'300ms Extraction Window Visualization (Recording #{RECORDING_NUM})\n'
                 f'Green = Extracted | Red = Not Extracted', 
                 fontsize=14, fontweight='bold')
    
    out = output_dir / f'visualize_300ms_extraction_rec{RECORDING_NUM}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return results, out


def print_summary(results):
    """Print summary statistics"""
    print("\n" + "=" * 60)
    print("SUMMARY: 300ms Extraction Window Coverage")
    print("=" * 60)
    
    if not results:
        print("No results to display")
        return
    
    avg_pct = np.mean([r['pct'] for r in results.values()])
    min_pct = min([r['pct'] for r in results.values()])
    max_pct = max([r['pct'] for r in results.values()])
    
    total_500ms = sum([r['total_500ms'] for r in results.values()])
    total_300ms = sum([r['captured_300ms'] for r in results.values()])
    
    print(f"Average capture:  {avg_pct:.1f}% of events")
    print(f"Range:            {min_pct:.1f}% - {max_pct:.1f}%")
    print(f"Combined total:   {total_300ms:,} / {total_500ms:,} events captured")
    print()
    
    if min_pct >= 95:
        print("✓ EXCELLENT: ≥95% of events captured")
        print("  → 300ms window is MORE than sufficient")
        print("  → Could potentially reduce to 250ms")
    elif min_pct >= 90:
        print("✓ GOOD: 90-95% of events captured")
        print("  → 300ms window is sufficient")
    elif min_pct >= 85:
        print("⚠ ACCEPTABLE: 85-90% of events captured")
        print("  → 300ms works but consider 350ms for better coverage")
    else:
        print("✗ INSUFFICIENT: <85% of events captured")
        print("  → Increase to 400ms or 500ms")
    
    print("=" * 60)


if __name__ == '__main__':
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    output_dir = Path(os.getenv("OUTPUT_DIR")) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('=' * 60)
    print(f'VISUALIZING 300ms EXTRACTION WINDOW (Recording #{RECORDING_NUM})')
    print('=' * 60)
    print(f'Input:   {base}')
    print(f'Output:  {output_dir}')
    print()
    print('This shows:')
    print('  - Event density over time (bar chart)')
    print('  - Visual histograms at 0ms, 100ms, 200ms, 300ms')
    print('  - What gets extracted vs what gets missed')
    print()
    
    results, out_path = visualize_300ms_extraction(base, output_dir)
    print_summary(results)
    
    print(f"\n✓ Visualization saved to: {out_path}")