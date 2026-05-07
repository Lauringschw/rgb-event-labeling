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


def analyze_300ms_window(base: Path, output_dir: Path):
    """
    Check if 300ms extraction window is sufficient by:
    1. Plotting event count over time (0-500ms)
    2. Showing cumulative event percentage
    3. Highlighting the 300ms extraction window
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    BIN_MS = 10  # 10ms bins
    TOTAL_WINDOW_MS = 500  # Check up to 500ms
    bin_us = BIN_MS * 1000
    n_bins = TOTAL_WINDOW_MS // BIN_MS
    
    results = []
    
    for gesture in GESTURES:
        prefix = gesture[0]  # 'r', 'p', 's'
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
        t_go = labels.get('go_time_us', t_initial)
        
        # Filter events from t_initial to t_initial + 500ms
        mask = (events['t'] >= t_initial) & (events['t'] < t_initial + TOTAL_WINDOW_MS * 1000)
        ev = events[mask]
        
        # Count events per bin
        counts = np.zeros(n_bins, dtype=np.int32)
        for i in range(n_bins):
            lo = t_initial + i * bin_us
            hi = lo + bin_us
            counts[i] = np.sum((ev['t'] >= lo) & (ev['t'] < hi))
        
        # Cumulative events
        cumsum = np.cumsum(counts)
        total_events = cumsum[-1]
        
        # Events in 300ms window
        events_300ms = cumsum[29]  # 0-300ms (bins 0-29)
        pct_300ms = (events_300ms / total_events * 100) if total_events > 0 else 0
        
        color = COLORS[gesture]
        centers = np.arange(n_bins) * BIN_MS + BIN_MS / 2
        
        # Plot 1: Event count per bin
        axes[0].plot(centers, counts, color=color, linewidth=2, 
                     label=f"{gesture.capitalize()}: {total_events:,} total")
        axes[0].axvline(300, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Plot 2: Cumulative events
        axes[1].plot(centers, cumsum, color=color, linewidth=2,
                     label=f"{gesture.capitalize()}: {pct_300ms:.1f}% at 300ms")
        axes[1].axvline(300, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
        axes[1].axhline(events_300ms, color=color, linestyle=':', alpha=0.3)
        
        # Plot 3: Cumulative percentage
        cumsum_pct = (cumsum / total_events * 100) if total_events > 0 else np.zeros_like(cumsum)
        axes[2].plot(centers, cumsum_pct, color=color, linewidth=2,
                     label=f"{gesture.capitalize()}")
        axes[2].axvline(300, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
        axes[2].axhline(pct_300ms, color=color, linestyle=':', alpha=0.3)
        
        results.append({
            'gesture': gesture,
            'total_events': total_events,
            'events_300ms': events_300ms,
            'pct_300ms': pct_300ms
        })
        
        print(f"{gesture.capitalize():<8} | Total: {total_events:>7,} events | "
              f"0-300ms: {events_300ms:>7,} events ({pct_300ms:>5.1f}%)")
    
    # Formatting
    axes[0].set_xlabel('Time from t_initial (ms)', fontsize=11)
    axes[0].set_ylabel(f'Event count per {BIN_MS}ms', fontsize=11)
    axes[0].set_title(f'Event Count Over Time (Recording #{RECORDING_NUM})', fontsize=12, fontweight='bold')
    axes[0].axvspan(0, 300, color='gray', alpha=0.1, label='300ms extraction window')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time from t_initial (ms)', fontsize=11)
    axes[1].set_ylabel('Cumulative event count', fontsize=11)
    axes[1].set_title('Cumulative Events', fontsize=12, fontweight='bold')
    axes[1].axvspan(0, 300, color='gray', alpha=0.1)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Time from t_initial (ms)', fontsize=11)
    axes[2].set_ylabel('Cumulative % of total events', fontsize=11)
    axes[2].set_title('Percentage of Events Captured', fontsize=12, fontweight='bold')
    axes[2].axvspan(0, 300, color='gray', alpha=0.1)
    axes[2].axhline(100, color='black', linestyle='-', alpha=0.2, linewidth=0.8)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 105])
    
    fig.tight_layout()
    out = output_dir / f'check_300ms_window_rec{RECORDING_NUM}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return results, out


def print_summary(results):
    """Print summary statistics"""
    print("\n" + "=" * 60)
    print("SUMMARY: Is 300ms sufficient?")
    print("=" * 60)
    
    avg_pct = np.mean([r['pct_300ms'] for r in results])
    min_pct = min([r['pct_300ms'] for r in results])
    max_pct = max([r['pct_300ms'] for r in results])
    
    print(f"Average: {avg_pct:.1f}% of events captured in 300ms")
    print(f"Range:   {min_pct:.1f}% - {max_pct:.1f}%")
    print()
    
    if min_pct >= 95:
        print("✓ 300ms captures ≥95% of events for all gestures")
        print("  → Window length is SUFFICIENT")
    elif min_pct >= 90:
        print("⚠ 300ms captures 90-95% of events")
        print("  → Window length is ACCEPTABLE but could be longer")
    else:
        print("✗ 300ms captures <90% of events for some gestures")
        print("  → Consider extending to 400ms or 500ms")
    
    print("=" * 60)


if __name__ == '__main__':
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    output_dir = Path(os.getenv("OUTPUT_DIR")) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('=' * 60)
    print(f'CHECKING 300ms EXTRACTION WINDOW (Recording #{RECORDING_NUM})')
    print('=' * 60)
    print(f'Input:   {base}')
    print(f'Output:  {output_dir}')
    print()
    
    results, out_path = analyze_300ms_window(base, output_dir)
    print_summary(results)
    
    print(f"\nPlot saved to: {out_path}")