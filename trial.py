import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from metavision_core.event_io import EventsIterator
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

def visualize_clipping_from_real_data():
    """Create clipping visualization using actual recorded event data"""
    
    # Path to one of your recordings
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    
    # Pick a recording (e.g., paper gesture)
    recording_folder = base / "paper" / "p_1"
    
    # Load labels to get t_initial
    labels_path = recording_folder / 'labels.npy'
    if not labels_path.exists():
        print(f"!! Labels not found at {labels_path}")
        print("Using paper/p_147 instead...")
        recording_folder = base / "paper" / "p_147"
        labels_path = recording_folder / 'labels.npy'
    
    labels = np.load(labels_path, allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']
    
    print(f"Loading events from: {recording_folder.name}")
    print(f"t_initial: {t_initial} µs")
    
    # Load events
    event_file = recording_folder / 'prophesee_events.raw'
    mv_it = EventsIterator(str(event_file))
    all_events = []
    for evs in mv_it:
        all_events.append(evs)
    events = np.concatenate(all_events)
    
    # Extract 150ms window at t_initial
    window_us = 150_000
    mask = (events['t'] >= t_initial) & (events['t'] < t_initial + window_us)
    sample_events = events[mask]
    
    print(f"Total events in 150ms window: {len(sample_events)}")
    
    # Create histogram BEFORE clipping
    height, width = 720, 1280
    histogram_before = np.zeros((height, width), dtype=np.float32)
    
    # Count only ON events (p=1)
    for ev in sample_events:
        x, y, p = ev['x'], ev['y'], ev['p']
        if p == 1:
            histogram_before[y, x] += 1
    
    # Create histogram AFTER clipping
    histogram_after = np.clip(histogram_before, 0, 200)
    
    # Statistics
    max_before = histogram_before.max()
    max_after = histogram_after.max()
    mean_before = histogram_before.mean()
    mean_after = histogram_after.mean()
    hot_pixels = (histogram_before > 200).sum()
    
    print(f"\nStatistics:")
    print(f"  Max before: {max_before:.0f}")
    print(f"  Max after: {max_after:.0f}")
    print(f"  Mean before: {mean_before:.2f}")
    print(f"  Mean after: {mean_after:.2f}")
    print(f"  Hot pixels (>200): {hot_pixels}")
    
    # Find coordinates of hot pixels for visualization
    hot_pixel_coords = np.argwhere(histogram_before > 200)
    print(f"  Hot pixel coordinates: {len(hot_pixel_coords)} found")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Before clipping
    im1 = axes[0].imshow(histogram_before, cmap='hot', vmin=0, vmax=max_before)
    axes[0].set_title(f'Before Clipping\n(Hot pixels dominate: max={max_before:.0f})', 
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    axes[0].grid(False)
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Event Count', rotation=270, labelpad=20)
    
    # Highlight hot pixels (limit to first 50 for visibility)
    for coord in hot_pixel_coords[:50]:
        y, x = coord
        circle = plt.Circle((x, y), radius=15, color='cyan', fill=False, linewidth=1.5)
        axes[0].add_patch(circle)
    
    axes[0].text(50, height-50, f'Hot pixels: {hot_pixels}', 
                 bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7),
                 fontsize=11, color='black')
    
    # Plot 2: After clipping
    im2 = axes[1].imshow(histogram_after, cmap='hot', vmin=0, vmax=200)
    axes[1].set_title(f'After Clipping to 200\n(Outliers capped)', 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X coordinate')
    axes[1].set_ylabel('Y coordinate')
    axes[1].grid(False)
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Event Count (clipped)', rotation=270, labelpad=20)
    
    # Highlight capped pixels
    for coord in hot_pixel_coords[:50]:
        y, x = coord
        circle = plt.Circle((x, y), radius=15, color='lime', fill=False, linewidth=1.5)
        axes[1].add_patch(circle)
    
    axes[1].text(50, height-50, f'Capped at 200', 
                 bbox=dict(boxstyle='round', facecolor='lime', alpha=0.7),
                 fontsize=11, color='black')
    
    # Plot 3: Distribution
    # Only plot non-zero pixels for better visualization
    before_nonzero = histogram_before[histogram_before > 0]
    after_nonzero = histogram_after[histogram_after > 0]
    
    max_range = min(int(max_before) + 50, 500)
    axes[2].hist(before_nonzero, bins=100, alpha=0.6, label='Before clipping', 
                 color='red', edgecolor='black', range=(0, max_range))
    axes[2].hist(after_nonzero, bins=100, alpha=0.6, label='After clipping', 
                 color='green', edgecolor='black', range=(0, 200))
    axes[2].axvline(x=200, color='blue', linestyle='--', linewidth=2.5, 
                    label='Clip threshold (200)')
    
    axes[2].set_xlabel('Event Count', fontsize=12)
    axes[2].set_ylabel('Number of Pixels (log scale)', fontsize=12)
    axes[2].set_yscale('log')
    axes[2].set_title('Pixel Value Distribution\n(Active pixels only)', 
                      fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Statistics text
    stats_text = (f'Max before: {max_before:.0f}\n'
                  f'Max after: {max_after:.0f}\n'
                  f'Mean before: {mean_before:.2f}\n'
                  f'Mean after: {mean_after:.2f}\n'
                  f'Hot pixels: {hot_pixels}')
    
    axes[2].text(0.98, 0.95, stats_text, 
                 transform=axes[2].transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Add recording info
    fig.suptitle(f'Real Event Data: {recording_folder.parent.name}/{recording_folder.name} '
                 f'(150ms window at t_initial)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = Path(__file__).parent / 'real_data_clipping_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    plt.show()
    
    return histogram_before, histogram_after

if __name__ == '__main__':
    visualize_clipping_from_real_data()