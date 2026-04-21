import numpy as np
import matplotlib.pyplot as plt
from metavision_core.event_io import EventsIterator
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / '.env')


def events_to_histogram(events, height=720, width=1280):
    """2D event count histogram with normalization (matching RQ1 pipeline)"""
    frame = np.zeros((height, width), dtype=np.float32)
    
    # accumulate events (only positive/ON events)
    for ev in events:
        x, y, p = ev['x'], ev['y'], ev['p']
        if p == 1:  # only count ON events
            frame[y, x] += 1
    
    # clip maximum events per pixel to 200
    frame = np.clip(frame, 0, 200)
    
    # 3-sigma normalization to [0, 1] range
    mean = frame.mean()
    std = frame.std()
    if std > 0:
        frame = (frame - mean) / (3 * std + 1e-8)
        frame = np.clip(frame, 0, 1)
    
    return frame


def extract_window_histograms(recording_folder):
    """
    Extract histograms at t_initial for 100ms, 150ms, 200ms windows
    Returns dict with frames and statistics
    """
    try:
        labels_path = f'{recording_folder}/labels.npy'
        if not Path(labels_path).exists():
            return None
        
        # load labels
        labels = np.load(labels_path, allow_pickle=True).item()
        t_initial = labels['t_initial_time_us']
        
        # find event file
        event_file = Path(recording_folder) / 'prophesee_events.raw'
        if not event_file.exists():
            return None
        
        # load all events
        mv_it = EventsIterator(str(event_file))
        all_events = []
        for evs in mv_it:
            all_events.append(evs)
        events = np.concatenate(all_events)
        
        # extract histograms for each window length
        windows = {
            '100ms': 100_000,
            '150ms': 150_000,
            '200ms': 200_000
        }
        
        results = {}
        
        for name, window_us in windows.items():
            # create temporal mask
            mask = (events['t'] >= t_initial) & (events['t'] < t_initial + window_us)
            sample_events = events[mask]
            
            # generate histogram
            frame = events_to_histogram(sample_events)
            
            # compute statistics
            event_count = len(sample_events)
            active_pixels = np.sum(frame > 0.1)  # pixels with significant activity
            mean_intensity = frame.mean()
            max_intensity = frame.max()
            
            results[name] = {
                'frame': frame,
                'event_count': event_count,
                'active_pixels': active_pixels,
                'mean_intensity': mean_intensity,
                'max_intensity': max_intensity,
                'window_us': window_us
            }
        
        return results
        
    except Exception as e:
        print(f"Error processing {recording_folder}: {e}")
        return None


if __name__ == '__main__':
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    output_dir = Path(os.getenv("EXPLORATION_DIR")) / "rq1_paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("RQ1 WINDOW LENGTH VISUALIZATION FOR PAPER")
    print("="*60)
    print(f"Input: {base}")
    print(f"Output: {output_dir}")
    print()
    
    # create single paper figure: 3 gestures x 3 windows = 9 subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    window_names = ['100ms', '150ms', '200ms']
    gestures = ['rock', 'paper', 'scissor']
    
    for row_idx, gesture in enumerate(gestures):
        gesture_dir = base / gesture
        recording_folders = sorted([f for f in gesture_dir.iterdir() if f.is_dir()])
        
        # use first recording
        if recording_folders:
            results = extract_window_histograms(str(recording_folders[0]))
            
            if results is not None:
                for col_idx, window in enumerate(window_names):
                    ax = axes[row_idx, col_idx]
                    data = results[window]
                    
                    # plot histogram
                    im = ax.imshow(data['frame'], cmap='hot', vmin=0, vmax=1, aspect='auto')
                    
                    # column titles (only first row)
                    if row_idx == 0:
                        ax.set_title(f'{window}', fontsize=14, fontweight='bold')
                    
                    # row labels (only first column)
                    if col_idx == 0:
                        ax.set_ylabel(gesture.upper(), fontsize=14, fontweight='bold', rotation=0, labelpad=40)
                    
                    # stats annotation in bottom-left corner
                    stats = f"{data['event_count']:,} events"
                    ax.text(0.02, 0.02, stats,
                           transform=ax.transAxes,
                           fontsize=9,
                           verticalalignment='bottom',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.axis('off')
                    
                    # colorbar (only last column)
                    if col_idx == 2:
                        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label('Intensity', rotation=270, labelpad=15, fontsize=10)
                
                print(f"{gesture.upper()}: {recording_folders[0].name}")
    
    fig.suptitle('RQ1: Effect of Event Window Length on Histogram Representation', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    output_path = output_dir / 'rq1_paper_figure_3x3.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Paper figure saved to {output_path}")
    
    print()
    print("="*60)
    print(f"Done. Check {output_dir}/")
    print("="*60)