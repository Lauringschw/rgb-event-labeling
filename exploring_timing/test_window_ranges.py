import numpy as np
import matplotlib.pyplot as plt
import glob
from metavision_core.event_io import EventsIterator
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / '.env')

def extract_window_variations(recording_folder):
    """
    Extract event frames using different window configurations:
    - Different lengths: 20ms, 50ms, 100ms, 150ms, 200ms
    - Different offsets from t_initial: 0ms, +25ms, +50ms, +75ms
    """
    labels_path = f'{recording_folder}/labels.npy'
    if not Path(labels_path).exists():
        return None
    
    labels = np.load(labels_path, allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']
    
    # load events
    event_files = glob.glob(f'{recording_folder}/prophesee_events.raw')
    if not event_files:
        return None
    
    mv_it = EventsIterator(event_files[0])
    all_events = []
    for evs in mv_it:
        all_events.append(evs)
    events = np.concatenate(all_events)
    
    # test different configurations
    window_lengths = [20, 50, 100, 150, 200]  # ms
    offsets = [0, 25, 50, 75, 100]  # ms from t_initial
    
    results = {}
    
    for offset_ms in offsets:
        for window_ms in window_lengths:
            start_time = t_initial + (offset_ms * 1000)
            end_time = start_time + (window_ms * 1000)
            
            mask = (events['t'] >= start_time) & (events['t'] < end_time)
            sample_events = events[mask]
            
            # create event frame
            frame = np.zeros((720, 1280), dtype=np.float32)
            for ev in sample_events:
                if ev['p'] == 1:  # only ON events
                    frame[ev['y'], ev['x']] += 1
            
            # normalize
            frame = np.clip(frame, 0, 200)
            mean = frame.mean()
            std = frame.std()
            if std > 0:
                frame = (frame - mean) / (3 * std + 1e-8)
                frame = np.clip(frame, 0, 1)
            
            key = f'offset{offset_ms}ms_window{window_ms}ms'
            results[key] = {
                'frame': frame,
                'event_count': len(sample_events),
                'active_pixels': np.sum(frame > 0.1),  # pixels with significant activity
                'offset_ms': offset_ms,
                'window_ms': window_ms
            }
    
    return results

def plot_window_grid(results, recording_name, gesture):
    """
    Create grid plot showing all window configurations
    Rows = offsets, Columns = window lengths
    """
    offsets = sorted(set(r['offset_ms'] for r in results.values()))
    windows = sorted(set(r['window_ms'] for r in results.values()))
    
    fig, axes = plt.subplots(len(offsets), len(windows), 
                             figsize=(3*len(windows), 3*len(offsets)))
    
    for i, offset in enumerate(offsets):
        for j, window in enumerate(windows):
            ax = axes[i, j] if len(offsets) > 1 else axes[j]
            
            key = f'offset{offset}ms_window{window}ms'
            data = results[key]
            
            im = ax.imshow(data['frame'], cmap='hot', vmin=0, vmax=1)
            
            # title with stats
            title = f"offset={offset}ms\nwindow={window}ms"
            stats = f"{data['event_count']} events\n{data['active_pixels']} pixels"
            ax.set_title(title, fontsize=8)
            ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                   fontsize=6, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            ax.axis('off')
    
    fig.suptitle(f'{gesture}/{recording_name}\nWindow Configuration Exploration', 
                 fontsize=12)
    plt.tight_layout()
    
    return fig

def summarize_configurations(all_results):
    """
    Summarize which configurations capture most events/information
    """
    summary = {}
    
    for recording, results in all_results.items():
        gesture = recording.split('/')[0]
        
        if gesture not in summary:
            summary[gesture] = {
                'event_counts': {},
                'active_pixels': {}
            }
        
        for key, data in results.items():
            if key not in summary[gesture]['event_counts']:
                summary[gesture]['event_counts'][key] = []
                summary[gesture]['active_pixels'][key] = []
            
            summary[gesture]['event_counts'][key].append(data['event_count'])
            summary[gesture]['active_pixels'][key].append(data['active_pixels'])
    
    # compute averages
    for gesture in summary:
        for key in summary[gesture]['event_counts']:
            summary[gesture]['event_counts'][key] = np.mean(summary[gesture]['event_counts'][key])
            summary[gesture]['active_pixels'][key] = np.mean(summary[gesture]['active_pixels'][key])
    
    return summary

def plot_configuration_heatmap(summary, metric='event_counts'):
    """
    Create heatmap showing average event counts or active pixels
    for each configuration across gestures
    """
    offsets = [0, 25, 50, 75, 100]
    windows = [20, 50, 100, 150, 200]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, gesture in enumerate(['rock', 'paper', 'scissor']):
        if gesture not in summary:
            continue
        
        # create matrix
        matrix = np.zeros((len(offsets), len(windows)))
        for i, offset in enumerate(offsets):
            for j, window in enumerate(windows):
                key = f'offset{offset}ms_window{window}ms'
                matrix[i, j] = summary[gesture][metric].get(key, 0)
        
        ax = axes[idx]
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        
        ax.set_xticks(range(len(windows)))
        ax.set_xticklabels([f'{w}ms' for w in windows])
        ax.set_yticks(range(len(offsets)))
        ax.set_yticklabels([f'+{o}ms' for o in offsets])
        
        ax.set_xlabel('Window Length')
        ax.set_ylabel('Offset from t_initial')
        ax.set_title(f'{gesture.upper()}\nAvg {metric.replace("_", " ")}')
        
        # add text annotations
        for i in range(len(offsets)):
            for j in range(len(windows)):
                text = ax.text(j, i, f'{matrix[i, j]:.0f}',
                             ha="center", va="center", color="white", fontsize=8)
        
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    
    print("="*60)
    print("WINDOW CONFIGURATION EXPLORATION")
    print("="*60)
    
    all_results = {}
    
    # test on first 3 recordings per gesture
    for gesture in ['rock', 'paper', 'scissor']:
        gesture_dir = base / gesture
        if not gesture_dir.exists():
            continue
        
        recording_folders = sorted([f for f in gesture_dir.iterdir() if f.is_dir()])[:3]
        
        print(f"\n{gesture}:")
        for folder in recording_folders:
            results = extract_window_variations(str(folder))
            if results is not None:
                recording_name = f"{gesture}/{folder.name}"
                all_results[recording_name] = results
                
                # plot individual grid
                fig = plot_window_grid(results, folder.name, gesture)
                output_path = Path('exploratory_timing_plots') / 'windows' / f'{gesture}_{folder.name}.png'
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"  ✓ {folder.name}")
    
    # create summary heatmaps
    print("\nGenerating summary heatmaps...")
    summary = summarize_configurations(all_results)
    
    fig1 = plot_configuration_heatmap(summary, metric='event_counts')
    fig1.savefig('exploratory_timing_plots/summary_event_counts.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("  ✓ Event counts heatmap")
    
    fig2 = plot_configuration_heatmap(summary, metric='active_pixels')
    fig2.savefig('exploratory_timing_plots/summary_active_pixels.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("  ✓ Active pixels heatmap")
    
    print("\n" + "="*60)
    print("Window exploration complete. Check exploratory_timing_plots/ folder")
    print("="*60)