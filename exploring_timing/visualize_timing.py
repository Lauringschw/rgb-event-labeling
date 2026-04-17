import numpy as np
import matplotlib.pyplot as plt
import glob
from metavision_core.event_io import EventsIterator
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / '.env')

def analyze_recording_timing(recording_folder, window_size_ms=10):
    """
    Analyze event distribution over time for a single recording
    Returns event counts in sliding windows from GO to end
    """
    labels_path = f'{recording_folder}/labels.npy'
    if not Path(labels_path).exists():
        print(f"⚠ Skipping {recording_folder} - labels.npy not found")
        return None
    
    # load labels
    labels = np.load(labels_path, allow_pickle=True).item()
    t_go = labels['go_frame_time_us']
    t_initial = labels['t_initial_time_us']
    
    # find the .raw event file
    event_files = glob.glob(f'{recording_folder}/prophesee_events.raw')
    if not event_files:
        print(f"⚠ No .raw file found in {recording_folder}")
        return None
    
    event_file = event_files[0]
    
    # load all events
    mv_it = EventsIterator(event_file)
    all_events = []
    for evs in mv_it:
        all_events.append(evs)
    events = np.concatenate(all_events)
    
    # focus on events from GO to GO+500ms (typical gesture duration)
    mask = (events['t'] >= t_go) & (events['t'] < t_go + 500_000)
    gesture_events = events[mask]
    
    if len(gesture_events) == 0:
        print(f"⚠ No events in gesture window for {recording_folder}")
        return None
    
    # create sliding windows
    window_us = window_size_ms * 1000
    start_time = t_go
    end_time = t_go + 500_000
    
    window_centers = []
    event_counts = []
    
    current_time = start_time
    while current_time < end_time:
        mask = (gesture_events['t'] >= current_time) & (gesture_events['t'] < current_time + window_us)
        count = np.sum(mask)
        
        window_centers.append((current_time - t_go) / 1000)  # convert to ms from GO
        event_counts.append(count)
        
        current_time += window_us
    
    # mark important timestamps
    t_initial_offset = (t_initial - t_go) / 1000  # ms from GO
    
    return {
        'window_centers': np.array(window_centers),
        'event_counts': np.array(event_counts),
        't_initial_offset': t_initial_offset,
        'recording': recording_folder
    }

def plot_timing_analysis(results, gesture_name):
    """Plot event distribution over time for multiple recordings of same gesture"""
    
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3*len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        # plot event counts
        ax.plot(result['window_centers'], result['event_counts'], 'b-', alpha=0.7)
        ax.fill_between(result['window_centers'], result['event_counts'], alpha=0.3)
        
        # mark t_initial
        ax.axvline(result['t_initial_offset'], color='red', linestyle='--', 
                   linewidth=2, label=f"t_initial ({result['t_initial_offset']:.1f}ms)")
        
        # mark potential window regions
        # 20ms, 30ms, 50ms windows from t_initial
        for window_ms, color in [(20, 'orange'), (30, 'green'), (50, 'purple')]:
            ax.axvspan(result['t_initial_offset'], 
                      result['t_initial_offset'] + window_ms,
                      alpha=0.15, color=color, 
                      label=f"{window_ms}ms window")
        
        ax.set_ylabel('Event count\n(per 10ms)')
        ax.set_title(f"{Path(result['recording']).name}")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time from GO signal (ms)')
    fig.suptitle(f'{gesture_name.upper()} - Event Distribution Over Time', fontsize=14, y=0.995)
    plt.tight_layout()
    
    return fig

def compare_landmarks(recording_folder):
    """
    Compare event frames at different temporal landmarks
    Helps determine which landmark has most discriminative features
    """
    labels_path = f'{recording_folder}/labels.npy'
    if not Path(labels_path).exists():
        return None
    
    labels = np.load(labels_path, allow_pickle=True).item()
    
    # get all landmarks if they exist
    landmarks = {}
    for key in ['t_initial_time_us', 't_early_time_us', 't_mid_time_us', 't_late_time_us']:
        if key in labels:
            landmarks[key.replace('_time_us', '')] = labels[key]
    
    if len(landmarks) == 0:
        return None
    
    # load events
    event_files = glob.glob(f'{recording_folder}/prophesee_events.raw')
    if not event_files:
        return None
    
    mv_it = EventsIterator(event_files[0])
    all_events = []
    for evs in mv_it:
        all_events.append(evs)
    events = np.concatenate(all_events)
    
    # extract 50ms windows from each landmark
    window_us = 50_000
    frames = {}
    
    for name, timestamp in landmarks.items():
        mask = (events['t'] >= timestamp) & (events['t'] < timestamp + window_us)
        sample_events = events[mask]
        
        frame = np.zeros((720, 1280), dtype=np.float32)
        for ev in sample_events:
            if ev['p'] == 1:  # only ON events
                frame[ev['y'], ev['x']] += 1
        
        frames[name] = frame
    
    return frames, landmarks

def plot_landmark_comparison(frames_dict, recording_name):
    """Plot event frames from different landmarks side by side"""
    
    n_landmarks = len(frames_dict)
    fig, axes = plt.subplots(1, n_landmarks, figsize=(5*n_landmarks, 5))
    if n_landmarks == 1:
        axes = [axes]
    
    for idx, (name, frame) in enumerate(frames_dict.items()):
        ax = axes[idx]
        im = ax.imshow(frame, cmap='hot', vmin=0, vmax=np.percentile(frame, 99))
        ax.set_title(f'{name.replace("_", " ")}\n({np.sum(frame > 0)} active pixels)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle(f'{recording_name} - 50ms windows at different landmarks', fontsize=14)
    plt.tight_layout()
    
    return fig

if __name__ == '__main__':
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    
    print("="*60)
    print("EXPLORATORY TIMING ANALYSIS")
    print("="*60)
    
    # analyze timing for each gesture (first 5 recordings per gesture)
    for gesture in ['rock', 'paper', 'scissor']:
        gesture_dir = base / gesture
        if not gesture_dir.exists():
            continue
        
        recording_folders = sorted([f for f in gesture_dir.iterdir() if f.is_dir()])[:5]
        
        print(f"\nAnalyzing {gesture}...")
        results = []
        
        for folder in recording_folders:
            result = analyze_recording_timing(str(folder), window_size_ms=10)
            if result is not None:
                results.append(result)
        
        if results:
            fig = plot_timing_analysis(results, gesture)
            output_path = Path('exploratory_timing_plots') / f'{gesture}_timing.png'
            output_path.parent.mkdir(exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Saved timing plot to {output_path}")
    
    # compare landmarks for a few samples
    print("\n" + "="*60)
    print("LANDMARK COMPARISON")
    print("="*60)
    
    for gesture in ['rock', 'paper', 'scissor']:
        gesture_dir = base / gesture
        if not gesture_dir.exists():
            continue
        
        recording_folders = sorted([f for f in gesture_dir.iterdir() if f.is_dir()])[:3]
        
        for folder in recording_folders:
            result = compare_landmarks(str(folder))
            if result is not None:
                frames, landmarks = result
                fig = plot_landmark_comparison(frames, f"{gesture}/{folder.name}")
                output_path = Path('exploratory_timing_plots') / 'landmarks' / f'{gesture}_{folder.name}.png'
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ {gesture}/{folder.name}")
    
    print("\n" + "="*60)
    print("Analysis complete. Check exploratory_timing_plots/ folder")
    print("="*60)