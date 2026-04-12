import numpy as np
import glob
from metavision_core.event_io import EventsIterator
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

def extract_event_samples_rq3(recording_folder):
    """
    RQ3: Extract samples in different event representations (histogram, voxel_grid, time_surface)
    at t_initial with fixed 50ms window.
    """
    # load labels
    labels = np.load(f'{recording_folder}/labels.npy', allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']
    
    # RQ3: fixed landmark and window
    landmark = t_initial
    window_us = 50_000  # 50ms
    
    # find the .raw event file
    event_files = glob.glob(f'{recording_folder}/prophesee_events.raw')
    if not event_files:
        raise FileNotFoundError(f"no prophesee_events.raw found in {recording_folder}")
    
    event_file = event_files[0]
    print(f"loading events from {event_file}")
    
    # load all events
    mv_it = EventsIterator(event_file)
    all_events = []
    for evs in mv_it:
        all_events.append(evs)
    events = np.concatenate(all_events)
    
    print(f"total events loaded: {len(events)}")
    
    # extract events in window
    mask = (events['t'] >= landmark) & (events['t'] < landmark + window_us)
    sample_events = events[mask]
    
    print(f"  events in window: {len(sample_events)}")
    
    # RQ3: generate three different representations
    samples = {
        'histogram': events_to_histogram(sample_events, height=720, width=1280),
        'voxel_grid': events_to_voxel_grid(sample_events, height=720, width=1280, n_bins=5),
        'time_surface': events_to_time_surface(sample_events, height=720, width=1280, landmark=landmark)
    }
    
    return samples

def events_to_histogram(events, height, width):
    """2D event count histogram"""
    frame = np.zeros((height, width), dtype=np.float32)
    
    for ev in events:
        x, y, p = ev['x'], ev['y'], ev['p']
        frame[y, x] += 1 if p == 1 else -1
    
    return frame

def events_to_voxel_grid(events, height, width, n_bins=5):
    """3D voxel grid: split time window into bins"""
    if len(events) == 0:
        return np.zeros((n_bins, height, width), dtype=np.float32)
    
    voxel = np.zeros((n_bins, height, width), dtype=np.float32)
    
    t_min = events['t'].min()
    t_max = events['t'].max()
    t_range = t_max - t_min
    
    if t_range == 0:
        # all events at same time, put in first bin
        for ev in events:
            x, y, p = ev['x'], ev['y'], ev['p']
            voxel[0, y, x] += 1 if p == 1 else -1
    else:
        for ev in events:
            x, y, p, t = ev['x'], ev['y'], ev['p'], ev['t']
            # calculate which time bin this event belongs to
            bin_idx = int((t - t_min) / t_range * (n_bins - 1))
            bin_idx = min(bin_idx, n_bins - 1)  # clamp to last bin
            voxel[bin_idx, y, x] += 1 if p == 1 else -1
    
    return voxel

def events_to_time_surface(events, height, width, landmark):
    """Time surface: store time-since-landmark for latest event at each pixel"""
    surface = np.zeros((height, width), dtype=np.float32)
    
    for ev in events:
        x, y, t = ev['x'], ev['y'], ev['t']
        # store time elapsed since landmark (in ms)
        surface[y, x] = (t - landmark) / 1000.0
    
    return surface

if __name__ == '__main__':
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    
    for gesture in ['rock', 'paper', 'scissor']:
        gesture_dir = base / gesture
        if not gesture_dir.exists():
            print(f'⚠ Gesture directory {gesture_dir} not found')
            continue
        
        # Get all subfolders (recordings) in the gesture directory
        recording_folders = [f for f in gesture_dir.iterdir() if f.is_dir()]
        recording_folders.sort()  # sort to process in order
        
        for folder in recording_folders:
            samples = extract_event_samples_rq3(str(folder))
            if samples is not None:
                np.save(folder / 'event_samples_rq3.npy', samples)
                print(f'✓ {gesture}/{folder.name}')
            else:
                print(f'✗ {gesture}/{folder.name} - skipped')