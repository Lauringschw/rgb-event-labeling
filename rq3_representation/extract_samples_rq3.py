import numpy as np
import glob
from metavision_core.event_io import EventsIterator
from pathlib import Path
from dotenv import load_dotenv
import os
from multiprocessing import Pool, cpu_count

load_dotenv(Path(__file__).parent.parent / '.env')

def extract_event_samples_rq3(recording_folder):
    """
    RQ3: Extract samples in different event representations (histogram, voxel_grid, time_surface)
    at t_initial with 150ms window (optimal from RQ1).
    """
    try:
        labels_path = f'{recording_folder}/labels.npy'
        if not Path(labels_path).exists():
            return None, f"!! Skipping {recording_folder} - labels.npy not found"
        
        # load labels
        labels = np.load(labels_path, allow_pickle=True).item()
        t_initial = labels['t_initial_time_us'] # t_initial
        
        # RQ3: fixed landmark and window (150ms from RQ1)
        landmark = t_initial
        window_us = 150_000  # 150ms (optimal from RQ1)
        
         # find the Prophesee .raw event file
        event_file = Path(recording_folder) / 'prophesee_events.raw'
        if not event_file.exists():
            return None, f"!! No prophesee_events.raw in {recording_folder}"
        
        # load all events
        mv_it = EventsIterator(event_file)
        all_events = []
        for evs in mv_it:
            all_events.append(evs)
        events = np.concatenate(all_events)
        
        # create temporal mask --> select events in the time windo
        mask = (events['t'] >= landmark) & (events['t'] < landmark + window_us)
        #       events['t'] >= t_initial AND events['t'] < t_initial + 150ms
        
        # filter events using the mask
        sample_events = events[mask] # only events in [t_initial, t_initial+window)
        
        # RQ3: generate three different representations
        samples = {
            'histogram': events_to_histogram(sample_events, height=720, width=1280),
            'voxel_grid': events_to_voxel_grid(sample_events, height=720, width=1280, n_bins=5),
            'time_surface': events_to_time_surface(sample_events, height=720, width=1280, landmark=landmark)
        }
        
        # save immediately
        np.save(Path(recording_folder) / 'event_samples_rq3.npy', samples)
        
        return recording_folder, "- COMPLETED!"
        
    except Exception as e:
        return recording_folder, f"✗ Error: {str(e)}"

def events_to_histogram(events, height, width):
    """2D event count histogram with normalization (matching RQ1/RQ2)"""
    frame = np.zeros((height, width), dtype=np.float32)
    
    # accumulate events (only positive/ON events as per paper)
    for ev in events:
        x, y, p = ev['x'], ev['y'], ev['p']
        if p == 1:  # only count ON events
            frame[y, x] += 1
    
    # clip maximum events per pixel to 200 (as per paper)
    frame = np.clip(frame, 0, 200)
    
    # 3-sigma normalization to [0, 1] range
    mean = frame.mean()
    std = frame.std()
    if std > 0:
        frame = (frame - mean) / (3 * std + 1e-8)
        frame = np.clip(frame, 0, 1)
    
    return frame

def events_to_voxel_grid(events, height, width, n_bins=5):
    """3D voxel grid: split time window into bins"""
    if len(events) == 0:
        return np.zeros((n_bins, height, width), dtype=np.float32)
    
    # 3D array: 5 time bins x 720 height x 1280 width
    voxel = np.zeros((n_bins, height, width), dtype=np.float32)
    
    t_min = events['t'].min() # t_initial
    t_max = events['t'].max() # last event
    t_range = t_max - t_min # total length
    
    if t_range == 0:
        # all events at same time, put in first bin
        for ev in events:
            x, y, p = ev['x'], ev['y'], ev['p']
            voxel[0, y, x] += 1 if p == 1 else -1 # all into voxel 1
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
    
    # 2D grid (720x1280)
    surface = np.zeros((height, width), dtype=np.float32)
    
    # each new event overwrites the previous value
    # only the last event matters
    for ev in events:
        x, y, t = ev['x'], ev['y'], ev['t']
        # store time elapsed since landmark (in ms)
        surface[y, x] = (t - landmark) / 1000.0
    
    return surface

if __name__ == '__main__':
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    
    # collect all recording folders
    all_folders = []
    for gesture in ['rock', 'paper', 'scissor']:
        gesture_dir = base / gesture
        if not gesture_dir.exists():
            print(f'!! Gesture directory {gesture_dir} not found')
            continue
        
        recording_folders = [str(f) for f in gesture_dir.iterdir() if f.is_dir()]
        all_folders.extend(recording_folders)
    
    all_folders.sort()
    
    print(f"Found {len(all_folders)} recordings to process")
    print(f"Using {cpu_count()} CPU cores for parallel processing\n")
    
    # process in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(extract_event_samples_rq3, all_folders)
    
    # print summary
    print("\n" + "="*60)
    print("RQ3 EXTRACTION SUMMARY")
    print("="*60)
    
    success_count = 0
    for folder, status in results:
        if folder is None:
            continue
        
        gesture = Path(folder).parent.name
        recording = Path(folder).name
        
        if status == "- COMPLETED!":
            success_count += 1
        
        print(f"{gesture}/{recording}: {status}")
    
    print(f"\n- Successfully processed {success_count}/{len(all_folders)} recordings")