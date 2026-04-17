import numpy as np
import glob
from metavision_core.event_io import EventsIterator
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

def extract_event_samples_rq2(recording_folder):
    """
    RQ2: Extract samples at different temporal landmarks using 150ms window (optimal from RQ1)
    Testing: t_initial (+0ms), t_early (+40ms), t_mid (+80ms), t_late (+120ms)
    """
    # load labels
    labels = np.load(f'{recording_folder}/labels.npy', allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']
    
    # RQ2: test four temporal landmarks with 150ms window (RQ1 optimal)
    window_us = 150_000  # 150ms
    
    landmarks = {
        't_initial': t_initial,           # [0ms, 150ms]
        't_early': t_initial + 40_000,    # [40ms, 190ms]
        't_mid': t_initial + 80_000,      # [80ms, 230ms]
        't_late': t_initial + 120_000     # [120ms, 270ms]
    }
    
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
    
    # extract samples at different landmarks with fixed window
    samples = {}
    for landmark_name, t_landmark in landmarks.items():
        mask = (events['t'] >= t_landmark) & (events['t'] < t_landmark + window_us)
        sample_events = events[mask]
        
        event_frame = events_to_frame(sample_events, height=720, width=1280)
        
        samples[landmark_name] = event_frame
        
        print(f"  {landmark_name}: {len(sample_events)} events")
    
    return samples

def events_to_frame(events, height, width):
    """2D event count histogram with normalization (matching RQ1 methodology)"""
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
            samples = extract_event_samples_rq2(str(folder))
            if samples is not None:
                np.save(folder / 'event_samples_rq2.npy', samples)
                print(f'✓ {gesture}/{folder.name}')
            else:
                print(f'✗ {gesture}/{folder.name} - skipped')