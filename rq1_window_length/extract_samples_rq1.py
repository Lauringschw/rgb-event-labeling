import numpy as np
import glob
from metavision_core.event_io import EventsIterator
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

def extract_event_samples_rq1(recording_folder):
    """
    RQ1: Extract samples at different window lengths (20ms, 30ms, 50ms)
    at t_initial landmark only.
    """
    labels_path = f'{recording_folder}/labels.npy'
    if not Path(labels_path).exists():
        print(f"⚠ Skipping {recording_folder} - labels.npy not found")
        return None
    
    # load labels
    labels = np.load(labels_path, allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']
    
    # RQ1: only t_initial landmark
    landmark = t_initial
    
    # RQ1: test three window lengths
    windows = [20_000, 30_000, 50_000]  # µs
    
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
    
    # extract samples at t_initial with different windows
    samples = {}
    for window_us in windows:
        mask = (events['t'] >= landmark) & (events['t'] < landmark + window_us)
        sample_events = events[mask]
        
        event_frame = events_to_frame(sample_events, height=720, width=1280)
        
        sample_name = f'{window_us//1000}ms'
        samples[sample_name] = event_frame
        
        print(f"  {sample_name}: {len(sample_events)} events")
    
    return samples

def events_to_frame(events, height, width):
    """2D event count histogram"""
    frame = np.zeros((height, width), dtype=np.float32)
    
    for ev in events:
        x, y, p = ev['x'], ev['y'], ev['p']
        frame[y, x] += 1 if p == 1 else -1
    
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
            samples = extract_event_samples_rq1(str(folder))
            if samples is not None:
                np.save(folder / 'event_samples_rq1.npy', samples)
                print(f'✓ {gesture}/{folder.name}')
            else:
                print(f'✗ {gesture}/{folder.name} - skipped')