import numpy as np
import glob
from metavision_core.event_io import EventsIterator
from pathlib import Path

def extract_event_samples_rq2(recording_folder):
    """
    RQ2: Extract samples at different temporal landmarks (t_initial, t_early, t_mid, t_late)
    using fixed 50ms window length.
    """
    # load labels
    labels = np.load(f'{recording_folder}/labels.npy', allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']
    
    # RQ2: test four temporal landmarks
    landmarks = {
        't_initial': t_initial,
        't_early': t_initial + 50_000,
        't_mid': t_initial + 100_000,
        't_late': t_initial + 200_000
    }
    
    # RQ2: fixed window length
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
    """2D event count histogram"""
    frame = np.zeros((height, width), dtype=np.float32)
    
    for ev in events:
        x, y, p = ev['x'], ev['y'], ev['p']
        frame[y, x] += 1 if p == 1 else -1
    
    return frame

if __name__ == '__main__':
    base = Path('/home/lau/Documents/test_2')
    
    for gesture in ['rock', 'paper', 'scissor']:
        for i in range(1, 21):
            folder = base / gesture / f'{gesture[0]}_{i}'
            samples = extract_event_samples_rq2(str(folder))
            np.save(folder / 'event_samples_rq2.npy', samples)
            print(f'✓ {gesture}_{i}')