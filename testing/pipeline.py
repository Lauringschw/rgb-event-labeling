import numpy as np
import glob
from metavision_core.event_io import EventsIterator

def extract_event_samples(recording_folder):
    # load labels
    labels = np.load(f'{recording_folder}/labels.npy', allow_pickle=True).item()
    t_initial = labels['t_initial_time_us']
    
    # define landmarks (relative to t_initial)
    landmarks = {
        't_initial': t_initial,
        't_early': t_initial + 50_000,
        't_mid': t_initial + 100_000,
        't_late': t_initial + 200_000
    }
    
    # window lengths
    windows = [20_000, 30_000, 50_000]  # µs
    
    # find the .raw event file (not basler .raw files)
    event_files = glob.glob(f'{recording_folder}/recording_*.raw')
    if not event_files:
        raise FileNotFoundError(f"no recording_*.raw found in {recording_folder}")
    
    event_file = event_files[0]  # should only be one
    print(f"loading events from {event_file}")
    
    # load all events
    mv_it = EventsIterator(event_file)
    all_events = []
    for evs in mv_it:
        all_events.append(evs)
    events = np.concatenate(all_events)
    
    print(f"total events loaded: {len(events)}")
    
    # extract samples
    samples = {}
    for landmark_name, t_landmark in landmarks.items():
        for window_us in windows:
            mask = (events['t'] >= t_landmark) & (events['t'] < t_landmark + window_us)
            sample_events = events[mask]
            
            event_frame = events_to_frame(sample_events, height=720, width=1280)
            
            sample_name = f'{landmark_name}_{window_us//1000}ms'
            samples[sample_name] = event_frame
            
            print(f"  {sample_name}: {len(sample_events)} events")
    
    return samples

def events_to_frame(events, height, width):
    frame = np.zeros((height, width), dtype=np.float32)
    
    for ev in events:
        x, y, p = ev['x'], ev['y'], ev['p']
        frame[y, x] += 1 if p == 1 else -1
    
    return frame

for gesture in ['rock', 'paper', 'scissor']:
    for i in range(1, 21):
        folder = f'/home/lau/Documents/test_1/{gesture}/{gesture[0]}_{i}'
        samples = extract_event_samples(folder)
        np.save(f'{folder}/event_samples.npy', samples)
        print(f'✓ {gesture}_{i}')