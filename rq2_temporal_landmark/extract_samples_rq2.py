import numpy as np
import glob
from metavision_core.event_io import EventsIterator
from pathlib import Path
from dotenv import load_dotenv
import os
from multiprocessing import Pool, cpu_count

load_dotenv(Path(__file__).parent.parent / '.env')

def extract_event_samples_rq2(recording_folder):
    """
    RQ2: Extract samples at different temporal landmarks using 150ms window (optimal from RQ1)
    Testing: t_initial (+0ms), t_early (+40ms), t_mid (+80ms), t_late (+120ms)
    """
    try:
        labels_path = f'{recording_folder}/labels.npy'
        if not Path(labels_path).exists():
            return None, f"⚠ No labels.npy"
        
        # load labels
        labels = np.load(labels_path, allow_pickle=True).item()
        t_initial = labels['t_initial_time_us'] # t_initial
        
        # RQ2: test four temporal landmarks with 150ms window (RQ1 optimal)
        window_us = 150_000  # 150ms
        
        # RQ2: test 4 different landmarks
        landmarks = {
            't_initial': t_initial,           # [0ms, 150ms]
            't_early': t_initial + 40_000,    # [40ms, 190ms]
            't_mid': t_initial + 80_000,      # [80ms, 230ms]
            't_late': t_initial + 120_000     # [120ms, 270ms]
        }
        
        # find the .raw event file
        event_file = Path(recording_folder) / 'prophesee_events.raw'
        if not event_file.exists():
            return None, f"!! No prophesee_events.raw in {recording_folder}"
        
        # load all events of one recording
        mv_it = EventsIterator(event_file)
        all_events = []
        for evs in mv_it:
            all_events.append(evs)
        events = np.concatenate(all_events)
        
        # extract samples at different landmarks with fixed window
        samples = {} # stores: {'t_initial': np array of frame 1, 't_early': np array of frame 2, ...}
        # np array 0.0 to 10 normalized ==> with shape 720x1280
        for landmark_name, t_landmark in landmarks.items():
        # landmark_name: 't_initial', 't_early', 't_mid', 't_late'
        # t_landmark: actual timestamp in microseconds
        
            # create temporal mask --> select events in the time window and landmark
            mask = (events['t'] >= t_landmark) & (events['t'] < t_landmark + window_us)
            #       events['t'] >= t_landmark AND events['t'] < t_landmark + 150ms
            
            # filter events using the mask
            sample_events = events[mask]
            
            # convert events to 2D histogram (720 x 1280)
            event_frame = events_to_frame(sample_events, height=720, width=1280)
            
            samples[landmark_name] = event_frame
        
        # save immediately as event_samples_rq2
        np.save(Path(recording_folder) / 'event_samples_rq2.npy', samples)
        
        return recording_folder, "- COMPLETED!"
        
    except Exception as e:
        return recording_folder, f"!! Error: {str(e)}"

def events_to_frame(events, height, width):
    """2D event count histogram with normalization (following https://sci-hub.box/10.1109/aicas.2019.8771472 methodology) [same as RQ1]"""
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
        results = pool.map(extract_event_samples_rq2, all_folders)
    
    # print summary
    print("\n" + "="*60)
    print("RQ2 EXTRACTION SUMMARY")
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
    
    print(f"\n✓ Successfully processed {success_count}/{len(all_folders)} recordings")