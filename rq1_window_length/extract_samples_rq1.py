import numpy as np
import glob
from metavision_core.event_io import EventsIterator
from pathlib import Path
from dotenv import load_dotenv
import os
from multiprocessing import Pool, cpu_count

load_dotenv(Path(__file__).parent.parent / '.env')

def extract_event_samples_rq1(recording_folder):
    """
    RQ1: Extract samples at different window lengths (100ms, 150ms, 200ms)
    at t_initial landmark only.
    """
    try:
        labels_path = f'{recording_folder}/labels.npy'
        if not Path(labels_path).exists():
            return None, f"!! Skipping {recording_folder} - labels.npy not found"
        
        # load labels
        labels = np.load(labels_path, allow_pickle=True).item()
        t_initial = labels['t_initial_time_us'] # t_initial
        
        # RQ1: only t_initial landmark
        landmark = t_initial
        
        # RQ1: test three window lengths
        windows = [100_000, 150_000, 200_000]  # µs
        
        # find the Prophesee .raw event file
        event_file = Path(recording_folder) / 'prophesee_events.raw'
        if not event_file.exists():
            return None, f"!! No prophesee_events.raw in {recording_folder}"
        
        # load all events of one recording
        mv_it = EventsIterator(event_file)
        all_events = []
        for evs in mv_it:
            all_events.append(evs)
        events = np.concatenate(all_events)
        
        # extract samples at t_initial with different windows
        samples = {} # stores: {'100ms': np array of frame1, '150ms': np array of frame2, '200ms': np array of frame3}
        # np array 0.0 to 1.0 normalized. with shape 720 x 1280 
        for window_us in windows: # [100ms, 150ms, 200ms]
            # create tempral mask --> select events in the time windo
            mask = (events['t'] >= landmark) & (events['t'] < landmark + window_us)
            #       events['t'] >= t_initial AND events['t'] < t_initial + 100ms/150ms/200ms
            
            # filter events using the mask
            sample_events = events[mask] # only events in [t_initial, t_initial+window)
            
            # convert events to 2D hisotgram (720 x 1280)
            event_frame = events_to_frame(sample_events, height=720, width=1280)
            
            # store
            sample_name = f'{window_us//1000}ms' # 100_000 µs --> '100ms'
            samples[sample_name] = event_frame
        
        # save immediately as event_samples_rq1.npy
        np.save(Path(recording_folder) / 'event_samples_rq1.npy', samples)
        
        return recording_folder, "- COMPLETE!!!"
        
    except Exception as e:
        return recording_folder, f"!! Error: {str(e)}"

def events_to_frame(events, height, width):
    """2D event count histogram with normalization (following https://sci-hub.box/10.1109/aicas.2019.8771472 methodology)"""
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
        results = pool.map(extract_event_samples_rq1, all_folders)
    
    # print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    success_count = 0
    for folder, status in results:
        if folder is None:
            continue
        
        gesture = Path(folder).parent.name
        recording = Path(folder).name
        
        if status == "- COMPLETE!!!":
            success_count += 1
        
        print(f"{gesture}/{recording}: {status}")
    
    print(f"\n- Successfully processed {success_count}/{len(all_folders)} recordings")