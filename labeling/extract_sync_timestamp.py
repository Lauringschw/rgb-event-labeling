from pathlib import Path
import numpy as np
from metavision_core.event_io import RawReader
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

def extract_trigger_timestamps(raw_path: Path) -> np.ndarray:
    """Extract rising-edge external trigger timestamps from RAW file."""
    reader = RawReader(str(raw_path)) # reading .raw files
    trigger_times = []

    try:
        while not reader.is_done():
            reader.load_n_events(100000) # read next 100,000 regular events into memory
            
            triggers = reader.get_ext_trigger_events() # gets the external triggers
            if len(triggers) > 0: # external triggers exist
                # stores only rising edges (p == 1) into array
                trigger_times.extend(t["t"] for t in triggers if t["p"] == 1) # stores µs timestamps of each rising-edge trigger pulse
                reader.clear_ext_trigger_events() # clears memory for next 100,000
    finally:
        reader.reset()

    if not trigger_times:
        raise ValueError("No rising-edge external trigger events found in RAW file")

    # remove duplicate and sort by trigger time ==> SDK duplicate bug
    trigger_times = np.array(sorted(set(trigger_times)), dtype=np.int64)

    # 1 trigger = 1 frame
    duration_s = (trigger_times[-1] - trigger_times[0]) / 1e6 # duration in seconds = last - first
    fps = (len(trigger_times) - 1) / duration_s # fps = # of frames / duration
    
    print(f"  Total triggers: {len(trigger_times)}")
    print(f"  Duration: {duration_s:.3f}s")
    print(f"  FPS: {fps:.2f}")

    return trigger_times # Basler frame capture times, encoded as Prophesee DVS timestamps

def process_recording(folder: Path) -> bool:
    """Extract and save trigger timestamps for a single recording."""
    raw_file = folder / "prophesee_events.raw"
    if not raw_file.exists():
        return False

    try:
        timestamps = extract_trigger_timestamps(raw_file)
        output_path = folder / "basler_frame_timestamps.npy"
        np.save(output_path, timestamps)
        print(f"  - Saved to: {output_path.name}")
        return True
    except Exception as e:
        print(f"  !! Error: {e}")
        return False


if __name__ == "__main__":
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    gestures = ['rock', 'paper', 'scissor', 'other']
    
    total_processed = 0
    total_failed = 0
    
    for gesture in gestures:
        prefix = gesture[0]
        gesture_processed = 0
        gesture_failed = 0
        
        i = 1
        while True:
            folder = base / gesture / f"{prefix}_{i}"
            
            if not folder.exists():
                break
            
            print(f"\n{gesture}/{prefix}_{i}")
            
            if process_recording(folder):
                total_processed += 1
                gesture_processed += 1
            else:
                total_failed += 1
                gesture_failed += 1
            
            i += 1
        
        print(f"\n{gesture.upper()}: {gesture_processed} processed, {gesture_failed} failed")
    
    print(f"\n{'='*50}")
    print(f"TOTAL - Processed: {total_processed} recordings")
    print(f"TOTAL - Failed: {total_failed} recordings")