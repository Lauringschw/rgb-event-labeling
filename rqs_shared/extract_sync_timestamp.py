from pathlib import Path
import numpy as np
from metavision_core.event_io import RawReader
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

def extract_trigger_timestamps(raw_path: Path) -> np.ndarray:
    """Extract rising-edge external trigger timestamps from a RAW file."""
    raw_path = Path(raw_path)

    reader = RawReader(str(raw_path))
    trigger_times = []

    try:
        while not reader.is_done():
            # Load some CD events so the reader advances through the RAW file
            reader.load_n_events(100000)

            # External trigger events accumulated since last query
            triggers = reader.get_ext_trigger_events()
            if len(triggers) > 0:
                # Keep only rising edges
                trigger_times.extend(t["t"] for t in triggers if t["p"] == 1)

                # Clear them, otherwise will read them again next loop
                reader.clear_ext_trigger_events()
    finally:
        reader.reset()

    if not trigger_times:
        raise ValueError("No rising-edge external trigger events found in RAW file.")

    trigger_times = np.array(sorted(set(trigger_times)), dtype=np.int64)

    print(f"total unique rising edges: {len(trigger_times)}")
    print(f"first trigger: {trigger_times[0]} µs ({trigger_times[0] / 1e6:.6f}s)")
    print(f"last trigger:  {trigger_times[-1]} µs ({trigger_times[-1] / 1e6:.6f}s)")
    print(f"duration: {(trigger_times[-1] - trigger_times[0]) / 1e6:.6f}s")

    duration_s = (trigger_times[-1] - trigger_times[0]) / 1e6
    fps = (len(trigger_times) - 1) / duration_s
    print(f"Actual FPS: {fps:.2f}")

    return trigger_times


if __name__ == "__main__":
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))

    # Process all recordings
    for gesture in ['rock', 'paper', 'scissor', 'other']:
        prefix = gesture[0]  # r, p, s
        
        for i in range(1, 100):  # 1 to 1000
            folder = base / gesture / f"{prefix}_{i}"
            
            # Find .raw file
            raw_files = sorted(folder.glob("prophesee_events*.raw"))
            if not raw_files:
                print(f"⚠ No .raw file in {folder}")
                continue
            
            raw_path = raw_files[0]
            print(f"\nProcessing {gesture}/{prefix}_{i}...")
            
            try:
                timestamps = extract_trigger_timestamps(raw_path)
                
                # Save
                output_path = folder / "basler_frame_timestamps.npy"
                np.save(output_path, timestamps)
                print(f"✓ Saved to: {output_path}")
                
            except Exception as e:
                print(f"✗ Error processing {folder}: {e}")