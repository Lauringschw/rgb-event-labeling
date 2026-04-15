from pathlib import Path
import numpy as np
from metavision_core.event_io import RawReader
from dotenv import load_dotenv
import os
import gc

load_dotenv(Path(__file__).parent.parent / '.env')


def extract_trigger_timestamps(raw_path: Path) -> np.ndarray:
    # IMPORTANT: match max_events to your chunk size, not the 10,000,000 default
    reader = RawReader(str(raw_path), max_events=100000)
    trigger_times = set()

    try:
        while not reader.is_done():
            events = reader.load_n_events(100000)

            triggers = reader.get_ext_trigger_events()
            if len(triggers):
                for t in triggers:
                    if t["p"] == 1:
                        trigger_times.add(int(t["t"]))
                reader.clear_ext_trigger_events()

            del events
            del triggers

        if not trigger_times:
            raise ValueError("No rising-edge external trigger events found in RAW file.")

        arr = np.fromiter(sorted(trigger_times), dtype=np.int64)

        duration_s = (arr[-1] - arr[0]) / 1e6
        fps = (len(arr) - 1) / duration_s if duration_s > 0 else 0.0

        print(f"  Total triggers: {len(arr)}")
        print(f"  Duration: {duration_s:.3f}s")
        print(f"  FPS: {fps:.2f}")
        return arr

    finally:
        try:
            reader.clear_ext_trigger_events()
        except Exception:
            pass

        del reader
        gc.collect()


def process_recording(folder: Path) -> bool:
    raw_files = sorted(folder.glob("prophesee_events*.raw"))
    if not raw_files:
        return False

    try:
        timestamps = extract_trigger_timestamps(raw_files[0])
        output_path = folder / "basler_frame_timestamps.npy"
        np.save(output_path, timestamps)
        print(f"  ✓ Saved to: {output_path.name}")
        del timestamps
        gc.collect()
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        gc.collect()
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