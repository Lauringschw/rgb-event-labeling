from pathlib import Path
import numpy as np
from metavision_core.event_io import RawReader
from dotenv import load_dotenv
import os
import time

load_dotenv(Path(__file__).parent.parent / ".env")


def extract_trigger_timestamps(raw_path: Path) -> np.ndarray:
    """Extract rising-edge external trigger timestamps from a RAW file."""
    reader = RawReader(str(raw_path))
    trigger_times = []

    prev_trigger_count = 0
    stalled_iterations = 0
    max_stalled = 10

    # smaller chunk = less CPU / memory pressure
    chunk_size = 20_000

    try:
        while not reader.is_done():
            reader.load_n_events(chunk_size)

            triggers = reader.get_ext_trigger_events()

            if len(triggers) > 0:
                trigger_times.extend(t["t"] for t in triggers if t["p"] == 1)
                reader.clear_ext_trigger_events()
            else:
                # no triggers found in this chunk
                pass

            current_trigger_count = len(trigger_times)

            # only increment stall counter once per loop
            if current_trigger_count == prev_trigger_count:
                stalled_iterations += 1
            else:
                stalled_iterations = 0

            prev_trigger_count = current_trigger_count

            if stalled_iterations >= max_stalled:
                print(f"  Warning: no new trigger events for {max_stalled} iterations, stopping.")
                break

            # tiny pause to keep system responsive
            time.sleep(0.001)

    finally:
        reader.reset()

    if not trigger_times:
        raise ValueError("No rising-edge external trigger events found in RAW file.")

    trigger_times = np.unique(np.array(trigger_times, dtype=np.int64))

    duration_s = (trigger_times[-1] - trigger_times[0]) / 1e6
    fps = (len(trigger_times) - 1) / duration_s if duration_s > 0 else 0.0

    print(f"  Total triggers: {len(trigger_times)}")
    print(f"  Duration: {duration_s:.3f}s")
    print(f"  FPS: {fps:.2f}")

    return trigger_times


def process_recording(folder: Path) -> bool:
    """Extract and save trigger timestamps for a single recording."""
    raw_files = sorted(folder.glob("prophesee_events*.raw"))
    if not raw_files:
        return False

    try:
        print(f"    Processing {raw_files[0].name}...", end=" ", flush=True)
        timestamps = extract_trigger_timestamps(raw_files[0])
        output_path = folder / "basler_frame_timestamps.npy"
        np.save(output_path, timestamps)
        print(f"✓ Saved {len(timestamps)} triggers")
        return True
    except ValueError as e:
        print(f"✗ {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    gestures = ["rock", "paper", "scissor", "other"]

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

    print(f"\n{'=' * 50}")
    print(f"TOTAL - Processed: {total_processed} recordings")
    print(f"TOTAL - Failed: {total_failed} recordings")