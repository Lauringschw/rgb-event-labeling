from pathlib import Path
import numpy as np
from metavision_core.event_io import RawReader
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')


def extract_trigger_timestamps(raw_path: Path) -> np.ndarray:
    reader = RawReader(str(raw_path))
    trigger_times = []

    try:
        while not reader.is_done():
            reader.load_n_events(100000)

            triggers = reader.get_ext_trigger_events()
            if len(triggers) > 0:
                trigger_times.extend(int(t["t"]) for t in triggers if t["p"] == 1)
                reader.clear_ext_trigger_events()

    finally:
        try:
            reader.clear_ext_trigger_events()
        except Exception:
            pass

        if hasattr(reader, "close"):
            try:
                reader.close()
            except Exception:
                pass

        del reader

    if not trigger_times:
        raise ValueError("No rising-edge external trigger events found in RAW file.")

    trigger_times = np.array(sorted(set(trigger_times)), dtype=np.int64)

    duration_s = (trigger_times[-1] - trigger_times[0]) / 1e6
    fps = (len(trigger_times) - 1) / duration_s if duration_s > 0 else 0.0

    print(f"  Total triggers: {len(trigger_times)}")
    print(f"  Duration: {duration_s:.3f}s")
    print(f"  FPS: {fps:.2f}")

    return trigger_times


def process_recording(folder: Path) -> bool:
    raw_files = sorted(folder.glob("prophesee_events*.raw"))
    if not raw_files:
        return False

    output_path = folder / "basler_frame_timestamps.npy"
    if output_path.exists():
        print("  - Already processed, skipping")
        return True

    try:
        timestamps = extract_trigger_timestamps(raw_files[0])
        np.save(output_path, timestamps)
        print(f"  ✓ Saved to: {output_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


if __name__ == "__main__":
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    gestures = ['rock', 'paper', 'scissor', 'other']

    BATCH_SIZE = 100
    processed_this_run = 0

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

            if processed_this_run >= BATCH_SIZE:
                print(f"\nReached batch limit of {BATCH_SIZE}. Restart the script to continue.")
                raise SystemExit(0)

            print(f"\n{gesture}/{prefix}_{i}")

            if process_recording(folder):
                total_processed += 1
                gesture_processed += 1
                processed_this_run += 1
            else:
                total_failed += 1
                gesture_failed += 1
                processed_this_run += 1

            i += 1

        print(f"\n{gesture.upper()}: {gesture_processed} processed, {gesture_failed} failed")

    print(f"\n{'='*50}")
    print(f"TOTAL - Processed: {total_processed} recordings")
    print(f"TOTAL - Failed: {total_failed} recordings")