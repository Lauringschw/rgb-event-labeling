from pathlib import Path
import numpy as np
from metavision_core.event_io import RawReader
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')


def extract_trigger_timestamps(raw_path: Path) -> np.ndarray:
    """Extract rising-edge external trigger timestamps from a RAW file."""
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
    """Extract and save trigger timestamps for a single recording."""
    raw_files = sorted(folder.glob("prophesee_events*.raw"))
    if not raw_files:
        print("  ✗ No RAW file found")
        return False

    try:
        timestamps = extract_trigger_timestamps(raw_files[0])
        output_path = folder / "basler_frame_timestamps.npy"
        np.save(output_path, timestamps)  # overwrite if it already exists
        print(f"  ✓ Saved to: {output_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


if __name__ == "__main__":
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))

    gesture = os.getenv("GESTURE")
    if gesture is None:
        raise ValueError("GESTURE is not set in .env")

    gesture = gesture.strip().lower()
    allowed_gestures = ["rock", "paper", "scissor", "other"]

    if gesture not in allowed_gestures:
        raise ValueError(
            f"Invalid GESTURE='{gesture}'. Must be one of: {allowed_gestures}"
        )

    prefix = gesture[0]
    total_processed = 0
    total_failed = 0

    print(f"Processing only gesture: {gesture}")

    i = 1
    while True:
        folder = base / gesture / f"{prefix}_{i}"

        if not folder.exists():
            break

        print(f"\n{gesture}/{prefix}_{i}")

        if process_recording(folder):
            total_processed += 1
        else:
            total_failed += 1

        i += 1

    print(f"\n{'='*50}")
    print(f"{gesture.upper()} - Processed: {total_processed} recordings")
    print(f"{gesture.upper()} - Failed: {total_failed} recordings")