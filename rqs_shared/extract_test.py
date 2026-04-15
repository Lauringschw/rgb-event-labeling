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

    # Tunables to keep CPU usage under control on slower machines.
    chunk_size = max(1_000, int(os.getenv("RAW_CHUNK_SIZE", "10_000")))
    loop_sleep_s = max(0.0, float(os.getenv("RAW_LOOP_SLEEP_S", "0.0005")))
    report_every_chunks = max(0, int(os.getenv("RAW_REPORT_EVERY_CHUNKS", "500")))
    chunks_processed = 0

    try:
        while not reader.is_done():
            reader.load_n_events(chunk_size)
            chunks_processed += 1

            triggers = reader.get_ext_trigger_events()
            if len(triggers) > 0:
                trigger_times.extend(t["t"] for t in triggers if t["p"] == 1)
                reader.clear_ext_trigger_events()

            if report_every_chunks and chunks_processed % report_every_chunks == 0:
                print(
                    f"\r      chunks={chunks_processed} triggers={len(trigger_times)}",
                    end="",
                    flush=True,
                )

            if loop_sleep_s > 0.0:
                time.sleep(loop_sleep_s)

        if report_every_chunks:
            print()

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