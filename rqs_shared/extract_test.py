from pathlib import Path
import numpy as np
from metavision_core.event_io import RawReader
from dotenv import load_dotenv
from multiprocessing import Process, Queue
import os

load_dotenv(Path(__file__).parent.parent / '.env')


def extract_trigger_timestamps(raw_path: Path, max_events: int = 1_000_000) -> np.ndarray:
    reader = RawReader(str(raw_path), max_events=max_events)
    trigger_times = set()

    try:
        while not reader.is_done():
            reader.load_n_events(max_events)

            triggers = reader.get_ext_trigger_events()
            if len(triggers):
                for t in triggers:
                    if t["p"] == 1:
                        trigger_times.add(int(t["t"]))
                reader.clear_ext_trigger_events()

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

        if hasattr(reader, "close"):
            try:
                reader.close()
            except Exception:
                pass


def process_recording(folder: Path, max_events: int = 1_000_000) -> bool:
    raw_files = sorted(folder.glob("prophesee_events*.raw"))
    if not raw_files:
        return False

    timestamps = extract_trigger_timestamps(raw_files[0], max_events=max_events)
    output_path = folder / "basler_frame_timestamps.npy"
    np.save(output_path, timestamps)
    print(f"  ✓ Saved to: {output_path.name}")
    return True


def worker(folder_str: str, max_events: int, q: Queue):
    try:
        ok = process_recording(Path(folder_str), max_events=max_events)
        q.put((ok, None))
    except Exception as e:
        q.put((False, str(e)))


def process_recording_isolated(folder: Path, max_events: int = 1_000_000) -> bool:
    q = Queue()
    p = Process(target=worker, args=(str(folder), max_events, q))
    p.start()
    p.join()

    if not q.empty():
        ok, err = q.get()
        if err:
            print(f"  ✗ Error: {err}")
        return ok

    print("  ✗ Error: worker crashed without returning a result")
    return False


if __name__ == "__main__":
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    gestures = ['rock', 'paper', 'scissor', 'other']

    total_processed = 0
    total_failed = 0

    MAX_EVENTS = 1_000_000

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

            if process_recording_isolated(folder, max_events=MAX_EVENTS):
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