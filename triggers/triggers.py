# extract_trigger_timestamps.py
from pathlib import Path
import numpy as np
from metavision_core.event_io import RawReader


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

                # Important: clear them, otherwise you will read them again next loop
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

    print("\nFirst 5 triggers:")
    for i, t in enumerate(trigger_times[:5]):
        print(f"  Frame {i}: {t} µs ({t / 1e6:.6f}s)")

    print("\nLast 5 triggers:")
    start = max(0, len(trigger_times) - 5)
    for i, t in enumerate(trigger_times[start:], start=start):
        print(f"  Frame {i}: {t} µs ({t / 1e6:.6f}s)")

    return trigger_times


if __name__ == "__main__":
    raw_path = Path("/home/lau/Documents/test_1/rock/r_1/recording_2026-04-02_14-31-40.raw")

    timestamps = extract_trigger_timestamps(raw_path)

    output_path = raw_path.parent / "basler_frame_timestamps.npy"
    np.save(output_path, timestamps)
    print(f"\nSaved to: {output_path}")