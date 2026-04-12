# Testing Utilities

This folder contains quick scripts to inspect recorded RGB/event data and timing metadata.

## Scripts Overview

1. test_rgb.py
   1. Loads one Basler raw frame and displays it in grayscale.
   2. Useful for checking image quality and frame indexing.
2. test_dvs.py
   1. Reads one Prophesee event file and summarizes event timeline.
   2. Prints total events, first timestamp, last timestamp, and duration.
3. npy.py
   1. Loads basler_frame_timestamps.npy and prints its values.
   2. Useful for verifying frame-to-time mapping exists.
4. go_getter.py
   1. Loads recording_metadata.npy and prints GO-related information.
   2. Useful for confirming expected GO timing/frame before labeling.
5. preview_extracted_samples.py
   1. Loads generated event sample files (`event_samples_rq1.npy`, `event_samples_rq2.npy`, `event_samples_rq3.npy`).
   2. Saves a few preview PNGs to `SAMPLES_DIR/previews/...`.
   3. Optional: open interactive matplotlib windows to inspect the samples.

## Quick usage

Run from the project root:

```bash
python testing/test_rgb.py
python testing/test_dvs.py
python testing/npy.py
python testing/go_getter.py
python testing/preview_extracted_samples.py --rq rq1 --num 6
python testing/preview_extracted_samples.py --rq all --num 4 --save-only
```

## Typical outputs

1. test_rgb.py
   1. One plotted grayscale frame window.
2. test_dvs.py
   1. Event count and recording duration in seconds.
3. npy.py
   1. NumPy array of Basler frame timestamps.
4. go_getter.py
   1. Metadata dictionary and GO summary fields.
5. preview_extracted_samples.py
   1. Saved PNG previews in `SAMPLES_DIR/previews/rq1`, `rq2`, `rq3`.
   2. Interactive plots if `--save-only` is not set.
