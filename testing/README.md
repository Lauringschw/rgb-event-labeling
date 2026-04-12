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

## Quick usage

Run from the project root:

```bash
python testing/test_rgb.py
python testing/test_dvs.py
python testing/npy.py
python testing/go_getter.py
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
