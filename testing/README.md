# Testing Utilities

This folder contains quick scripts to inspect recorded RGB/event data and timing metadata.

## Scripts Overview

1. test_rgb.py
   1. Loads one Basler raw frame and displays it in grayscale.
   2. Useful for checking image quality and frame indexing.
2. test_dvs.py
   1. Reads one Prophesee event file and summarizes event timeline.
   2. Prints total events, first timestamp, last timestamp, and duration.

## Quick usage

Run from the project root:

```bash
python3 testing/test_rgb.py
python3 testing/test_dvs.py
```

## Typical outputs

1. test_rgb.py
   1. One plotted grayscale frame window.
2. test_dvs.py
   1. Event count and recording duration in seconds.
