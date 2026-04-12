# Dual Camera Recording (GUI)

This folder documents the GUI workflow for recording synchronized data from:

1. Basler RGB camera
2. Prophesee event camera

Primary script: record_dual_camera_gui.py

## What the GUI does

1. Lets you choose gesture type and recording number.
2. Shows where the next recording will be saved.
3. Captures one calibration frame first.
4. Starts synchronized recording for both cameras.
5. Runs a visible countdown (3, 2, 1, GO).
6. Automatically stops and saves recording outputs.
7. Increments the recording number for the next take.

## Recording flow

1. Single Shot
   1. Capture one calibration image for the selected recording folder.
2. Start Recording
   1. Start event recording and RGB frame capture.
   2. Run countdown and GO sequence.
   3. Continue briefly after GO and stop automatically.
3. Optional manual stop
   1. You can stop early with Stop Recording.

## Saved outputs

Each recording is saved under:

`gesture/prefix_index`

Example:

`rock/r_1`

Typical files:

1. calibration_frame.raw
2. Basler_acA1920-155um\_\_\*.raw
3. prophesee_events.raw
4. basler_frame_timestamps.npy
5. recording_metadata.npy

## Metadata summary

recording_metadata.npy stores high-level timing information for later labeling, including:

1. recording start and end timing
2. GO timing offset
3. estimated GO frame index
4. total number of captured frames

## Quick start

1. Connect both cameras.
2. Run:

```bash
python record_dual_camera_gui.py
```

3. In the GUI:
   1. Select gesture and recording number.
   2. Click Single Shot.
   3. Click Start Recording.

## Setup image

![Recording setup](/images/recording.png)
