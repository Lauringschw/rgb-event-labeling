# rqs_shared

Shared utilities used across RQ1, RQ2, and RQ3.

## Overview

1. label_tool.py
   1. Interactive GUI to label GO and t_initial on Basler frame sequences.
   2. Saves labels to labels.npy per recording.
2. extract_sync_timestamp.py
   1. Reads DVS RAW files and extracts rising-edge external trigger timestamps.
   2. Saves per-recording timestamps to basler_frame_timestamps.npy.
3. dataset_loader.py
   1. Loads extracted event samples for each RQ.
   2. Provides stratified train/val/test splits.
4. utils.py

## label_tool.py

Manual labeling GUI for gesture recordings.

### What it loads

1. basler_frame_timestamps.npy from the selected recording folder.
2. Basler\*.raw image frames, sorted by numeric frame index.
3. Existing labels.npy if available; otherwise it falls back to recording_metadata.npy to auto-load the expected GO frame.

### UI and controls

1. Slider for frame navigation.
2. Buttons:
   1. Mark GO
   2. Mark t_initial
   3. Save Labels
   4. Save & Next
   5. Next ->
3. Keyboard shortcuts:
   1. Left/Right arrows: previous/next frame
   2. Down arrow: mark GO
   3. Up arrow: mark t_initial
   4. Shift: save and move to next recording

The selected t_initial frame is shown as a red dashed vertical marker on the slider.

### Output

Saves labels.npy with:

1. go_frame
2. go_time_us
3. t_initial_frame
4. t_initial_time_us
5. recording_folder

### Next-recording behavior

When using Save & Next or Next ->, the tool advances in this order:

1. Current gesture folder (e.g., r_1 -> r_2 -> ...)
2. Then the next gesture (rock -> paper -> scissor)

If no further folder exists, it prints that labeling is complete.

![Gesture Labeling Tool](/images/label.png)

## extract_sync_timestamp.py

Extracts synchronization timestamps from Prophesee RAW files using external trigger events.

### Core behavior

1. Opens each RAW file with RawReader.
2. Iterates through event chunks via load_n_events(...).
3. Collects only rising edges (p == 1).
4. Deduplicates and sorts timestamps.
5. Prints summary:
   1. number of unique rising edges
   2. first trigger time
   3. last trigger time
   4. total duration

If no rising-edge trigger is found, it raises a ValueError.

### Batch processing in **main**

1. Uses environment variables RECORDINGS_DIR and DIR to find the dataset root.
2. Iterates gesture folders: rock, paper, scissor.
3. Iterates recording folders by index (prefix_i pattern, currently i = 1..99).
4. Searches for prophesee_events\*.raw in each recording.
5. Saves extracted timestamps to basler_frame_timestamps.npy in that same folder.

Example interpretation:

1. One trigger timestamp corresponds to one Basler frame.
2. The synchronized RGB-DVS interval is from first trigger to last trigger.

## dataset_loader.py

Loads pre-extracted event samples for each research question and creates stratified splits.

### Initialization

1. Base path defaults to:
   1. Path(RECORDINGS_DIR) / Path(DIR)
2. Class labels:
   1. rock -> 0
   2. paper -> 1
   3. scissor -> 2

### RQ1 methods

1. load_rq1_samples()
   1. Loads event_samples_rq1.npy per recording.
   2. Returns arrays for windows: 20ms, 30ms, 50ms.
2. get_rq1_split(dataset, window, test_size=0.2, val_size=0.1)
   1. Stratified split with fixed seeds.
   2. Returns X_train, y_train, X_val, y_val, X_test, y_test.

### RQ2 methods

1. load_rq2_samples()
   1. Loads event_samples_rq2.npy per recording.
   2. Returns arrays for landmarks: t_initial, t_early, t_mid, t_late.
2. get_rq2_split(dataset, landmark, test_size=0.2, val_size=0.1)
   1. Stratified split with fixed seeds.
   2. Returns X_train, y_train, X_val, y_val, X_test, y_test.

### RQ3 methods

1. load_rq3_samples()
   1. Loads event_samples_rq3.npy per recording.
   2. Returns arrays for representations: histogram, voxel_grid, time_surface.
2. get_rq3_split(dataset, representation, test_size=0.2, val_size=0.1)
   1. Stratified split with fixed seeds.
   2. Returns X_train, y_train, X_val, y_val, X_test, y_test.
