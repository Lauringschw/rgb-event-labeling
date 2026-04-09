# Summary

1. label_tool: initiates the GO and t_initial
   1. Output: dictionary
      1. go_frame, go_time_us, t_initial_frame, t_initial_time_us, recording_folder
   2. new file: labels.npy
2. extract_sync_timestamp.py
   1. Output: 1D NumPy array of trigger timestamps
      1. Each frame now maps to the timestamp of the event timeline
   2. new file: basler_frame_timestamps.npy
3. dataset_loader: helper functions to laod/split the data
   1. Output: Each split returns:
      1. X_train, y_train — training data
      2. X_val, y_val — validation data (for hyperparameter tuning)
      3. X_test, y_test — test data (for final evaluation)
4. utils.py

## label_tool

This tool is a simple interactive GUI for manually labeling gesture timing in Basler frame recordings.

The interface displays one grayscale Basler frame from a recording sequence.

**At the top, it shows**:

- the current frame index
- the corresponding timestamp in seconds

**At the bottom, there is**:

- a **slider** to move through all frames
- a **Mark GO** button to mark the frame where the gesture happens
- a **Mark t_initial** button to mark the initial frame before the gesture
- a **Save Labels** button to store the selected labels

![Gesture Labeling Tool](/images/Screenshot%20from%202026-04-06%2021-12-32.png)

## extract_sync_timestamp.py

Extracts the synchronization mapping between your RGB camera and DVS camera by reading hardware trigger timestamps from the `.raw` file.

Keep p = 1 and discard p = 0

The Basler camera sends an electrical pulse to the DVS camera every time it captures a frame. These pulses are recorded in the DVS .raw file as trigger events.

```r
Basler captures frame 0 → sends pulse → DVS records trigger at 1843788 µs
Basler captures frame 1 → sends pulse → DVS records trigger at 1850188 µs
Basler captures frame 2 → sends pulse → DVS records trigger at 1856589 µs
...
Basler captures frame 661 → sends pulse → DVS records trigger at 6074017 µs
```

### Confirmation

**Output Example**: `rock/r_1`

```r
total unique rising edges: 662
first trigger: 1843788 µs (1.843788s)
last trigger:  6074017 µs (6.074017s)
duration: 4.230229s
First 5 triggers:
  Frame 0: 1843788 µs (1.843788s)
  Frame 1: 1850188 µs (1.850188s)
  Frame 2: 1856589 µs (1.856589s)
  Frame 3: 1862989 µs (1.862989s)
  Frame 4: 1869390 µs (1.869390s)
Last 5 triggers:
  Frame 657: 6048891 µs (6.048891s)
  Frame 658: 6055292 µs (6.055292s)
  Frame 659: 6061692 µs (6.061692s)
  Frame 660: 6068093 µs (6.068093s)
  Frame 661: 6074017 µs (6.074017s)
```

**Confirmation**:

- 662 RGB frames captured
- 4.23 second recording (6.074017s - 1.843788s = 4.230229s)
- ~6.4ms between frames (1.850188s - 1.843788s = 0.0064 and 1/0.0064 = 156.25)

**What this means**:

- Before 1843788 µs: DVS is recording, but RGB hasn't started yet → no sync
- 1843788 → 6074017 µs: Both cameras recording, fully synchronized → use this window
- After 6074017 µs: RGB stopped, DVS still recording → no sync

## dataset_loader.py

Loads event samples and splits them into train/validation/test sets for each research question.

**Each RQ has two methods**:

1. load_rqX_samples() — loads all samples from disk
2. get_rqX_split() — splits into train (70%) / val (10%) / test (20%)

### RQ1: Window Length Comparison

**Tests**: Does window length (20ms vs 30ms vs 50ms) affect accuracy?

```python
loader = GestureDataset()
rq1_data = loader.load_rq1_samples()  # loads 20ms, 30ms, 50ms samples

for window in ['20ms', '30ms', '50ms']:
    split = loader.get_rq1_split(rq1_data, window)
    # train model on split['X_train'], split['y_train']
```

### RQ2: Temperal Landmark Comparison

**Tests**: Can we predict gestures early (t_initial) or must we wait (t_late)?

```python
rq2_data = loader.load_rq2_samples()  # loads t_initial, t_early, t_mid, t_late

for landmark in ['t_initial', 't_early', 't_mid', 't_late']:
    split = loader.get_rq2_split(rq2_data, landmark)
    # train model on split['X_train'], split['y_train']
```

### RQ3: Representation Comparison

**Tests**: Which event encoding (2D histogram vs 3D voxel vs time surface) works best?

```python
rq3_data = loader.load_rq3_samples()  # loads histogram, voxel_grid, time_surface

for rep in ['histogram', 'voxel_grid', 'time_surface']:
    split = loader.get_rq3_split(rq3_data, rep)
    # train model on split['X_train'], split['y_train']
```

## utils.py
