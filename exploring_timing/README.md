# Exploring Timing

This folder contains exploratory scripts for understanding when event data becomes most informative after the GO signal. The goal is to help choose better temporal windows and landmark offsets for the experiments in the thesis.

Both scripts read recordings from the dataset structure defined in the project `.env` file and expect each recording folder to contain at least:

- `labels.npy`
- `prophesee_events.raw`

## Scripts

### `test_window_ranges.py`

This script explores different event-window configurations around `t_initial`.

What it does:

- loads each recording's `labels.npy` and extracts `t_initial_time_us`
- reads all events from `prophesee_events.raw`
- tests multiple window lengths: `20ms`, `50ms`, `100ms`, `150ms`, and `200ms`
- tests multiple offsets from `t_initial`: `0ms`, `+25ms`, `+50ms`, `+75ms`, and `+100ms`
- converts each event slice into a simple event frame
- saves one grid image per recording plus summary heatmaps per gesture

Output:

- `windows/<gesture>_<recording>.png`
- `summary_event_counts.png`
- `summary_active_pixels.png`

Use this script when you want to compare which time window captures the strongest gesture signal and which offset is most useful.

### `visualize_timing.py`

This script analyzes the temporal distribution of events after the GO signal and compares different temporal landmarks.

What it does:

- loads `go_time_us` and `t_initial_time_us` from `labels.npy`
- builds a 10ms sliding-window event count series from GO to GO + 500ms
- plots event counts over time for each recording
- marks `t_initial` on the timeline
- overlays example window regions of `20ms`, `30ms`, and `50ms`
- compares 50ms event frames at temporal landmarks such as `t_initial`, `t_early`, `t_mid`, and `t_late`
- saves timing plots and landmark comparison images

Output:

- `<gesture>_timing.png`
- `landmarks/<gesture>_<recording>.png`

Use this script when you want to see how event activity evolves after GO and whether earlier or later temporal landmarks produce more discriminative frames.

## How To Run

Run either script directly from this folder:

```bash
python3 exploring_timing/test_window_ranges.py
python3 exploring_timing/visualize_timing.py
```

The scripts process the first few recordings per gesture by default, so they are intended for quick analysis rather than full dataset training.

## Example Images

Place example figures here in the README while you are still collecting final outputs.

### Window exploration example

![Window exploration example](/images/window.png)

### Timing distribution example

![Timing distribution example](/images/timing.png)

### Landmark comparison example

![Landmark comparison example](/images/landmark.png)
