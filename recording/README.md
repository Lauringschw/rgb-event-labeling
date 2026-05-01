# recording/

Scripts for synchronized dual-camera data collection (Basler RGB + Prophesee EVK4 HD).

## GUI

![Recording GUI](/images/recording.gif)

## Usage

```bash
python3 record_dual_camera_gui.py
```

Set gesture, distance, and recording number in the GUI, then click **Capture & Record**.

## Sequence

1. Captures a single calibration frame (`calibration_frame.raw`)
2. Initializes Prophesee — starts logging events to `prophesee_events.raw` immediately
3. Runs a 3 → 2 countdown; Basler starts grabbing at "1"
4. Shows **GO** cue — participant performs gesture
5. Records for ~1s after GO, then stops both cameras
6. Saves Basler frames, timestamps, and metadata

## Output per recording

```
<gesture>/<prefix>_<n>/
├── prophesee_events.raw          # raw event stream (Prophesee clock, µs)
├── Basler_acA1920-155um__0.raw   # frame 0, 1920×1200 uint8 raw bytes
├── Basler_acA1920-155um__1.raw
├── ...
├── calibration_frame.raw         # single pre-recording frame
├── basler_frame_timestamps.npy   # Basler hardware timestamps (overwritten by extract_sync_timestamp.py)
└── recording_metadata.npy        # GO timing, distance, frame count, expected_go_frame estimate
```

`recording_metadata.npy` keys: `go_timestamp_system`, `recording_start_time`, `go_offset_from_start`, `recording_end_time`, `distance`, `total_frames`, `expected_go_frame`.

`expected_go_frame` is a rough estimate (`offset_s × 140`) used only as a jump hint in the labeling tool — not used for any downstream processing.

## Hardware sync

Basler Line2 is configured as `ExposureActive` output. Each exposure fires a rising-edge TTL pulse into the Prophesee external trigger input. These trigger timestamps (in Prophesee µs) are extracted in the next pipeline step to give a shared time base across both cameras.

## Configuration

Paths read from `.env` at the repo root:

```dotenv
RECORDINGS_DIR=/media/lau/T7/thesis/recordings   # storage root for all recordings
DIR=trial2                                        # session subfolder; base path = RECORDINGS_DIR/DIR
```

Bias settings (`bias_diff_on`, `bias_diff_off`) are adjustable in the GUI. Camera settings (140 fps, 1920×1200) are loaded from Basler UserSet1.

## Auto-capture

Enable **Automatically do capture** to chain recordings back-to-back without clicking between takes.
