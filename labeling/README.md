# labeling/

Scripts for extracting hardware-synchronized frame timestamps and manually labeling gesture onset in each recording.

Run `extract_sync_timestamp.py` first, then `label_tool.py`.

---

## 1. extract_sync_timestamp.py

Reads the Prophesee `.raw` file for each recording and extracts rising-edge external trigger timestamps. Each trigger corresponds to one Basler frame exposure, so the result is a Prophesee-clock timestamp array aligned to the RGB frame sequence.

```bash
python3 extract_sync_timestamp.py
```

Overwrites `basler_frame_timestamps.npy` in each recording folder with timestamps in Prophesee µs. This is the shared time base used by all downstream extraction scripts.

**Output:** `basler_frame_timestamps.npy` — shape `(n_frames,)`, dtype `int64`, units µs (Prophesee clock).

Prints trigger count, duration, and FPS per recording as a sanity check.

---

## 2. label_tool.py

![Recording GUI](/images/label.png)

Matplotlib GUI for marking two temporal landmarks per recording:

| Label         | Key | Meaning                           |
| ------------- | --- | --------------------------------- |
| **GO**        | `↓` | Frame where the GO cue is visible |
| **t_initial** | `↑` | First visible motion onset        |

```bash
python3 label_tool.py
```

Opens at the recording defined in `__main__`. Use **Save & Next** (or `Shift`) to move through the full dataset in order (rock → paper → scissor, numeric within each class).

**Controls**

| Key / Button | Action                                    |
| ------------ | ----------------------------------------- |
| `←` / `→`    | Previous / next frame                     |
| `↓`          | Mark current frame as GO                  |
| `↑`          | Mark current frame as t_initial           |
| `Shift`      | Save labels and advance to next recording |
| **Next →**   | Advance without saving                    |

GO frame is auto-seeded from `recording_metadata.npy` on first open. On subsequent opens, saved labels are restored from `labels.npy`.

**Output per recording:** `labels.npy` — dict with keys `go_frame`, `go_time_us`, `t_initial_frame`, `t_initial_time_us`, `recording_folder`.

---

## Configuration

Paths read from `.env` at the repo root:

```dotenv
RECORDINGS_DIR=/media/lau/T7/thesis/recordings   # storage root for all recordings
DIR=trial2                                        # session subfolder; base path = RECORDINGS_DIR/DIR
```
