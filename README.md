# rgb-event-labeling

Dataset creation pipeline for synchronized RGB + event camera gesture recordings. Produces labeled training data for low-latency hand gesture recognition.

**Hardware:** Basler a2A1920-160ucPRO (~140 fps) + Prophesee EVK4 HD, hardware-synchronized via TTL trigger wire (Basler Line2 → Prophesee external trigger input).

---

## Pipeline

```
1. record_dual_camera_gui.py       capture synchronized RGB + event data
2. extract_sync_timestamp.py       extract hardware-sync timestamps from event stream
3. label_tool.py                   manually mark GO and t_initial per recording
```

### Step 1 — Record (`recording/`)

`record_dual_camera_gui.py` runs a GUI that captures both cameras in sync. Each recording produces a folder under `<gesture>/<prefix>_<n>/` containing the Prophesee event stream (`.raw`), all Basler frames (`.raw`, 1920×1200 uint8), and recording metadata.

The Basler fires a hardware trigger pulse on every exposure; the Prophesee logs these as external trigger events in its own µs clock, providing a precise shared time base.

### Step 2 — Extract sync timestamps (`labeling/`)

`extract_sync_timestamp.py` reads each `prophesee_events.raw` file, extracts rising-edge trigger timestamps, and saves them as `basler_frame_timestamps.npy` (Prophesee µs). One timestamp per Basler frame — this is the alignment used by all downstream scripts.

### Step 3 — Label (`labeling/`)

`label_tool.py` opens each recording as a frame-by-frame viewer. Two landmarks are marked per recording:

- **GO** — frame where the GO cue appeared
- **t_initial** — first visible motion onset

Labels are saved as `labels.npy` and consumed by extraction scripts to anchor event windows relative to gesture onset.

---

## Repository structure

```
rgb-event-labeling/
├── .env
├── recording/
│   └── record_dual_camera_gui.py
└── labeling/
    ├── extract_sync_timestamp.py
    └── label_tool.py
```

## Configuration

All scripts load paths from `.env` at the repo root. Copy and edit:

```dotenv
DIR=trial2                                       # session subfolder; base path = RECORDINGS_DIR/DIR
RECORDINGS_DIR=/media/lau/T7/thesis/recordings   # where recording folders live
EXPLORATION_DIR=/media/lau/T7/thesis/exploring   # where testing occurs
```

`THESIS_DIR`, `BACKUP_DIR`, and `EXPLORATION_DIR` are not referenced by any script in this repo.
