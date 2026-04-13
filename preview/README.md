# Preview Scripts

This folder contains scripts for previewing and analyzing data.

## Scripts

### 1. `preview_extracted_samples.py`
Previews extracted event samples from the research questions (RQ1, RQ2, RQ3).

**What it does:**
- Loads generated event sample files (`event_samples_rq1.npy`, `event_samples_rq2.npy`, `event_samples_rq3.npy`) from `SAMPLES_DIR`.
- Saves preview PNG images to `SAMPLES_DIR/previews/rq1`, `rq2`, `rq3` folders.
- Optionally displays interactive plots for visualization.

**Usage:**
```bash
# Preview RQ1 samples (6 recordings)
python preview_extracted_samples.py --rq rq1 --num 6

# Preview all RQs (4 recordings each), save only (no interactive display)
python preview_extracted_samples.py --rq all --num 4 --save-only
```

**Arguments:**
- `--rq`: Choose `rq1`, `rq2`, `rq3`, or `all`
- `--num`: Number of recordings to preview per RQ
- `--save-only`: Save PNGs without showing interactive plots
- `--seed`: Random seed for reproducible sample selection (default: 42)

### 2. `read_results.py`
Reads and analyzes results from the trained models for each research question.

**What it does:**
- Loads result files (`rq1_results.npy`, `rq2_results.npy`, `rq3_results.npy`) from `RESULTS_DIR`.
- Computes classification metrics (precision, recall, F1-score) from confusion matrices.
- Displays performance statistics for each gesture class.

**Note:** Currently hardcoded to RQ1 (`num = 1`). Edit the script to change to RQ2 or RQ3.

### 3. `timestamp.py`
Displays timestamps and labels for a specific recording.

**What it does:**
- Loads three data files from a recording folder and prints their contents.
- Useful for verifying synchronization and labeling accuracy.

**Usage:**
```bash
python timestamp.py
```

**Output:**
1. **Basler Timestamps**: Array of frame timestamps in microseconds from the RGB camera.
2. **Labels**: Dictionary with manually labeled gesture timing:
   - `go_frame`: Frame number when "GO" was detected
   - `go_time_us`: Timestamp of GO event
   - `t_initial_frame`: Frame number of initial gesture
   - `t_initial_time_us`: Timestamp of initial gesture
   - `recording_folder`: Path to the recording
3. **Recording Metadata**: Dictionary with recording setup info:
   - `expected_go_frame`: Auto-detected GO frame from audio
   - `go_offset_from_start`: Time offset of GO from recording start
   - Other recording parameters

## Data Flow and File Creation

### Recording Process (`recording/record_dual_camera_gui.py`)
Creates **`recording_metadata.npy`** with auto-detected GO timing:
- `expected_go_frame`: Auto-calculated frame where "GO!" occurred during recording
- Used as initial/default GO detection for labeling

### Labeling Process (`rqs_shared/label_tool.py`) 
Creates **`labels.npy`** with manually corrected timing:
- `go_frame`: Manually marked GO frame number
- `t_initial_frame`: Manually marked gesture start frame  
- Overrides auto-detected values when present
- The labeling tool loads `labels.npy` first, falls back to `recording_metadata.npy` if no manual labels exist
