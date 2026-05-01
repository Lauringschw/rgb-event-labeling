# exploring/

Exploratory visualizations for thesis presentation. Produces three figures from a small sample of labeled recordings.

```bash
python3 explore_data.py
```

---

## Figures

**`event_density_over_time.png`**

Average event count per 10ms bin from GO to GO+500ms, one line per gesture. Shaded band = ±1 std across recordings. Dashed vertical line = mean t_initial. Shows when motion actually starts relative to the GO cue and how event density differs across gesture classes.

![Event Density](/images/event_density_over_time.png)


**`gesture_window_grid.png`**

3 rows (rock / paper / scissor) × 10 columns (10ms → 100ms windows), anchored at t_initial. Each cell is a single representative recording. Matches the RQ1 window range exactly.

![Gesture Window](/images/gesture_window_grid.png)


**`offset_window_heatmap.png`**

One heatmap per gesture. Rows = offset from t_initial (0 → 100ms), columns = window length (20 → 200ms). Cell value = average event count across recordings. Shows how much information is available at different temporal positions.

![Heatmap](/images/offset_window_heatmap.png)


---

## Configuration

```python
N_SAMPLES_DENSITY = 5  # recordings averaged for density plot
N_SAMPLES_GRID    = 1  # single recording per gesture for grid (no averaging)
N_SAMPLES_HEATMAP = 5  # recordings averaged for heatmap
```

Paths read from `.env` at the repo root:

```dotenv
RECORDINGS_DIR=/media/lau/T7/thesis/recordings   # storage root for all recordings
DIR=trial2                                        # session subfolder; base path = RECORDINGS_DIR/DIR
EXPLORATION_DIR=/media/lau/T7/thesis/exploring   # output directory for figures
```