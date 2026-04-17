# Preview Utilities

This folder contains small scripts to inspect extracted samples, timestamps, and saved experiment results.

## Files

### [preview_extracted_samples.py](preview_extracted_samples.py)

Generates visual previews of extracted samples for RQ1, RQ2, and RQ3 and saves PNG outputs.

What it does:

- Selects random sample files from gesture recordings.
- Visualizes each sample component as a heatmap.
- Adds simple per-image statistics (active pixel ratio, value range, mean).
- Saves figures into a per-RQ preview output folder.

Example usage:

```bash
python3 preview/preview_extracted_samples.py --rq all --num 6
python3 preview/preview_extracted_samples.py --rq rq2 --per-gesture 3 --save-only
```

### [timestamp.py](timestamp.py)

Prints available timestamp and metadata arrays from a recording metadata folder.

What it does:

- Loads and prints Basler frame timestamps.
- Loads and prints manual labels.
- Loads and prints automatic recording metadata.
- Handles missing files with warnings.

Example usage:

```bash
python3 preview/timestamp.py
python3 preview/timestamp.py --metadata /path/to/recording/other/o_1
```

### [read_results.py](read_results.py)

Reads stored RQ result files and prints metrics plus a confusion-matrix-based classification report.

What it does:

- Loads `rq{num}_results.npy` for the selected research question.
- Prints test and validation accuracy per window.
- Prints confusion matrices.
- Computes and prints precision, recall, and F1 (per class, macro, weighted).

To switch between RQ files, change `num` in the script.

Example usage:

```bash
python3 preview/read_results.py
```

## Preview Images

RQ1 preview placeholder:

![RQ1 preview](../images/preview_rq1_placeholder.png)

RQ2 preview placeholder:

![RQ2 preview](../images/preview_rq2_placeholder.png)

RQ3 preview placeholder:

![RQ3 preview](../images/preview_rq3_placeholder.png)
