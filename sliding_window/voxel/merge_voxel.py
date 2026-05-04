from pathlib import Path
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

SLIDING_DIR    = Path(os.getenv("SLIDING_DIR"))
SLIDING_DIR_T7 = Path(os.getenv("SLIDING_DIR_T7")) / "voxel"
SLIDING_DIR_T7.mkdir(parents=True, exist_ok=True)


def get_batch_num(filepath: Path) -> int:
    return int(filepath.stem.split('_')[-1])


def merge():
    print("=" * 60)
    print("MERGE VOXEL: batches -> final dataset")
    print("=" * 60)
    print(f"Batches : {SLIDING_DIR}")
    print(f"Output  : {SLIDING_DIR_T7}\n")

    data_files  = sorted(SLIDING_DIR.glob("voxel_data_batch_*.npy"),  key=get_batch_num)
    label_files = sorted(SLIDING_DIR.glob("voxel_labels_batch_*.npy"), key=get_batch_num)
    recid_files = sorted(SLIDING_DIR.glob("voxel_recids_batch_*.npy"), key=get_batch_num)

    if not data_files:
        print("No batch files found.")
        return

    n_batches = len(data_files)
    print(f"Found {n_batches} batches\n")

    print("Counting samples ...")
    total_samples = sum(len(np.load(f)) for f in label_files)
    sample_shape  = np.load(data_files[0]).shape[1:]  # (5, 360, 640)

    bytes_needed = total_samples * int(np.prod(sample_shape)) * 4
    gb_needed    = bytes_needed / 1e9
    stat         = os.statvfs(SLIDING_DIR_T7)
    available_gb = (stat.f_bavail * stat.f_frsize) / 1e9

    print(f"Samples      : {total_samples:,}")
    print(f"Sample shape : {sample_shape}")
    print(f"Space needed : {gb_needed:.1f} GB")
    print(f"T7 available : {available_gb:.1f} GB")

    one_batch_gb = int(np.prod(sample_shape)) * 4 * 500 / 1e9
    peak_gb      = gb_needed + one_batch_gb
    print(f"Peak usage   : ~{peak_gb:.1f} GB\n")

    if peak_gb > available_gb * 0.95:
        print(f"❌ NOT ENOUGH SPACE")
        return
    print("✓ Enough space\n")

    out_data   = SLIDING_DIR_T7 / "voxel_data.npy"
    out_labels = SLIDING_DIR_T7 / "voxel_labels.npy"
    out_recids = SLIDING_DIR_T7 / "voxel_recording_ids.npy"

    print("Creating memory-mapped output files ...")
    mm_data   = np.lib.format.open_memmap(
        str(out_data),   mode='w+', dtype=np.float32,
        shape=(total_samples,) + sample_shape)
    mm_labels = np.lib.format.open_memmap(
        str(out_labels), mode='w+', dtype=np.int64,
        shape=(total_samples,))
    mm_recids = np.lib.format.open_memmap(
        str(out_recids), mode='w+', dtype=np.int64,
        shape=(total_samples,))
    print("✓ Files created\n")

    print("Merging (each batch deleted after writing) ...\n")
    current_idx = 0

    for i, (df, lf, rf) in enumerate(zip(data_files, label_files, recid_files)):
        bn = get_batch_num(df)
        print(f"  [{i+1}/{n_batches}] batch_{bn} ... ", end='', flush=True)

        bd = np.load(df)
        bl = np.load(lf)
        br = np.load(rf)
        bs = len(bd)

        mm_data  [current_idx : current_idx + bs] = bd
        mm_labels[current_idx : current_idx + bs] = bl
        mm_recids[current_idx : current_idx + bs] = br
        current_idx += bs

        del bd, bl, br
        df.unlink()
        lf.unlink()
        rf.unlink()

        print(f"✓  ({current_idx:,}/{total_samples:,})")

    del mm_data, mm_labels, mm_recids
    print("\nFlushed to disk.")

    print("\nVerifying ...")
    final_labels = np.load(out_labels)
    final_recids = np.load(out_recids)
    print(f"  rock    : {np.sum(final_labels == 0):,}")
    print(f"  paper   : {np.sum(final_labels == 1):,}")
    print(f"  scissor : {np.sum(final_labels == 2):,}")
    print(f"  TOTAL   : {len(final_labels):,}")
    print(f"  Unique recordings: {len(np.unique(final_recids))}")
    print(f"  Data file size: {out_data.stat().st_size / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"Output: {SLIDING_DIR_T7}")
    print("Next step: run train_voxel.py")


if __name__ == "__main__":
    merge()