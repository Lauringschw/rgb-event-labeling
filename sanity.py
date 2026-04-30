import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / '.env')

SLIDING_DIR = Path(os.getenv("SLIDING_DIR_T7"))

data   = np.load(SLIDING_DIR / "histogram_data.npy", mmap_mode='r')
labels = np.load(SLIDING_DIR / "histogram_labels.npy")

print(f"Shape: {data.shape}")          # (N, 2, H, W) — check H and W
print(f"dtype: {data.dtype}")
print(f"Non-zero fraction: {np.count_nonzero(data[0]) / data[0].size:.4f}")
print(f"Sample 0 max: {data[0].max():.4f}")
print(f"x-axis spread (dim=2 max per col): {(data[0,0].max(axis=0) > 0).sum()} / {data.shape[3]} cols active")
print(f"y-axis spread (dim=1 max per row): {(data[0,0].max(axis=1) > 0).sum()} / {data.shape[2]} rows active")