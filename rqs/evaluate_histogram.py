import numpy as np
from pathlib import Path

TEST_DIR = Path("/Volumes/T7/thesis/test_samples")
labels = np.load(TEST_DIR / "rq1_labels.npy")
print(f"rock:    {np.sum(labels == 0)}")
print(f"paper:   {np.sum(labels == 1)}")
print(f"scissor: {np.sum(labels == 2)}")
print(f"total:   {len(labels)}")