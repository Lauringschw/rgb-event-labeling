from pathlib import Path

base = Path("/media/lau/seagate/trial2")
for gesture in ['rock', 'paper', 'scissor']:
    prefix = gesture[0]
    folders = sorted(base.glob(f"{gesture}/{prefix}_*"))
    print(f"{gesture}: {len(folders)} folders ({prefix}_1 to {prefix}_{len(folders)})")
    

import numpy as np
recids = np.load("/media/lau/T7/thesis/test_samples/rq1_recording_ids.npy")
print(f"Test rec IDs range: {recids.min()} to {recids.max()}")
print(f"Unique: {len(np.unique(recids))}")