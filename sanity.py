import numpy as np
from pathlib import Path

folder = Path("/media/lau/seagate/trial2/rock/r_1")
labels = np.load(folder / "labels.npy", allow_pickle=True).item()
print(f"go_time_us:      {labels['go_time_us']}")
print(f"t_initial_us:    {labels['t_initial_time_us']}")
diff_ms = (labels['t_initial_time_us'] - labels['go_time_us']) / 1000
print(f"t_initial - GO:  {diff_ms:.1f} ms")