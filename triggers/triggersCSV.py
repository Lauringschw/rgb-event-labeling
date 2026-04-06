# triggersCSV.py
import pandas as pd
import numpy as np

# load with comma separator, no header
triggers = pd.read_csv('triggers_triggers.csv', sep=',', header=None, names=['p', 'id', 't'])

print(f"total triggers: {len(triggers)}")
print(f"\nfirst 10 rows:")
print(triggers.head(10))

print(f"\ndata types:")
print(triggers.dtypes)

# filter rising edges (p=1)
rising = triggers[triggers['p'] == 1].copy()
print(f"\nrising edges: {len(rising)}")

# deduplicate by timestamp
rising_unique = rising.drop_duplicates(subset='t')
print(f"unique rising edges: {len(rising_unique)}")

# save timestamps
trigger_times = rising_unique['t'].values
np.save('basler_frame_timestamps.npy', trigger_times)

print(f"\nfirst trigger: {trigger_times[0] / 1e6:.3f}s")
print(f"last trigger: {trigger_times[-1] / 1e6:.3f}s")
print(f"duration: {(trigger_times[-1] - trigger_times[0]) / 1e6:.2f}s")
print(f"total synced frames: {len(trigger_times)}")