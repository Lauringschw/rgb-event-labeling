import numpy as np
from pathlib import Path
from metavision_core.event_io import EventsIterator

folder = Path("/media/lau/seagate/trial2/rock/r_1")
labels = np.load(folder / "labels.npy", allow_pickle=True).item()
t_initial = labels['t_initial_time_us']

mv_iterator = EventsIterator(str(folder / "prophesee_events.raw"))
chunks = [ev for ev in mv_iterator]
all_events = np.concatenate(chunks)

mask = (all_events['t'] >= t_initial) & (all_events['t'] < t_initial + 300_000)
events = all_events[mask]

print(f"Event dtype: {events.dtype}")
print(f"Event fields: {events.dtype.names}")
print(f"\nx range: {events['x'].min()} to {events['x'].max()}")
print(f"y range: {events['y'].min()} to {events['y'].max()}")
print(f"\nFirst 10 events:")
for e in events[:10]:
    print(f"  x={e['x']:4d}  y={e['y']:4d}  t={e['t']}  p={e['p']}")