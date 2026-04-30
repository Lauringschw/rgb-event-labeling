import numpy as np
from pathlib import Path
from metavision_core.event_io import EventsIterator

SENSOR_HEIGHT = 360
SENSOR_WIDTH  = 640
ORIG_HEIGHT   = 720
ORIG_WIDTH    = 1280

folder = Path("/media/lau/seagate/trial2/rock/r_1")
labels = np.load(folder / "labels.npy", allow_pickle=True).item()
t_initial = labels['t_initial_time_us']

mv_iterator = EventsIterator(str(folder / "prophesee_events.raw"))
chunks = [ev for ev in mv_iterator]
all_events = np.concatenate(chunks)

mask = (all_events['t'] >= t_initial) & (all_events['t'] < t_initial + 300_000)
events = all_events[mask]

# fixed coordinate scaling
x = (events['x'].astype(np.int32) * SENSOR_WIDTH  // ORIG_WIDTH)
y = (events['y'].astype(np.int32) * SENSOR_HEIGHT // ORIG_HEIGHT)

valid = (x >= 0) & (x < SENSOR_WIDTH) & (y >= 0) & (y < SENSOR_HEIGHT)
x, y = x[valid], y[valid]
p = events['p'][valid]

histogram = np.zeros((2, SENSOR_HEIGHT, SENSOR_WIDTH), dtype=np.float32)
on_mask  = p == 1
off_mask = ~on_mask
np.add.at(histogram[0], (y[on_mask],  x[on_mask]),  1)
np.add.at(histogram[1], (y[off_mask], x[off_mask]), 1)

print(f"Histogram shape: {histogram.shape}")
print(f"Non-zero pixels: {(histogram != 0).mean():.4f}")
print(f"x range after scaling: {x.min()} to {x.max()}")
print(f"y range after scaling: {y.min()} to {y.max()}")
print(f"Max value: {histogram.max()}")

# visualize
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.imshow(histogram[0], cmap='hot')
ax1.set_title('ON events')
ax2.imshow(histogram[1], cmap='hot')
ax2.set_title('OFF events')
plt.tight_layout()
plt.savefig('/tmp/histogram_test.png')
print("Saved to /tmp/histogram_test.png")