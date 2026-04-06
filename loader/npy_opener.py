import numpy as np

trigger_times = np.load('basler_frame_timestamps.npy')

intervals = np.diff(trigger_times)

print(f"frame intervals (µs):")
print(f"  min: {intervals.min()}")
print(f"  max: {intervals.max()}")
print(f"  mean: {intervals.mean():.1f}")
print(f"  median: {np.median(intervals):.1f}")

actual_fps = 1e6 / np.median(intervals)
print(f"\nactual frame rate: {actual_fps:.1f} fps")

# histogram
import matplotlib.pyplot as plt
plt.hist(intervals, bins=50)
plt.xlabel('interval (µs)')
plt.ylabel('count')
plt.title('frame interval distribution')
plt.show()