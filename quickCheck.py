import numpy as np

trigger_times = np.load('basler_frame_timestamps.npy')

print(f"frame 0 → prophesee {trigger_times[0]} µs ({trigger_times[0]/1e6:.3f}s)")
print(f"frame 100 → prophesee {trigger_times[100]} µs ({trigger_times[100]/1e6:.3f}s)")
print(f"frame 661 → prophesee {trigger_times[661]} µs ({trigger_times[661]/1e6:.3f}s)")