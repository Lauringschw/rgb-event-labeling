# RGB Basler one frame

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

index = 1
frame_num = 1
base = Path("/home/lau/Documents/test_1")

category = "rock"   # "rock" or "paper" or "scissor"
prefix = {"rock": "r", "paper": "p", "scissor": "s"}[category]

frame_pattern = base / f"{category}/{prefix}_{index}" / f"Basler_*_{frame_num:04d}.raw"
matches = sorted(frame_pattern.parent.glob(frame_pattern.name))
if not matches:
	raise FileNotFoundError(f"No frame files matched: {frame_pattern}")

frame = np.fromfile(matches[0], dtype=np.uint8)
frame = frame.reshape(1200, 1920)

plt.imshow(frame, cmap='gray')
plt.title(f'basler frame {frame_num:04d}')
plt.show()