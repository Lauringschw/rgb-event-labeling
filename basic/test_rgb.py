# RGB Basler one frame

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

index = 1
base = Path("/home/lau/Documents/test_1")

category = "rock"   # "rock" or "paper" or "scissor"
prefix = {"rock": "r", "paper": "p", "scissor": "s"}[category]

sample_dir = base / f"{category}/{prefix}_{index}"
frame_pattern = "Basler_a2A1920-160ucPRO__*.raw"
frame_paths = sorted(sample_dir.glob(frame_pattern))
if not frame_paths:
    raise FileNotFoundError(f"No frame files matched: {sample_dir / frame_pattern}")

frame = np.fromfile(frame_paths[0], dtype=np.uint8)
frame = frame.reshape(1200, 1920)

plt.imshow(frame, cmap='gray')
plt.title(f'Basler frame: {frame_paths[0].name}')
plt.show()