# RGB Basler one frame

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

index = 1
frame_num = 1
base = Path("/home/lau/Documents/test_1")

category = "rock"   # "rock" or "paper" or "scissor"
prefix = {"rock": "r", "paper": "p", "scissor": "s"}[category]

frame = np.fromfile(f'/home/lau/Documents/test_1/{category}/{prefix}_{index}/Basler_a2A1920-160ucPRO__40648144__20260402_143142134_{frame_num:04d}.raw', dtype=np.uint8)
frame = frame.reshape(1200, 1920)

plt.imshow(frame, cmap='gray')
plt.title(f'basler frame {frame_num:04d}')
plt.show()