import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

index = 1 # set this
frame_num = 1 # set this
base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))

category = "paper"   # "rock" or "paper" or "scissor"
prefix = {"rock": "r", "paper": "p", "scissor": "s"}[category]

frame_pattern = base / f"{category}/{prefix}_{index}" / f"Basler_acA1920-155um__{frame_num}.raw"
matches = sorted(frame_pattern.parent.glob(frame_pattern.name))
if not matches:
	raise FileNotFoundError(f"No frame files matched: {frame_pattern}")

frame = np.fromfile(matches[0], dtype=np.uint8)
frame = frame.reshape(1200, 1920)

plt.imshow(frame, cmap='gray')
plt.title(f'basler frame {frame_num:04d}')
plt.show()