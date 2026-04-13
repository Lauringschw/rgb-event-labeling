import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')
base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))

metadata = base / "rock" / "r_1"

basler = np.load(metadata / "basler_frame_timestamps.npy")
labels = np.load(metadata / "labels.npy", allow_pickle=True)
recording = np.load(metadata / "recording_metadata.npy", allow_pickle=True)

print("Basler Timestamps:")
print(basler)
print("\nManual Labels:")
print(labels)
print("\nAutomatic Recording Metadata:")
print(recording)
