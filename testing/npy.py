
import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')

base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))

data = np.load(base / "rock" / "r_1" / "basler_frame_timestamps.npy")
print(data)