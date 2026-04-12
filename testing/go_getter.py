import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')
base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))

metadata = np.load(base / "paper" / "p_1" / "recording_metadata.npy", allow_pickle=True).item()

print(metadata)
print(f"\nGO timestamp: {metadata['go_timestamp_system']}")
print(f"GO offset from start: {metadata['go_offset_from_start']:.3f} seconds")
print(f"Expected GO frame: {metadata['expected_go_frame']}")