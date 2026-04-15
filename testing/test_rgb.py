import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / '.env')

def show_basler_frame(category: str, index: int, frame_num: int):
    """Display a single Basler RGB frame from a recording."""
    base = Path(os.getenv("RECORDINGS_DIR")) / Path(os.getenv("DIR"))
    prefix = {"rock": "r", "paper": "p", "scissor": "s"}[category]
    
    folder = base / category / f"{prefix}_{index}"
    frame_path = folder / f"Basler_acA1920-155um__{frame_num}.raw"
    
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")
    
    # Load raw grayscale image --> 1200x1920 uint8
    frame = np.fromfile(frame_path, dtype=np.uint8).reshape(1200, 1920)
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(frame, cmap='gray')
    plt.title(f'{category}/{prefix}_{index} | Frame {frame_num}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_basler_frame(category="paper", index=1, frame_num=1)