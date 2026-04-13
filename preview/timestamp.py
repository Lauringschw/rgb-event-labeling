import argparse
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


def safe_load_npy(path: Path, allow_pickle: bool = False):
    if not path.exists():
        print(f"Warning: file not found: {path}")
        return None
    try:
        return np.load(path, allow_pickle=allow_pickle)
    except Exception as exc:
        print(f"Error loading {path}: {exc}")
        return None


def get_metadata_folder(env_path: Path, metadata_override: Path | None):
    if metadata_override is not None:
        return metadata_override

    if not env_path.exists():
        print(f"Warning: .env file not found at {env_path}")
    else:
        load_dotenv(env_path)

    recordings_dir = os.getenv("RECORDINGS_DIR")
    directory = os.getenv("DIR")
    if not recordings_dir or not directory:
        print("Error: RECORDINGS_DIR and DIR must be set in .env or pass --metadata.")
        return None

    return Path(recordings_dir) / Path(directory) / "other" / "o_1"


def main():
    parser = argparse.ArgumentParser(description="Print recording timestamps and metadata.")
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Path to the metadata folder containing .npy files.",
    )
    args = parser.parse_args()

    env_path = Path(__file__).resolve().parent.parent / ".env"
    metadata = get_metadata_folder(env_path, args.metadata)
    if metadata is None:
        sys.exit(1)

    if not metadata.exists():
        print(f"Error: metadata folder not found: {metadata}")
        sys.exit(1)

    files = [
        ("Basler Timestamps", metadata / "basler_frame_timestamps.npy", False),
        ("Manual Labels", metadata / "labels.npy", True),
        ("Automatic Recording Metadata", metadata / "recording_metadata.npy", True),
    ]

    for title, path, allow_pickle in files:
        print(f"\n{title}:")
        data = safe_load_npy(path, allow_pickle=allow_pickle)
        if data is None:
            continue
        print(data)


if __name__ == "__main__":
    main()
