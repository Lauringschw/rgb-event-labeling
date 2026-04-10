from pypylon import pylon
from metavision_core.event_io import EventsIterator
import numpy as np
from pathlib import Path

# Setup output folder
output_dir = Path("recording_001")
output_dir.mkdir(exist_ok=True)

# Configure Basler
camera_basler = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera_basler.Open()

# Enable GPIO trigger output
camera_basler.LineSelector.SetValue("Line2")
camera_basler.LineMode.SetValue("Output")
camera_basler.LineSource.SetValue("ExposureActive")

camera_basler.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Configure Prophesee (live recording)
from metavision_sdk_core import EventsIterator
mv_iterator = EventsIterator(input_path="", delta_t=50000)  # live camera

# Record loop
print("Recording... press Ctrl+C to stop")
frame_idx = 0

try:
    while camera_basler.IsGrabbing():
        # Grab Basler frame
        grab_result = camera_basler.RetrieveResult(5000)
        if grab_result.GrabSucceeded():
            img = grab_result.Array
            np.save(output_dir / f"basler_{frame_idx}.npy", img)
            frame_idx += 1
        
        # Prophesee records automatically in background
        
except KeyboardInterrupt:
    print("Stopping...")

camera_basler.StopGrabbing()
print(f"Saved {frame_idx} frames to {output_dir}")