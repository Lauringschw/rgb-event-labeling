from pypylon import pylon
from metavision_hal import DeviceDiscovery, I_TriggerIn
import numpy as np
from pathlib import Path
import time
import threading

# Setup output folder
recording_num = 1
output_dir = Path(f"/home/lau/Documents/recordings/recording_{recording_num:03d}")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Saving to: {output_dir}")

# Configure Basler
print("Initializing Basler camera...")
camera_basler = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera_basler.Open()

# Load UserSet1
try:
    camera_basler.UserSetSelector.SetValue("UserSet1")
    camera_basler.UserSetLoad.Execute()
    print("✓ Loaded UserSet1")
except Exception as e:
    print(f"⚠ Could not load UserSet1: {e}")

# STEP 1: SINGLE SHOT
print("\n📸 STEP 1: Single calibration frame")
input("   Press ENTER to capture...")
camera_basler.AcquisitionMode.SetValue("SingleFrame")
camera_basler.StartGrabbing(pylon.GrabStrategy_OneByOne)
grab_result = camera_basler.RetrieveResult(5000)
if grab_result.GrabSucceeded():
    calib_img = grab_result.Array
    calib_path = output_dir / "calibration_frame.raw"
    calib_img.tofile(calib_path)
    print(f"✓ Saved: {calib_path}")
grab_result.Release()
camera_basler.StopGrabbing()

# Configure for continuous
camera_basler.AcquisitionMode.SetValue("Continuous")
camera_basler.AcquisitionFrameRateEnable.SetValue(True)
camera_basler.AcquisitionFrameRate.SetValue(140.0)

# Configure trigger OUTPUT
try:
    camera_basler.LineSelector.SetValue("Line2")
    camera_basler.LineMode.SetValue("Output")
    camera_basler.LineSource.SetValue("ExposureActive")
    camera_basler.LineInverter.SetValue(False)
    print("✓ Trigger output: Line2 = ExposureActive")
except Exception as e:
    print(f"⚠ Trigger setup failed: {e}")

# STEP 2: START PROPHESEE
print("\n🔴 STEP 2: Starting Prophesee...")
input("   Press ENTER to start Prophesee recording...")

device = DeviceDiscovery.open("")

if device is None:
    print("❌ ERROR: Could not open Prophesee camera!")
    print("   Make sure Metavision Viewer is CLOSED")
    exit(1)

# Configure biases
try:
    i_ll_biases = device.get_i_ll_biases()
    if i_ll_biases is not None:
        i_ll_biases.set("bias_diff_on", 32)
        i_ll_biases.set("bias_diff_off", 32)
        print("✓ Biases set: bias_diff_on=32, bias_diff_off=32")
    else:
        print("⚠ Bias interface not available")
except Exception as e:
    print(f"⚠ Bias configuration: {e}")

# Configure external trigger
try:
    i_trigger_in = device.get_i_trigger_in()
    if i_trigger_in is not None:
        i_trigger_in.enable(I_TriggerIn.Channel.MAIN)
        print("✓ Prophesee trigger enabled (MAIN)")
    else:
        print("⚠ No trigger input interface")
except Exception as e:
    print(f"⚠ Prophesee trigger setup: {e}")

prophesee_output = str(output_dir / "prophesee_events.raw")
i_events_stream = device.get_i_events_stream()
i_events_stream.log_raw_data(prophesee_output)
i_events_stream.start()
print("✓ Prophesee recording started")

# STEP 3: START BASLER
print("\n🔴 STEP 3: Starting Basler...")
input("   Press ENTER to start Basler recording...")

camera_basler.StartGrabbing(pylon.GrabStrategy_OneByOne)

basler_timestamps = []
frame_idx = 0
start_time = time.time()

print("\n✓ BOTH CAMERAS RECORDING")
print("  Basler Line2 → Prophesee trigger input")
print("  Press Ctrl+C to stop Basler\n")

stop_recording = False

def prophesee_poll():
    while not stop_recording:
        try:
            i_events_stream.get_latest_raw_data()
            time.sleep(0.001)
        except:
            break

prophesee_thread = threading.Thread(target=prophesee_poll, daemon=True)
prophesee_thread.start()

try:
    while True:
        if camera_basler.IsGrabbing():
            grab_result = camera_basler.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
            
            if grab_result.GrabSucceeded():
                img = grab_result.Array
                frame_path = output_dir / f"Basler_acA1920-155um__{frame_idx}.raw"
                img.tofile(frame_path)
                
                timestamp_us = int(grab_result.TimeStamp)
                basler_timestamps.append(timestamp_us)
                
                frame_idx += 1
                
                if frame_idx % 140 == 0:
                    elapsed = time.time() - start_time
                    print(f"Frames: {frame_idx}, Time: {elapsed:.1f}s")
                
                grab_result.Release()
        
except KeyboardInterrupt:
    print("\n\n⏹ Stopping Basler...")
    camera_basler.StopGrabbing()
    end_time = time.time()
    print(f"✓ Basler stopped ({frame_idx} frames)")
    
    input("\n   Press ENTER to stop Prophesee...")
    
    print("  Flushing event buffer...")
    time.sleep(2)
    
    stop_recording = True
    time.sleep(0.5)
    i_events_stream.stop()
    i_events_stream.stop_log_raw_data()
    print("✓ Prophesee stopped")

# Cleanup
camera_basler.Close()

timestamps_path = output_dir / "basler_frame_timestamps.npy"
np.save(timestamps_path, np.array(basler_timestamps))

elapsed = end_time - start_time
print(f"\n✓ Recording complete!")
print(f"  Duration: {elapsed:.1f}s")
print(f"  Basler: {frame_idx} frames ({frame_idx/elapsed:.1f} fps)")
print(f"  Prophesee: {prophesee_output}")
print(f"  Check triggers in Metavision Viewer")
print(f"  Saved to: {output_dir}")