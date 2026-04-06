# extract_trigger_timestamps.py
from pathlib import Path
from metavision_hal import DeviceDiscovery
import numpy as np

def extract_trigger_timestamps(raw_path):
    """
    Extract Basler frame timestamps from DVS trigger channel.
    
    Args:
        raw_path: Path to .raw recording file
        
    Returns:
        numpy array of timestamps (µs) for each RGB frame
    """
    device = DeviceDiscovery.open_raw_file(str(raw_path))
    
    all_triggers = []
    
    def trigger_callback(triggers):
        all_triggers.extend(triggers)
    
    i_ext_trigger = device.get_i_event_ext_trigger_decoder()
    if not i_ext_trigger:
        raise RuntimeError("No external trigger decoder found in recording")
    
    i_ext_trigger.add_event_buffer_callback(trigger_callback)
    
    i_decoder = device.get_i_events_stream_decoder()
    i_events_stream = device.get_i_events_stream()
    i_events_stream.start()
    
    while True:
        ret = i_events_stream.poll_buffer()
        if ret < 0:
            break
        elif ret > 0:
            raw_data = i_events_stream.get_latest_raw_data()
            if raw_data:
                i_decoder.decode(raw_data)
    
    i_events_stream.stop()
    
    if len(all_triggers) == 0:
        raise ValueError("No triggers found in recording")
    
    # sort and filter rising edges
    all_triggers = sorted(all_triggers, key=lambda t: t['t'])
    rising = [t for t in all_triggers if t['p'] == 1]
    
    # deduplicate by timestamp
    seen = set()
    unique = []
    for t in rising:
        if t['t'] not in seen:
            unique.append(t)
            seen.add(t['t'])
    
    trigger_times = np.array([t['t'] for t in unique])
    
    print(f"Total triggers: {len(all_triggers)}")
    print(f"Rising edges: {len(rising)}")
    print(f"Unique rising edges: {len(unique)}")
    print(f"Time range: {trigger_times[0]/1e6:.3f}s to {trigger_times[-1]/1e6:.3f}s")
    print(f"Duration: {(trigger_times[-1] - trigger_times[0])/1e6:.2f}s")
    
    return trigger_times


if __name__ == "__main__":
    # example usage
    raw_path = Path("/home/lau/Documents/test_1/rock/r_1/recording_2026-04-02_14-31-40.raw")
    
    timestamps = extract_trigger_timestamps(raw_path)
    
    # save to same directory as raw file
    output_path = raw_path.parent / "basler_frame_timestamps.npy"
    np.save(output_path, timestamps)
    print(f"\nSaved to: {output_path}")