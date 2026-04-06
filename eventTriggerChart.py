from metavision_hal import DeviceDiscovery
import numpy as np

path = '/home/lau/Documents/test_1/rock/r_1/recording_2026-04-02_14-31-40.raw'

device = DeviceDiscovery.open_raw_file(path)

all_triggers = []

def trigger_callback(triggers):
    all_triggers.extend(triggers)

i_ext_trigger = device.get_i_event_ext_trigger_decoder()
if i_ext_trigger:
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

print(f"total triggers: {len(all_triggers)}")

if len(all_triggers) > 0:
    # check trigger format
    print(f"trigger dtype: {all_triggers[0].dtype}")
    print(f"first trigger: {all_triggers[0]}")
    
    # sort by 't' field (not attribute)
    all_triggers = sorted(all_triggers, key=lambda t: t['t'])
    
    print(f"\ntime range: {all_triggers[0]['t'] / 1e6:.3f}s to {all_triggers[-1]['t'] / 1e6:.3f}s")
    
    # deduplicate rising edges
    rising = [t for t in all_triggers if t['p'] == 1]
    seen = set()
    unique = []
    for t in rising:
        if t['t'] not in seen:
            unique.append(t)
            seen.add(t['t'])
    
    print(f"unique rising edges: {len(unique)}")
    
    trigger_times = np.array([t['t'] for t in unique])
    np.save('basler_frame_timestamps.npy', trigger_times)