from metavision_core.event_io import EventsIterator

path = '/home/lau/Documents/test_1/rock/r_1/recording_2026-04-02_14-31-40.raw'

mv_it = EventsIterator(path)

event_count = 0
first_t = None
last_t = None

for ev in mv_it:
    if first_t is None:
        first_t = ev['t'][0]
    last_t = ev['t'][-1]
    event_count += len(ev)

print(f"total events: {event_count}")
print(f"first timestamp: {first_t} µs")
print(f"last timestamp: {last_t} µs")
print(f"duration: {(last_t - first_t) / 1e6:.2f} seconds")