import numpy as np

metadata = np.load('/home/lau/Documents/test_2/paper/p_1/recording_metadata.npy', allow_pickle=True).item()

print(metadata)
print(f"\nGO timestamp: {metadata['go_timestamp_system']}")
print(f"GO offset from start: {metadata['go_offset_from_start']:.3f} seconds")
print(f"Expected GO frame: {metadata['expected_go_frame']}")