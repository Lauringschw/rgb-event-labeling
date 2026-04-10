import os

base_path = "/home/lau/Documents/test_2/rock"  

for i in range(1, 1001):
    folder_name = f"r_{i}"
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

print("Folders created.")