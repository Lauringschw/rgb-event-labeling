import os

base_path = "."  

for i in range(1, 1001):
    folder_name = f"p_{i}"
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

print("Folders p_1 to p_1000 created.")