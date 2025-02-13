import os
import zipfile

dataset_dir = os.path.join("..\dataset")
os.makedirs(dataset_dir, exist_ok=True)

zip_path = r"..\drive-download-20250213T054302Z-001.zip"


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_dir)

print("Files extracted successfully to dataset directory")