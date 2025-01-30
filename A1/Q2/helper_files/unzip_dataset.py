import zipfile
import os

zip_path = r"..\dataset\drive-download-20250130T214341Z-001.zip"

extract_path = "..\dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"Files extracted to: {extract_path}")