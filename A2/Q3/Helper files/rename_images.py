import os

# Path to your dataset
folder_path = r'../custom_dataset'

# Get all image filenames (you can add more extensions if needed)
image_extensions = ['.jpg', '.jpeg', '.png']
images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]

# Sort files to rename consistently
images.sort()

# Rename files
for idx, filename in enumerate(images, start=1):
    new_name = f"{idx}.jpg"
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    os.rename(src, dst)
    print(f"Renamed {filename} → {new_name}")