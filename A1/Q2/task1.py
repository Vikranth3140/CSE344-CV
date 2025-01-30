import os
import torch
import random
import shutil
import wandb
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.io import read_image
from torchvision.datasets import ImageFolder



CLASS_MAPPING = {
    'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3, 'brown_bear': 4,
    'dog': 5, 'roe_deer': 6, 'sika_deer': 7, 'wild_boar': 8, 'people': 9
}



# Define dataset path
DATASET_PATH = r"dataset"
TRAIN_PATH = r"train_dataset"
VAL_PATH = r"val_dataset"

# Get class-wise image paths
data = []
labels = []

for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            data.append(img_path)
            labels.append(CLASS_MAPPING[class_name])  # Assign numeric label

# Perform a stratified split (80% train, 20% validation)
train_files, val_files, train_labels, val_labels = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

# Function to move files to the new structure
def move_files(file_list, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for file_path in tqdm(file_list, desc=f"Moving files to {target_dir}"):
        class_name = os.path.basename(os.path.dirname(file_path))  # Get the class name
        class_dir = os.path.join(target_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        shutil.copy(file_path, os.path.join(class_dir, os.path.basename(file_path)))

# Move train and validation images
move_files(train_files, TRAIN_PATH)
move_files(val_files, VAL_PATH)