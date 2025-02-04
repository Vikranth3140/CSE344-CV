import os
import torch
import random
import shutil
import wandb
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np


CLASS_MAPPING = {
    'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3, 'brown_bear': 4,
    'dog': 5, 'roe_deer': 6, 'sika_deer': 7, 'wild_boar': 8, 'people': 9
}



# Define dataset path
DATASET_PATH = r"dataset"
TRAIN_PATH = r"train_dataset"
VAL_PATH = r"val_dataset"

# # Get class-wise image paths
# data = []
# labels = []

# for class_name in os.listdir(DATASET_PATH):
#     class_path = os.path.join(DATASET_PATH, class_name)
#     if os.path.isdir(class_path):
#         for img_name in os.listdir(class_path):
#             img_path = os.path.join(class_path, img_name)
#             data.append(img_path)
#             labels.append(CLASS_MAPPING[class_name])  # Assign numeric label

# # Perform a stratified split (80% train, 20% validation)
# train_files, val_files, train_labels, val_labels = train_test_split(
#     data, labels, test_size=0.2, stratify=labels, random_state=42
# )

# # Function to move files to the new structure
# def move_files(file_list, target_dir):
#     os.makedirs(target_dir, exist_ok=True)
#     for file_path in tqdm(file_list, desc=f"Moving files to {target_dir}"):
#         class_name = os.path.basename(os.path.dirname(file_path))  # Get the class name
#         class_dir = os.path.join(target_dir, class_name)
#         os.makedirs(class_dir, exist_ok=True)
#         shutil.copy(file_path, os.path.join(class_dir, os.path.basename(file_path)))

# # Move train and validation images
# move_files(train_files, TRAIN_PATH)
# move_files(val_files, VAL_PATH)





# Custom Dataset Class

class WildlifeDataset(Dataset):
    def __init__(self, img_dir, class_mapping, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.class_mapping = class_mapping
        self.img_labels = []

        # Collect image paths and labels
        for class_name in os.listdir(img_dir):
            class_path = os.path.join(img_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.img_labels.append((img_path, class_mapping[class_name]))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = read_image(img_path).float()  # Load image as a tensor

        if self.transform:
            image = self.transform(image)

        return image, label

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.Normalize([0.5], [0.5])  # Normalize
])

# Initialize datasets
train_dataset = WildlifeDataset(TRAIN_PATH, CLASS_MAPPING, transform=transform)
val_dataset = WildlifeDataset(VAL_PATH, CLASS_MAPPING, transform=transform)


# 2.1.b.

# Create DataLoaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize WandB
wandb.init(
    project="russian-wildlife-classification",
    entity="vikranth2764-na"
)

# Log dataset info
wandb.config.update({
    "train_size": len(train_dataset),
    "val_size": len(val_dataset),
    "batch_size": batch_size
})

# Get a batch of training data
train_features, train_labels = next(iter(train_dataloader))

# Log batch shape info
wandb.log({
    "train_batch_shape": train_features.shape[0],
    "train_batch_channels": train_features.shape[1],
    "train_batch_height": train_features.shape[2],
    "train_batch_width": train_features.shape[3]
})

# Log sample images
img_grid = torchvision.utils.make_grid(train_features[:16])  # First 16 images
wandb.log({"Sample Train Images": [wandb.Image(img_grid, caption="Train Batch")]})

# Log label distribution
wandb.log({"Label Distribution": wandb.Histogram(train_labels.tolist())})

# Print batch info
print(f"Feature batch shape: {train_features.shape}")
print(f"Labels batch shape: {train_labels.shape}")

print("WandB Logging Done! Check your dashboard.")


# 2.1.c.

# Count class occurrences in the training set
train_label_counts = {class_name: 0 for class_name in CLASS_MAPPING.keys()}
for _, label in train_dataset.img_labels:
    class_name = list(CLASS_MAPPING.keys())[list(CLASS_MAPPING.values()).index(label)]
    train_label_counts[class_name] += 1

# Count class occurrences in the validation set
val_label_counts = {class_name: 0 for class_name in CLASS_MAPPING.keys()}
for _, label in val_dataset.img_labels:
    class_name = list(CLASS_MAPPING.keys())[list(CLASS_MAPPING.values()).index(label)]
    val_label_counts[class_name] += 1

# Define function to plot distributions
def plot_distribution(label_counts, title, color):
    classes = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts, color=color)
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.show()

# Plot Training Set Distribution
plot_distribution(train_label_counts, "Training Set Class Distribution", "blue")

# Plot Validation Set Distribution
plot_distribution(val_label_counts, "Validation Set Class Distribution", "green")

# Log class distributions to WandB
wandb.log({
    "Training Class Distribution": wandb.Table(
        data=[(k, v) for k, v in train_label_counts.items()], 
        columns=["Class", "Count"]
    ),
    "Validation Class Distribution": wandb.Table(
        data=[(k, v) for k, v in val_label_counts.items()], 
        columns=["Class", "Count"]
    )
})













# 2.2.a.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ✅ Define CNN Model
class WildlifeCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(WildlifeCNN, self).__init__()
        
        # Conv Layer 1: 3 → 32 channels, 3x3 kernel, padding=1, stride=1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)  # 4x4 max pooling

        # Conv Layer 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 max pooling

        # Conv Layer 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 max pooling

        # Fully Connected Layer (Flatten + Classification Head)
        self.fc1 = nn.Linear(128 * 14 * 14, 256)  # Adjust size based on pooling
        self.fc2 = nn.Linear(256, num_classes)  # Output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = torch.flatten(x, start_dim=1)  # Flatten before FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



# ✅ Define Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WildlifeCNN(num_classes=10).to(device)

# ✅ Define Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer


# ✅ Training Function
def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")
        wandb.log({"Train Loss": avg_loss, "Train Accuracy": train_acc})

    print("✅ Training Complete!")

# ✅ Train the Model
train(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=10)


torch.save(model.state_dict(), "wildlife_cnn.pth")
print("✅ Model Saved!")
