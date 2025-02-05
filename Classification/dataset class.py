import os
import torch
import wandb
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.io import read_image
import matplotlib.pyplot as plt


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
        try:
            image = read_image(img_path).float()  # Load image as a tensor
            
            # Convert grayscale images to RGB
            if image.shape[0] == 1:  # If 1-channel (grayscale), repeat across 3 channels
                image = image.repeat(3, 1, 1)
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a default or skip this image
            raise










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


class WildlifeCNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(WildlifeCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,          # RGB input
            out_channels=32,        # 32 feature maps
            kernel_size=3,          # 3x3 kernel
            stride=1,
            padding=1
        )
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=4, stride=4)
        
        # Second Convolutional Block
        self.conv2 = torch.nn.Conv2d(
            in_channels=32,         # Input from previous layer
            out_channels=64,        # 64 feature maps
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third Convolutional Block
        self.conv3 = torch.nn.Conv2d(
            in_channels=64,         # Input from previous layer
            out_channels=128,       # 128 feature maps
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of flattened features
        # Input: 224x224 -> After pool1: 56x56 -> After pool2: 28x28 -> After pool3: 14x14
        self.flat_features = 128 * 14 * 14
        
        # Classification head
        self.classifier = torch.nn.Linear(self.flat_features, num_classes)
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Third block
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(-1, self.flat_features)
        
        # Classification
        x = self.classifier(x)
        
        return x

# Initialize the model
model = WildlifeCNN(num_classes=len(CLASS_MAPPING))

# Log model architecture to WandB
wandb.watch(model)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Device configuration
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# If multiple GPUs are available, use DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model = model.to(device)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

# Training loop
num_epochs = 10
best_val_acc = 0.0

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc = validate(model, val_dataloader, criterion, device)
    
    # Log metrics to WandB
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc
    })
    
    # Print metrics
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

print('Training finished!')
wandb.finish()