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

# Dataset file paths
# DATASET_PATH = r"dataset"
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
    def __init__(self, img_dir, class_mapping, transform=None, train=False):
        self.img_dir = img_dir
        self.transform = transform
        self.class_mapping = class_mapping
        self.train = train  # Flag to determine if this is training set
        self.img_labels = []

        # Normal transform for training convnet and resnet without data augmentation
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        # # Data augmentation for training
        # self.train_transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip image horizontally
        #     transforms.RandomRotation(degrees=15),    # Randomly rotate image up to 15 degrees
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust color properties
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                        std=[0.229, 0.224, 0.225])
        # ])

        # Transform for validation (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

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
            image = read_image(img_path).float()
            
            # Convert grayscale to RGB
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            
            # Apply appropriate transform
            if self.train and self.train_transform:
                image = self.train_transform(image)
            elif self.val_transform:
                image = self.val_transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            raise

def visualize_augmentations():
    plt.figure(figsize=(15, 10))
    
    train_dataset = WildlifeDataset(TRAIN_PATH, CLASS_MAPPING, transform=None, train=True)
    image, label = train_dataset[0]
    
    # Convert tensor to PIL Image for visualization
    original_image = transforms.ToPILImage()(image)
    
    # Show original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Show 4 different augmentations
    for i in range(4):
        augmented_image, _ = train_dataset[0]  # Get the same image with different augmentations
        augmented_image = transforms.ToPILImage()(augmented_image)
        
        plt.subplot(2, 3, i + 2)
        plt.imshow(augmented_image)
        plt.title(f'Augmentation {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Log to wandb
    wandb.init(project="russian-wildlife-classification", name="data-augmentation-viz")
    wandb.log({"augmentation_examples": wandb.Image(plt)})
    wandb.finish()
    
    plt.show()
    plt.close()




# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize images
#     transforms.Normalize([0.5], [0.5])  # Normalize
# ])

# # Initialize datasets
# train_dataset = WildlifeDataset(TRAIN_PATH, CLASS_MAPPING, transform=transform)
# val_dataset = WildlifeDataset(VAL_PATH, CLASS_MAPPING, transform=transform)


# # 2.1.b.

# # Create DataLoaders
# batch_size = 64
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# wandb.init(
#     project="russian-wildlife-classification",
#     entity="vikranth2764-na"
# )

# wandb.config.update({
#     "train_size": len(train_dataset),
#     "val_size": len(val_dataset),
#     "batch_size": batch_size
# })

# train_features, train_labels = next(iter(train_dataloader))

# img_grid = torchvision.utils.make_grid(train_features[:16])  # First 16 images
# wandb.log({"Sample Train Images": [wandb.Image(img_grid, caption="Train Batch")]})

# wandb.log({"Label Distribution": wandb.Histogram(train_labels.tolist())})

# print(f"Feature batch shape: {train_features.shape}")
# print(f"Labels batch shape: {train_labels.shape}")

# print("WandB Logging Done! Check your dashboard.")


# # 2.1.c.

# # Count class occurrences in the training set
# train_label_counts = {class_name: 0 for class_name in CLASS_MAPPING.keys()}
# for _, label in train_dataset.img_labels:
#     class_name = list(CLASS_MAPPING.keys())[list(CLASS_MAPPING.values()).index(label)]
#     train_label_counts[class_name] += 1

# # Count class occurrences in the validation set
# val_label_counts = {class_name: 0 for class_name in CLASS_MAPPING.keys()}
# for _, label in val_dataset.img_labels:
#     class_name = list(CLASS_MAPPING.keys())[list(CLASS_MAPPING.values()).index(label)]
#     val_label_counts[class_name] += 1

# def plot_distribution(label_counts, title, color):
#     classes = list(label_counts.keys())
#     counts = list(label_counts.values())

#     plt.figure(figsize=(10, 5))
#     plt.bar(classes, counts, color=color)
#     plt.xlabel("Class Labels")
#     plt.ylabel("Number of Images")
#     plt.xticks(rotation=45, ha="right")
#     plt.title(title)
#     plt.show()

if __name__ == "__main__":
    # visualize_augmentations()
    
    # # Initialize datasets with augmentation for training
    # train_dataset = WildlifeDataset(TRAIN_PATH, CLASS_MAPPING, train=True)
    # val_dataset = WildlifeDataset(VAL_PATH, CLASS_MAPPING, train=False)
    
    # # 2.1.b.

    # # Create DataLoaders
    # batch_size = 64
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # # Initialize WandB
    # wandb.init(
    #     project="russian-wildlife-classification",
    #     entity="vikranth2764-na"
    # )

    # # Log dataset info
    # wandb.config.update({
    #     "train_size": len(train_dataset),
    #     "val_size": len(val_dataset),
    #     "batch_size": batch_size
    # })

    # # Get a batch of training data
    # train_features, train_labels = next(iter(train_dataloader))

    # # Log sample images
    # img_grid = torchvision.utils.make_grid(train_features[:16])  # First 16 images
    # wandb.log({"Sample Train Images": [wandb.Image(img_grid, caption="Train Batch")]})

    # # Log label distribution
    # wandb.log({"Label Distribution": wandb.Histogram(train_labels.tolist())})

    # # Print batch info
    # print(f"Feature batch shape: {train_features.shape}")
    # print(f"Labels batch shape: {train_labels.shape}")

    # print("WandB Logging Done! Check your dashboard.")


    # # 2.1.c.

    # # Count class occurrences in the training set
    # train_label_counts = {class_name: 0 for class_name in CLASS_MAPPING.keys()}
    # for _, label in train_dataset.img_labels:
    #     class_name = list(CLASS_MAPPING.keys())[list(CLASS_MAPPING.values()).index(label)]
    #     train_label_counts[class_name] += 1

    # # Count class occurrences in the validation set
    # val_label_counts = {class_name: 0 for class_name in CLASS_MAPPING.keys()}
    # for _, label in val_dataset.img_labels:
    #     class_name = list(CLASS_MAPPING.keys())[list(CLASS_MAPPING.values()).index(label)]
    #     val_label_counts[class_name] += 1

    # plot_distribution(train_label_counts, "Training Set Class Distribution", "blue")
    # plot_distribution(val_label_counts, "Validation Set Class Distribution", "green")

    # # Log class distributions to WandB
    # wandb.log({
    #     "Training Class Distribution": wandb.Table(
    #         data=[(k, v) for k, v in train_label_counts.items()], 
    #         columns=["Class", "Count"]
    #     ),
    #     "Validation Class Distribution": wandb.Table(
    #         data=[(k, v) for k, v in val_label_counts.items()], 
    #         columns=["Class", "Count"]
    #     )
    # })

    # wandb.finish()