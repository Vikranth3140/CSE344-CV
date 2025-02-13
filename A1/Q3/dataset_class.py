import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
import pandas as pd
import matplotlib.pyplot as plt

# Read the class mapping from CSV
CLASS_MAPPING = pd.read_csv('dataset/class_dict.csv')

def analyze_dataset(data_dir):
    """Analyze the dataset structure and contents"""
    train_images_dir = os.path.join(data_dir, 'train')
    train_masks_dir = os.path.join(data_dir, 'train_labels')
    test_images_dir = os.path.join(data_dir, 'test_images')
    test_masks_dir = os.path.join(data_dir, 'test_labels')
    
    # Count files
    train_images = sorted([f for f in os.listdir(train_images_dir) if f.endswith('.jpg') or f.endswith('.png')])
    train_masks = sorted([f for f in os.listdir(train_masks_dir) if f.endswith('.jpg') or f.endswith('.png')])
    test_images = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.jpg') or f.endswith('.png')])
    test_masks = sorted([f for f in os.listdir(test_masks_dir) if f.endswith('.jpg') or f.endswith('.png')])
    
    # Load a sample image and mask to get dimensions
    sample_img = Image.open(os.path.join(train_images_dir, train_images[0]))
    sample_mask = Image.open(os.path.join(train_masks_dir, train_masks[0]))
    
    print("Dataset Analysis:")
    print(f"Number of training images: {len(train_images)}")
    print(f"Number of training masks: {len(train_masks)}")
    print(f"Number of test images: {len(test_images)}")
    print(f"Number of test masks: {len(test_masks)}")
    print(f"Original image dimensions: {sample_img.size}")
    print(f"Original mask dimensions: {sample_mask.size}")
    print(f"\nNumber of classes: {len(CLASS_MAPPING)}")
    print("\nClass distribution:")
    print(CLASS_MAPPING[['name', 'r', 'g', 'b']].head())
    
    # Visualize a sample pair
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(sample_img)
    plt.title("Sample Training Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(sample_mask)
    plt.title("Sample Training Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return len(train_images), sample_img.size

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.class_mapping = CLASS_MAPPING
        
        # Get all image and mask files based on split
        if split == 'train':
            self.images_dir = os.path.join(data_dir, 'train')
            self.masks_dir = os.path.join(data_dir, 'train_labels')
        else:  # test
            self.images_dir = os.path.join(data_dir, 'test_images')
            self.masks_dir = os.path.join(data_dir, 'test_labels')
        
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(self.masks_dir) if f.endswith('.jpg') or f.endswith('.png')])
        
        # Basic transforms for both image and mask
        self.resize = transforms.Resize((360, 480), interpolation=transforms.InterpolationMode.BILINEAR)
        self.resize_mask = transforms.Resize((360, 480), interpolation=transforms.InterpolationMode.NEAREST)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        # Open images using PIL
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')  # Convert mask to RGB since it's color-coded
        
        # Resize both image and mask
        image = self.resize(image)
        mask = self.resize_mask(mask)  # Use nearest neighbor interpolation for masks
        
        # Convert to tensor
        image = F.to_tensor(image)
        mask = torch.from_numpy(np.array(mask))
        mask = mask.permute(2, 0, 1)  # Convert to CxHxW format
        
        # Normalize only the image, not the mask
        image = self.normalize(image)
        
        # Convert mask to class indices
        mask, self.class_to_idx = self.convert_mask_to_class_indices(mask)
        return image, mask
    
    def convert_mask_to_class_indices(self, mask):
        """Convert RGB mask to class indices from 0 to 31"""
        mask = mask.float()
        class_mask = torch.zeros(mask.shape[1], mask.shape[2], dtype=torch.long)
        
        # Create class name to index mapping
        class_to_idx = {row['name']: idx for idx, row in self.class_mapping.iterrows()}
        
        # Create RGB to class index mapping
        for idx, row in self.class_mapping.iterrows():
            r, g, b = row['r'], row['g'], row['b']
            # Find pixels matching this class's RGB values
            rgb_match = (mask[0] == r) & (mask[1] == g) & (mask[2] == b)
            class_mask[rgb_match] = class_to_idx[row['name']]  # Assign class index to matching pixels
            
        return class_mask, class_to_idx

def get_dataloaders(data_dir, batch_size=4):
    # Create train and test datasets
    train_dataset = SegmentationDataset(data_dir, split='train')
    test_dataset = SegmentationDataset(data_dir, split='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader

def analyze_class_distribution(dataset):
    """Analyze the distribution of classes in the dataset"""
    class_counts = torch.zeros(len(CLASS_MAPPING))
    
    for _, mask in dataset:
        # Count pixels of each class
        for class_idx in range(len(CLASS_MAPPING)):
            class_counts[class_idx] += (mask == class_idx).sum().item()
    
    return class_counts

def plot_class_distribution(train_dataset, test_dataset):
    """Plot the class distribution for both training and test sets"""
    # Get class counts
    train_counts = analyze_class_distribution(train_dataset)
    test_counts = analyze_class_distribution(test_dataset)
    
    # Convert to percentages
    train_dist = train_counts / train_counts.sum() * 100
    test_dist = test_counts / test_counts.sum() * 100
    
    # Create bar plot
    plt.figure(figsize=(15, 8))
    
    x = np.arange(len(CLASS_MAPPING))
    width = 0.35
    
    plt.bar(x - width/2, train_dist, width, label='Train')
    plt.bar(x + width/2, test_dist, width, label='Test')
    
    plt.xlabel('Classes')
    plt.ylabel('Percentage of Pixels')
    plt.title('Class Distribution in Training and Test Sets')
    plt.xticks(x, CLASS_MAPPING['name'], rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\nClass Distribution Statistics:")
    print(f"{'Class':<20} {'Train %':>10} {'Test %':>10}")
    print("-" * 42)
    
    for idx, class_name in enumerate(CLASS_MAPPING['name']):
        print(f"{class_name:<20} {train_dist[idx]:>10.2f} {test_dist[idx]:>10.2f}")

def visualize_class_samples(dataset, num_samples=2):
    """
    Visualize sample images and their masks for each class
    Args:
        dataset: The dataset to sample from
        num_samples: Number of samples to show for each class
    """
    # Dictionary to store samples for each class
    class_samples = {idx: [] for idx in range(len(CLASS_MAPPING))}
    
    # Collect samples for each class
    for img, mask in dataset:
        # Check which classes are present in this mask
        for class_idx in range(len(CLASS_MAPPING)):
            if (mask == class_idx).any() and len(class_samples[class_idx]) < num_samples:
                # Denormalize image
                img_np = img.permute(1, 2, 0).numpy()
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)
                
                # Create binary mask for this class
                class_mask = (mask == class_idx).numpy()
                
                class_samples[class_idx].append((img_np, class_mask))
                
        # Check if we have enough samples for all classes
        if all(len(samples) >= num_samples for samples in class_samples.values()):
            break
    
    # Plot samples for each class
    for class_idx, class_name in enumerate(CLASS_MAPPING['name']):
        samples = class_samples[class_idx]
        if samples:  # If we found any samples for this class
            plt.figure(figsize=(15, 4))
            plt.suptitle(f'Class: {class_name}', fontsize=14)
            
            for i, (img, mask) in enumerate(samples):
                # Plot original image
                plt.subplot(2, num_samples, i + 1)
                plt.imshow(img)
                plt.title(f'Image {i+1}')
                plt.axis('off')
                
                # Plot mask overlay
                plt.subplot(2, num_samples, i + 1 + num_samples)
                # Create overlay with mask in red
                overlay = img.copy()
                overlay[mask] = [1, 0, 0]  # Red color for mask
                plt.imshow(overlay)
                plt.title(f'Mask Overlay {i+1}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"No samples found for class: {class_name}")

if __name__ == "__main__":
    # Analyze dataset
    print("\nAnalyzing dataset structure...")
    num_images, image_size = analyze_dataset("dataset")
    
    # Create datasets for distribution analysis
    train_dataset = SegmentationDataset("dataset", split='train')
    test_dataset = SegmentationDataset("dataset", split='test')
    
    # Plot class distribution
    print("\nAnalyzing class distribution...")
    plot_class_distribution(train_dataset, test_dataset)
    
    # Visualize samples for each class
    print("\nVisualizing samples for each class...")
    visualize_class_samples(train_dataset, num_samples=2)
    
    # Test the dataloader
    print("\nTesting dataloader...")
    train_loader, test_loader = get_dataloaders("dataset")
    
    # Get a batch of training data
    images, masks = next(iter(train_loader))
    
    print(f"\nDataloader output:")
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")
    print(f"Mask value range: ({masks.min():.3f}, {masks.max():.3f})")
    print("fsddddddddddd")
    print(masks)
    print(f"Image value range: ({images.min():.3f}, {images.max():.3f})")
    print(f"Mask value range: ({masks.min():.3f}, {masks.max():.3f})")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Visualize a batch
    plt.figure(figsize=(15, 5))
    
    # Show image
    plt.subplot(1, 2, 1)
    img = images[0].permute(1, 2, 0).numpy()
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title("Sample Training Image from Batch")
    plt.axis('off')
    
    class_map = {row['name']: idx for idx, row in CLASS_MAPPING.iterrows()}
    print("class_map", class_map)
    color_map = CLASS_MAPPING

    # Show mask using the class_map and color_map
    mask = masks[0].numpy()
    
    # Create RGB mask using the color map
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    for class_name, idx in class_map.items():
        color = color_map[color_map['name'] == class_name].iloc[0]
        rgb_values = [color['r'], color['g'], color['b']]
        rgb_mask[mask == idx] = rgb_values
    
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_mask.astype(np.uint8))
    plt.title("Sample Training Mask from Batch")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
