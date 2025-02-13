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
        mask = self.convert_mask_to_class_indices(mask)
        
        return image, mask
    
    def convert_mask_to_class_indices(self, mask):
        """Convert RGB mask to class indices"""
        mask = mask.float()
        class_mask = torch.zeros(mask.shape[1], mask.shape[2], dtype=torch.long)
        
        for idx, row in self.class_mapping.iterrows():
            r, g, b = row['r'], row['g'], row['b']
            class_pixels = (mask[0] == r/255.0) & (mask[1] == g/255.0) & (mask[2] == b/255.0)
            class_mask[class_pixels] = idx
            
        return class_mask

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

if __name__ == "__main__":
    # Analyze dataset
    print("\nAnalyzing dataset structure...")
    num_images, image_size = analyze_dataset("dataset")
    
    # Test the dataloader
    print("\nTesting dataloader...")
    train_loader, test_loader = get_dataloaders("dataset")
    
    # Get a batch of training data
    images, masks = next(iter(train_loader))
    
    print(f"\nDataloader output:")
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")
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
    
    # Show mask
    plt.subplot(1, 2, 2)
    plt.imshow(masks[0].numpy())
    plt.title("Sample Training Mask from Batch")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
