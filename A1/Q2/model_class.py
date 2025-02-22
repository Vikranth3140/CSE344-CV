from torchvision import models
import torch
import torch.nn as nn
import wandb
from dataset_class import WildlifeDataset, CLASS_MAPPING
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

os.makedirs('weights', exist_ok=True)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Load pre-trained ResNet-18
        self.model = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-4]:  # Keep last few layers trainable
            param.requires_grad = False
            
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(
            in_channels=3,          # RGB input
            out_channels=32,        # 32 feature maps
            kernel_size=3,          # 3x3 kernel
            stride=1,
            padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(
            in_channels=32,         # Input from previous layer
            out_channels=64,        # 64 feature maps
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third Convolutional Block
        self.conv3 = nn.Conv2d(
            in_channels=64,         # Input from previous layer
            out_channels=128,       # 128 feature maps
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Classification head
        self.fc1 = nn.Linear(128 * 14 * 14, 256)  # Adjust size based on pooling
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)  # Output layer
        
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
        x = x.view(-1, 128 * 14 * 14)
        
        # Classification
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        
        return x

# We can use this function to train the resnet model with or without data augmentation by passing in the augmented parameter
def train_resnet(train_dataloader, val_dataloader, device, num_classes=10, augmented=False):
    # Initialize model
    model = ResNet18(num_classes=num_classes)
    model = model.to(device)
    
    # Initialize WandB
    wandb.init(
        project="russian-wildlife-classification",
        name="resnet18-finetuned",
        config={
            "architecture": "ResNet18",
            "dataset": "Wildlife",
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": train_dataloader.batch_size,
            "optimizer": "Adam"
        }
    )
    
    # Watch model in wandb
    wandb.watch(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_val_acc = 0.0
    num_epochs = 10
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_dataloader)
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_dataloader)
        val_acc = 100 * correct / total
        
        # Log metrics
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
            if augmented:
                torch.save(model.state_dict(), os.path.join('weights', 'resnet_aug.pth'))
            else:
                torch.save(model.state_dict(), os.path.join('weights', 'resnet.pth'))
    
    # After training loop, add validation metrics calculation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Log metrics
    wandb.log({
        "validation_accuracy": val_accuracy,
        "validation_f1": val_f1
    })
    
    # Create and log confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(CLASS_MAPPING.keys()),
                yticklabels=list(CLASS_MAPPING.keys()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()
    
    # Print metrics
    print("\nValidation Metrics:")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"F1-Score: {val_f1:.4f}")
    
    print('Training finished!')
    return model

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    
    # Remove the final classification layer to get features
    feature_extractor = torch.nn.Sequential(*list(model.model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            # Extract features (backbone output)
            batch_features = feature_extractor(images)
            # Flatten features
            batch_features = batch_features.view(batch_features.size(0), -1)
            
            features.append(batch_features.cpu())
            labels.append(batch_labels)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features.numpy(), labels.numpy()

def train_convnet(train_dataloader, val_dataloader, device, num_classes=10):
    # Initialize model
    model = ConvNet(num_classes=num_classes)
    model = model.to(device)
    
    # Initialize WandB
    wandb.init(
        project="russian-wildlife-classification",
        name="convnet-baseline",
        config={
            "architecture": "ConvNet",
            "dataset": "Wildlife",
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": train_dataloader.batch_size,
            "optimizer": "Adam"
        }
    )
    
    # Watch model in wandb
    wandb.watch(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_val_acc = 0.0
    num_epochs = 10
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_dataloader)
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_dataloader)
        val_acc = 100 * correct / total
        
        # Log metrics
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
            torch.save(model.state_dict(), os.path.join('weights', 'convnet.pth'))
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    wandb.log({
        "validation_accuracy": val_accuracy,
        "validation_f1": val_f1
    })
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(CLASS_MAPPING.keys()),
                yticklabels=list(CLASS_MAPPING.keys()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()
    
    # Metrics
    print("\nValidation Metrics:")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"F1-Score: {val_f1:.4f}")
    
    print('Training finished!')
    return model

def analyze_misclassified(model, dataloader, device, num_samples=3):
    model.eval()
    misclassified = {i: [] for i in range(len(CLASS_MAPPING))}
    inv_class_mapping = {v: k for k, v in CLASS_MAPPING.items()}
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Find misclassified samples
            mask = predicted != labels
            misclassified_images = images[mask]
            true_labels = labels[mask]
            pred_labels = predicted[mask]
            
            # Store misclassified samples
            for img, true_label, pred_label in zip(misclassified_images, true_labels, pred_labels):
                if len(misclassified[true_label.item()]) < num_samples:
                    misclassified[true_label.item()].append({
                        'image': img.cpu(),
                        'true_label': inv_class_mapping[true_label.item()],
                        'pred_label': inv_class_mapping[pred_label.item()]
                    })
    return misclassified

def visualize_misclassified(model, val_dataloader, device):
    # Get misclassified samples
    misclassified_samples = analyze_misclassified(model, val_dataloader, device)
    
    # Visualize misclassified samples for each class
    plt.figure(figsize=(15, 4*len(CLASS_MAPPING)))
    for class_idx, samples in misclassified_samples.items():
        if samples:  # If there are misclassified samples for this class
            class_name = list(CLASS_MAPPING.keys())[class_idx]
            
            for i, sample in enumerate(samples):
                plt.subplot(len(CLASS_MAPPING), 3, class_idx*3 + i + 1)
                
                # Denormalize and convert to numpy for visualization
                img = sample['image'].numpy()
                img = img * 0.5 + 0.5  # Denormalize
                img = np.transpose(img, (1, 2, 0))  # CHW to HWC
                
                plt.imshow(img)
                plt.title(f'True: {sample["true_label"]}\nPred: {sample["pred_label"]}')
                plt.axis('off')
    
    plt.tight_layout()
    
    # Log misclassified examples to wandb
    wandb.log({"Misclassified Examples": wandb.Image(plt)})
    plt.close()
    
    # Print analysis
    print("\nAnalysis of Misclassifications:")
    for class_idx, samples in misclassified_samples.items():
        if samples:
            class_name = list(CLASS_MAPPING.keys())[class_idx]
            print(f"\n{class_name}:")
            pred_counts = {}
            for sample in samples:
                pred_label = sample['pred_label']
                pred_counts[pred_label] = pred_counts.get(pred_label, 0) + 1
            print("Misclassified as:", pred_counts)

if __name__ == "__main__":
    # Define transforms

    # Use this for non data augmentation

    # transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                     std=[0.229, 0.224, 0.225])  # ImageNet normalization
    # ])
    

    # Use this for data augmentation for resnet augmentation
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images horizontally with 50% probability
    transforms.RandomRotation(degrees=15),   # Rotate images randomly within ±15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # can adjust brightness, contrast, saturation but as of now I putt 0.2 for all
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Initialize datasets
    train_dataset = WildlifeDataset("train_dataset", CLASS_MAPPING, transform=transform)
    val_dataset = WildlifeDataset("val_dataset", CLASS_MAPPING, transform=transform)
    
    # Create dataloaders
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train models
    print("\nTraining ResNet18")
    resnet_model = train_resnet(train_dataloader, val_dataloader, device, augmented=False)
    train_features, train_labels = extract_features(resnet_model, train_dataloader, device)
    val_features, val_labels = extract_features(resnet_model, val_dataloader, device)
    visualize_misclassified(resnet_model, val_dataloader, device)
    
    print("\nTraining ConvNet")
    convnet_model = train_convnet(train_dataloader, val_dataloader, device)
    visualize_misclassified(convnet_model, val_dataloader, device)

    print("\nTraining ResNet18 with augmented data")
    resnet_augmented_model = train_resnet(train_dataloader, val_dataloader, device, augmented=True)
    
    train_features_augmented, train_labels_augmented = extract_features(resnet_augmented_model, train_dataloader, device)
    val_features_augmented, val_labels_augmented = extract_features(resnet_augmented_model, val_dataloader, device)
    visualize_misclassified(resnet_augmented_model, val_dataloader, device)
    
    # Perform t-SNE
    # 2D t-SNE for training set
    print("Computing 2D t-SNE for training set...")
    tsne_2d = TSNE(n_components=2, random_state=42)
    train_tsne_2d = tsne_2d.fit_transform(train_features)

    # 2D t-SNE for validation set
    print("Computing 2D t-SNE for validation set...")
    val_tsne_2d = tsne_2d.fit_transform(val_features)

    # 3D t-SNE for validation set
    print("Computing 3D t-SNE for validation set...")
    tsne_3d = TSNE(n_components=3, random_state=42)
    val_tsne_3d = tsne_3d.fit_transform(val_features)

    # Plot 2D t-SNE for training set
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(train_tsne_2d[:, 0], train_tsne_2d[:, 1], 
                         c=train_labels, cmap='tab10')
    plt.title('2D t-SNE Visualization of Training Features')
    plt.colorbar(scatter, label='Class')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(handles=scatter.legend_elements()[0], 
              labels=list(CLASS_MAPPING.keys()),
              title="Classes",
              bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    wandb.log({"Training Features t-SNE (2D)": wandb.Image(plt)})
    plt.close()

    # Plot 2D t-SNE for validation set
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(val_tsne_2d[:, 0], val_tsne_2d[:, 1], 
                         c=val_labels, cmap='tab10')
    plt.title('2D t-SNE Visualization of Validation Features')
    plt.colorbar(scatter, label='Class')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(handles=scatter.legend_elements()[0], 
              labels=list(CLASS_MAPPING.keys()),
              title="Classes",
              bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    wandb.log({"Validation Features t-SNE (2D)": wandb.Image(plt)})
    plt.close()

    # Plot 3D t-SNE for validation set
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(val_tsne_3d[:, 0], val_tsne_3d[:, 1], val_tsne_3d[:, 2],
                        c=val_labels, cmap='tab10')
    ax.set_title('3D t-SNE Visualization of Validation Features')
    plt.colorbar(scatter, label='Class')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.legend(handles=scatter.legend_elements()[0], 
              labels=list(CLASS_MAPPING.keys()),
              title="Classes",
              bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    wandb.log({"Validation Features t-SNE (3D)": wandb.Image(plt)})
    plt.close()

    # Print feature space analysis
    print("\nFeature Space Analysis:")
    print("1. Feature vector dimensionality:", train_features.shape[1])
    print("2. Number of samples:")
    print(f"   - Training: {len(train_features)}")
    print(f"   - Validation: {len(val_features)}")
    print("\n3. Class separation analysis:")
    for class_name, class_idx in CLASS_MAPPING.items():
        class_mask = val_labels == class_idx
        if np.any(class_mask):
            class_features = val_features[class_mask]
            center = np.mean(class_features, axis=0)
            spread = np.mean(np.linalg.norm(class_features - center, axis=1))
            print(f"\n{class_name}:")
            print(f"   - Average distance from center: {spread:.2f}")

    wandb.finish()