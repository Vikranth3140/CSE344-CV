import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import wandb
from dataset_class import get_dataloaders, CLASS_MAPPING
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class SegNet_Encoder(nn.Module):

    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
        super(SegNet_Encoder, self).__init__()

        #SegNet Architecture
        #Takes input of size in_chn = 3 (RGB images have 3 channels)
        #Outputs size label_chn (N # of classes)

        #ENCODING consists of 5 stages
        #Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
        #Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

        #General Max Pool 2D for ENCODING layers
        #Pooling indices are stored for Upsampling in DECODING layers

        self.in_chn = in_chn
        self.out_chn = out_chn

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True) 

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)
    def forward(self,x):
        #ENCODE LAYERS
        #Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x))) 
        x = F.relu(self.BNEn12(self.ConvEn12(x))) 
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        #Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x))) 
        x = F.relu(self.BNEn22(self.ConvEn22(x))) 
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        #Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x))) 
        x = F.relu(self.BNEn32(self.ConvEn32(x))) 
        x = F.relu(self.BNEn33(self.ConvEn33(x)))   
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        #Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x))) 
        x = F.relu(self.BNEn42(self.ConvEn42(x))) 
        x = F.relu(self.BNEn43(self.ConvEn43(x)))   
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        #Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x))) 
        x = F.relu(self.BNEn52(self.ConvEn52(x))) 
        x = F.relu(self.BNEn53(self.ConvEn53(x)))   
        x, ind5 = self.MaxEn(x)
        size5 = x.size()
        return x,[ind1,ind2,ind3,ind4,ind5],[size1,size2,size3,size4,size5]
    


class SegNet_Decoder(nn.Module):
    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
        super(SegNet_Decoder, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        
        # General Max Unpool for all stages
        self.MaxDe = nn.MaxUnpool2d(2, stride=2)
        
        # Stage 5
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        
        # Stage 4
        self.ConvDe41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe43 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(256, momentum=BN_momentum)
        
        # Stage 3
        self.ConvDe31 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe33 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(128, momentum=BN_momentum)
        
        # Stage 2
        self.ConvDe21 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe22 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(64, momentum=BN_momentum)
        
        # Stage 1
        self.ConvDe11 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe12 = nn.Conv2d(64, out_chn, kernel_size=3, padding=1)

    def forward(self, x, indices, sizes):
        ind1, ind2, ind3, ind4, ind5 = indices
        size1, size2, size3, size4, size5 = sizes
        
        # Stage 5 decode
        x = self.MaxDe(x, ind5, output_size=size4)
        x = F.relu(self.BNDe51(self.ConvDe51(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        
        # Stage 4 decode
        x = self.MaxDe(x, ind4, output_size=size3)
        x = F.relu(self.BNDe41(self.ConvDe41(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        
        # Stage 3 decode
        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe31(self.ConvDe31(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        
        # Stage 2 decode
        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe21(self.ConvDe21(x)))
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        
        # Stage 1 decode
        x = self.MaxDe(x, ind1)
        x = F.relu(self.BNDe11(self.ConvDe11(x)))
        x = self.ConvDe12(x)
        
        return x


class SegNet_Pretrained(nn.Module):
    def __init__(self,encoder_weight_pth,in_chn=3, out_chn=32):
        super(SegNet_Pretrained, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.encoder=SegNet_Encoder(in_chn=self.in_chn,out_chn=self.out_chn)
        self.decoder=SegNet_Decoder(in_chn=self.in_chn,out_chn=self.out_chn)
        encoder_state_dict = torch.load(encoder_weight_pth,weights_only=True)

        # Load weights into the encoder
        self.encoder.load_state_dict(encoder_state_dict)

        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self,x):
        x,indexes,sizes=self.encoder(x)
        x=self.decoder(x,indexes,sizes)
        return x


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=32):
        super(DeepLabV3, self).__init__()
        # Initialize pre-trained DeepLabV3 with ResNet50 backbone
        self.model = models.segmentation.deeplabv3_resnet50(
            weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
            progress=True
        )
        
        # Replace the classifier head with a new one for our number of classes
        self.model.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            stride=1
        )
       
    def forward(self, x):
        return self.model(x)['out']

def train_and_evaluate(model, train_loader, test_loader, device, num_epochs=10):
    """Train and evaluate the model"""
    # Initialize wandb
    wandb.init(
        project="segmentation-segnet",
        config={
            "architecture": "SegNet",
            "dataset": "CamVid",
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_norm_momentum": 0.5
        }
    )
    
    # Train model
    trained_model = train_segnet(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        num_epochs=num_epochs
    )
    
    # Evaluate model
    print("\nEvaluating model on test set")
    class_names = CLASS_MAPPING['name'].tolist()
    metrics, miou = evaluate_model(trained_model, test_loader, device, class_names)
    
    # Log final test metrics to wandb
    wandb.log({
        "test_miou": miou,
        "test_mean_pixel_acc": np.mean(metrics['pixel_acc']),
        "test_mean_dice": np.mean(metrics['dice'])
    })
    
    # Visualize poor predictions
    print("\nVisualizing poor predictions")
    visualize_poor_predictions(trained_model, test_loader, device)
    
    wandb.finish()
    return trained_model, metrics, miou

def train_segnet(model, train_loader, val_loader, device, num_epochs=10):
    """Train the SegNet model"""
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=0.001)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log batch loss
            wandb.log({
                "batch": batch_idx + epoch * len(train_loader),
                "batch_loss": loss.item()
            })
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })
        
        print(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save decoder weights
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.decoder.state_dict(), 'decoder.pth')
    
    return model

def calculate_metrics(pred_mask, true_mask, num_classes, iou_thresholds=np.arange(0, 1.1, 0.1)):
    """
    Calculate segmentation metrics for each class
    Args:
        pred_mask: Predicted mask (B, H, W)
        true_mask: Ground truth mask (B, H, W)
        num_classes: Number of classes
        iou_thresholds: IoU thresholds for evaluation
    Returns:
        Dictionary containing metrics for each class
    """
    metrics = {
        'pixel_acc': np.zeros(num_classes),
        'dice': np.zeros(num_classes),
        'iou': np.zeros(num_classes),
        'precision': np.zeros((num_classes, len(iou_thresholds))),
        'recall': np.zeros((num_classes, len(iou_thresholds)))
    }
    
    # Convert tensors to numpy arrays
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    if torch.is_tensor(true_mask):
        true_mask = true_mask.cpu().numpy()
    
    # Calculate metrics for each class
    for class_idx in range(num_classes):
        # Create binary masks for current class
        pred_binary = (pred_mask == class_idx)
        true_binary = (true_mask == class_idx)
        
        # Pixel accuracy
        total_pixels = true_binary.size
        correct_pixels = np.sum(pred_binary == true_binary)
        metrics['pixel_acc'][class_idx] = correct_pixels / total_pixels
        
        # Intersection and Union
        intersection = np.sum(pred_binary & true_binary)
        union = np.sum(pred_binary | true_binary)
        
        # Dice coefficient
        dice = 2 * intersection / (np.sum(pred_binary) + np.sum(true_binary) + 1e-8)
        metrics['dice'][class_idx] = dice
        
        # IoU
        iou = intersection / (union + 1e-8)
        metrics['iou'][class_idx] = iou
        
        # Precision and Recall for different IoU thresholds
        for i, threshold in enumerate(iou_thresholds):
            # True Positive: prediction matches ground truth with IoU > threshold
            tp = np.sum((pred_binary & true_binary) & (iou > threshold))
            # False Positive: prediction doesn't match ground truth
            fp = np.sum(pred_binary & ~true_binary)
            # False Negative: missed ground truth
            fn = np.sum(~pred_binary & true_binary)
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            metrics['precision'][class_idx, i] = precision
            metrics['recall'][class_idx, i] = recall
    
    return metrics

def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate model on test set and report metrics
    """
    model.eval()
    num_classes = len(class_names)
    iou_thresholds = np.arange(0, 1.1, 0.1)
    
    # Initialize metrics storage
    all_metrics = {
        'pixel_acc': np.zeros(num_classes),
        'dice': np.zeros(num_classes),
        'iou': np.zeros(num_classes),
        'precision': np.zeros((num_classes, len(iou_thresholds))),
        'recall': np.zeros((num_classes, len(iou_thresholds)))
    }
    
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate metrics for this batch
            batch_metrics = calculate_metrics(predictions, masks, num_classes, iou_thresholds)
            
            # Accumulate metrics
            for metric in all_metrics:
                all_metrics[metric] += batch_metrics[metric]
            
            num_batches += 1
    
    # Average metrics
    for metric in all_metrics:
        all_metrics[metric] /= num_batches
    
    # Calculate mIoU
    miou = np.mean(all_metrics['iou'])
    
    # Print results
    print("\nTest Set Evaluation Results:")
    print("=" * 50)
    print(f"Mean IoU (mIoU): {miou:.4f}")
    print("\nClass-wise Results:")
    print("-" * 50)
    print(f"{'Class':<20} {'Pixel Acc':>10} {'Dice':>10} {'IoU':>10}")
    print("-" * 50)
    
    for idx, class_name in enumerate(class_names):
        print(f"{class_name:<20} {all_metrics['pixel_acc'][idx]:>10.4f} {all_metrics['dice'][idx]:>10.4f} {all_metrics['iou'][idx]:>10.4f}")
    
    # Print Precision and Recall for different IoU thresholds
    print("\nPrecision and Recall at different IoU thresholds:")
    print("-" * 80)
    print(f"{'Class':<20} {'Threshold':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 80)
    
    for idx, class_name in enumerate(class_names):
        for t_idx, threshold in enumerate(iou_thresholds):
            print(f"{class_name if t_idx == 0 else '':<20} {threshold:>10.1f} {all_metrics['precision'][idx, t_idx]:>10.4f} {all_metrics['recall'][idx, t_idx]:>10.4f}")
    
    return all_metrics, miou

def train_deeplabv3(model, train_loader, val_loader, device, num_epochs=10):
    """Train the DeepLabv3 model"""
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5  # Early stopping patience
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log batch loss
            wandb.log({
                "batch": batch_idx + epoch * len(train_loader),
                "batch_loss": loss.item()
            })
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'deeplabv3.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    return model

def visualize_poor_predictions(model, test_loader, device, iou_threshold=0.5, samples_per_class=3):
    """
    Visualize images where model performs poorly (IoU ≤ 0.5) for Car, Pedestrian, and Road classes
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run model on
        iou_threshold: Maximum IoU threshold to consider as poor prediction
        samples_per_class: Number of samples to show for each class
    """
    model.eval()
    
    # Select specific classes
    selected_class_names = ['Car', 'Pedestrian', 'Road']
    selected_classes = [CLASS_MAPPING[CLASS_MAPPING['name'] == name].index[0] for name in selected_class_names]
    class_samples = {idx: [] for idx in selected_classes}
    
    with torch.no_grad():
        for images, masks in test_loader:
            if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                break
                
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Move tensors to CPU
            images = images.cpu()
            masks = masks.cpu()
            predictions = predictions.cpu()
            
            # Check each image in batch
            for img, mask, pred in zip(images, masks, predictions):
                # Calculate IoU for selected classes in this image
                for class_idx in selected_classes:
                    if len(class_samples[class_idx]) >= samples_per_class:
                        continue
                        
                    true_mask = (mask == class_idx)
                    pred_mask = (pred == class_idx)
                    
                    if not true_mask.any():
                        continue  # Skip if class not present in ground truth
                    
                    intersection = (true_mask & pred_mask).sum().item()
                    union = (true_mask | pred_mask).sum().item()
                    iou = intersection / (union + 1e-8)
                    
                    if iou <= iou_threshold:
                        # Denormalize image
                        img_np = img.permute(1, 2, 0).numpy()
                        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        img_np = np.clip(img_np, 0, 1)
                        
                        class_samples[class_idx].append({
                            'image': img_np,
                            'true_mask': true_mask.numpy(),
                            'pred_mask': pred_mask.numpy(),
                            'iou': iou
                        })
    
    # Visualize samples for selected classes
    for class_idx in selected_classes:
        samples = class_samples[class_idx]
        class_name = CLASS_MAPPING['name'][class_idx]
        class_color = [
            CLASS_MAPPING['r'][class_idx]/255.0,
            CLASS_MAPPING['g'][class_idx]/255.0,
            CLASS_MAPPING['b'][class_idx]/255.0
        ]
        
        if samples:
            plt.figure(figsize=(15, 5*min(len(samples), samples_per_class)))
            plt.suptitle(f'Poor Predictions for Class: {class_name} (IoU ≤ {iou_threshold})', fontsize=14)
            
            for i, sample in enumerate(samples[:samples_per_class]):
                # Original image
                plt.subplot(min(len(samples), samples_per_class), 3, i*3 + 1)
                plt.imshow(sample['image'])
                plt.title(f'Original Image\nIoU: {sample["iou"]:.3f}')
                plt.axis('off')
                
                # Ground truth mask
                plt.subplot(min(len(samples), samples_per_class), 3, i*3 + 2)
                overlay = sample['image'].copy()
                overlay[sample['true_mask']] = class_color
                plt.imshow(overlay)
                plt.title('Ground Truth')
                plt.axis('off')
                
                # Predicted mask
                plt.subplot(min(len(samples), samples_per_class), 3, i*3 + 3)
                overlay = sample['image'].copy()
                overlay[sample['pred_mask']] = class_color
                plt.imshow(overlay)
                plt.title('Prediction')
                plt.axis('off')
                
                # Add analysis of failure case
                if sample['true_mask'].sum() < 100:
                    plt.figtext(0.99, 0.99-i*0.33, "Possible issue: Small/distant object", 
                              wrap=True, horizontalalignment='right')
                elif sample['pred_mask'].sum() > 2*sample['true_mask'].sum():
                    plt.figtext(0.99, 0.99-i*0.33, "Possible issue: Over-segmentation/confusion with surroundings", 
                              wrap=True, horizontalalignment='right')
                elif sample['pred_mask'].sum() < 0.5*sample['true_mask'].sum():
                    plt.figtext(0.99, 0.99-i*0.33, "Possible issue: Under-segmentation/occlusion", 
                              wrap=True, horizontalalignment='right')
                else:
                    plt.figtext(0.99, 0.99-i*0.33, "Possible issue: Misclassification due to similar appearance", 
                              wrap=True, horizontalalignment='right')
            
            plt.tight_layout()
            plt.show()
            
            # Log to wandb
            wandb.log({f"Poor_Predictions_{class_name}": wandb.Image(plt)})
        else:
            print(f"No poor predictions found for class: {class_name}")

if __name__ == "__main__":

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    # train_loader, test_loader = get_dataloaders("dataset", batch_size=8)
    
    
    train_loader, test_loader = get_dataloaders("dataset", batch_size=8, 
                                               drop_last=True)  # Increased batch size
    




    # # SegNet

    # # Initialize model
    # model = SegNet_Pretrained(
    #     encoder_weight_pth='encoder_model.pth',
    #     in_chn=3,
    #     out_chn=32
    # ).to(device)
    
    # # Train and evaluate model
    # trained_model, metrics, miou = train_and_evaluate(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=device,
    #     num_epochs=10
    # )




    # DeepLabV3
    
    # Initialize model
    num_classes = len(CLASS_MAPPING)
    model = DeepLabV3(num_classes=num_classes).to(device)
    
    # Initialize wandb
    wandb.init(
        project="segmentation-deeplabv3",
        config={
            "architecture": "DeepLabv3",
            "backbone": "ResNet50",
            "dataset": "CamVid",
            "num_classes": num_classes,
            "batch_size": train_loader.batch_size,
            "optimizer": "Adam",
            "learning_rate": 0.001
        }
    )
    
    # Train and evaluate model
    trained_model = train_deeplabv3(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        num_epochs=10
    )
    
    # Evaluate model
    print("\nEvaluating model on test set")
    class_names = CLASS_MAPPING['name'].tolist()
    metrics, miou = evaluate_model(trained_model, test_loader, device, class_names)
    
    # Log final test metrics
    wandb.log({
        "test_miou": miou,
        "test_mean_pixel_acc": np.mean(metrics['pixel_acc']),
        "test_mean_dice": np.mean(metrics['dice'])
    })
    
    wandb.finish()