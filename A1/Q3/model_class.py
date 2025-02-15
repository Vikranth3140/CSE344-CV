import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models,transforms
import wandb
from dataset_class import get_dataloaders, CLASS_MAPPING
import numpy as np
from tqdm import tqdm

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
    print("\nEvaluating model on test set...")
    class_names = CLASS_MAPPING['name'].tolist()
    metrics, miou = evaluate_model(trained_model, test_loader, device, class_names)
    
    # Log final test metrics to wandb
    wandb.log({
        "test_miou": miou,
        "test_mean_pixel_acc": np.mean(metrics['pixel_acc']),
        "test_mean_dice": np.mean(metrics['dice'])
    })
    
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
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_segnet.pth')
    
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
    """Train the DeepLabV3 model with different strategies from SegNet"""
    
    # Different loss function: Combine CrossEntropy with Dice Loss
    ce_criterion = nn.CrossEntropyLoss()
    dice_criterion = DiceLoss()
    
    # Different optimizer setup with weight decay
    optimizer = torch.optim.Adam([
        {'params': model.model.backbone.parameters(), 'lr': 0.00001},
        {'params': model.model.classifier.parameters(), 'lr': 0.0001}
    ], weight_decay=1e-4)
    
    # Add a more aggressive learning rate scheduler
    scheduler = torch.optim.OneCycleLR(
        optimizer,
        max_lr=[0.00001, 0.0001],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_ce_loss = 0
        total_dice_loss = 0
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate both losses
            ce_loss = ce_criterion(outputs, masks)
            dice_loss = dice_criterion(outputs, masks)
            loss = ce_loss + dice_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (different from SegNet)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update total losses
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_dice_loss += dice_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce_loss': f'{ce_loss.item():.4f}',
                'dice_loss': f'{dice_loss.item():.4f}'
            })
            
            # Log batch metrics
            wandb.log({
                "batch": batch_idx + epoch * len(train_loader),
                "batch_total_loss": loss.item(),
                "batch_ce_loss": ce_loss.item(),
                "batch_dice_loss": dice_loss.item()
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_dice_loss = total_dice_loss / len(train_loader)
        
        # Validation phase with metrics
        model.eval()
        val_loss = 0
        val_metrics = {
            'iou': [],
            'pixel_acc': [],
            'dice': []
        }
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                # Calculate validation loss
                val_ce_loss = ce_criterion(outputs, masks)
                val_dice_loss = dice_criterion(outputs, masks)
                val_loss += (val_ce_loss + val_dice_loss).item()
                
                # Calculate validation metrics
                predictions = torch.argmax(outputs, dim=1)
                batch_metrics = calculate_batch_metrics(predictions, masks)
                for k in val_metrics.keys():
                    val_metrics[k].extend(batch_metrics[k])
        
        avg_val_loss = val_loss / len(val_loader)
        mean_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_total_loss": avg_loss,
            "train_ce_loss": avg_ce_loss,
            "train_dice_loss": avg_dice_loss,
            "val_loss": avg_val_loss,
            "val_iou": mean_val_metrics['iou'],
            "val_pixel_acc": mean_val_metrics['pixel_acc'],
            "val_dice": mean_val_metrics['dice'],
            "learning_rate": current_lr
        })
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Train Loss: {avg_loss:.4f} (CE: {avg_ce_loss:.4f}, Dice: {avg_dice_loss:.4f})')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Metrics - IoU: {mean_val_metrics["iou"]:.4f}, '
              f'Pixel Acc: {mean_val_metrics["pixel_acc"]:.4f}, '
              f'Dice: {mean_val_metrics["dice"]:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save best model and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': mean_val_metrics
            }, 'best_deeplabv3.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
    
    return model

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # Convert predictions to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=predictions.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient for each class
        intersection = (predictions * targets_one_hot).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        # Calculate Dice loss
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss

def calculate_batch_metrics(predictions, targets):
    """Calculate metrics for a batch"""
    metrics = {
        'iou': [],
        'pixel_acc': [],
        'dice': []
    }
    
    for pred, target in zip(predictions, targets):
        # Calculate IoU
        intersection = torch.logical_and(pred, target)
        union = torch.logical_or(pred, target)
        iou = (intersection.sum() / (union.sum() + 1e-8)).item()
        
        # Calculate Pixel Accuracy
        pixel_acc = (pred == target).float().mean().item()
        
        # Calculate Dice Coefficient
        dice = (2 * intersection.sum() / (pred.sum() + target.sum() + 1e-8)).item()
        
        metrics['iou'].append(iou)
        metrics['pixel_acc'].append(pixel_acc)
        metrics['dice'].append(dice)
    
    return metrics

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = get_dataloaders("dataset", batch_size=8)
    
    # Initialize model
    model = SegNet_Pretrained(
        encoder_weight_pth='encoder_model.pth',
        in_chn=3,
        out_chn=32
    ).to(device)
    
    # Train and evaluate model
    trained_model, metrics, miou = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=10
    )