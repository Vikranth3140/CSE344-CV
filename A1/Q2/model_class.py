import torchvision 
from torchvision import models
import torch
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, num_classes: int=10):
        super(ResNet18, self).__init__()
        self.model = None  # Use resnet18 from torchvision with Pretrained Weights.
        self.model.fc = torch.nn.Linear(None, num_classes) #Replace None with resnet18's number of input features for FC.
    def forward(self, x):
        return self.model(x)
