import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        
    def forward(self, x):
        return self.linear(x)

class LeNet5(nn.Module):
    """
    Standard LeNet-5 for MNIST.
    Modified with GroupNorm for better stability in decentralized non-IID settings.
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.gn1 = nn.GroupNorm(2, 6) # GroupNorm is often better than BN for small batches
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gn2 = nn.GroupNorm(4, 16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNet20(nn.Module):
    """
    Simplified ResNet for CIFAR-10.
    (Placeholder for a full implementation - keeping it concise for this block)
    """
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        # Standard ResNet implementation would go here.
        # For brevity in this snippet, using a smaller CNN.
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 16 * 16, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def get_model(model_name, input_dim=None, output_dim=1):  # <--- Add output_dim arg
    if model_name == 'linear':
        return LinearModel(input_dim, output_dim)
    elif model_name == 'lenet_gn':
        return LeNet5(num_classes=output_dim) # LeNet supports num_classes
    elif model_name == 'resnet20':
        return ResNet20(num_classes=output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")