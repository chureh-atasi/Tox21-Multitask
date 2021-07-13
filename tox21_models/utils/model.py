import torch
import torch.nn as nn

class LiverNet(nn.Module):
    """Neural Network for liver data"""
    def __init__(self, D_in, D_out, H=200):
        """Add two hidden layers and output depending on number of classes"""
        super().__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, D_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x.squeeze()

