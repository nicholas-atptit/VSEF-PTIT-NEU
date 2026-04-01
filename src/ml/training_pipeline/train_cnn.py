"""1D-CNN Training Pipeline for LOB Microstructure Analysis.

Uses 10-level Limit Order Book (LOB) data to predict mid-price 
movement (-1, 0, 1) and optimize slippage.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ── Focal Loss for Imbalanced Tick Data ──────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

# ── 1D-CNN Model Architecture ───────────────────────────────────────

class MicrostructureCNN(nn.Module):
    def __init__(self, num_features=40, seq_len=100):
        super(MicrostructureCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * (seq_len // 2 // 2), 256)
        self.fc2 = nn.Linear(256, 3) # [Down, Neutral, Up]
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (Batch, Features, SeqLen)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ── Training Loop ───────────────────────────────────────────────────

def train_cnn(train_loader, val_loader, epochs=50):
    model = MicrostructureCNN(num_features=40, seq_len=100)
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            # Data Augmentation (Manual Gaussian Noise)
            noise = torch.randn_like(inputs) * 0.01
            inputs = inputs + noise

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation logic...
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("✅ CNN Training Complete. Saving weights...")
    torch.save(model.state_dict(), "models/cnn_microstructure.pth")
    return model

if __name__ == "__main__":
    print("🚀 1D-CNN Microstructure Training Script Loaded. Input Shape: (40, 100)")
