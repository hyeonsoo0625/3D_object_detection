import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        self.k = k

        self.mlp = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k)
        )

        self.id_matrix = torch.eye(k).flatten()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mlp(x)  # Shape: (B, 1024, N)
        x = torch.max(x, 2)[0]  # Global max pooling, Shape: (B, 1024)
        x = self.fc(x)  # Shape: (B, k*k)

        # Add identity matrix to ensure numerical stability
        x = x.view(-1, self.k, self.k) + self.id_matrix.to(x.device).view(self.k, self.k)
        return x

class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # Input Transform
        transform = self.input_transform(x)  # Shape: (B, 3, 3)
        x = torch.bmm(transform, x)  # Apply transform, Shape: (B, 3, N)

        # Feature Extraction
        x = self.mlp1(x)  # Shape: (B, 64, N)

        # Feature Transform
        transform = self.feature_transform(x)  # Shape: (B, 64, 64)
        x = torch.bmm(transform, x)  # Apply transform, Shape: (B, 64, N)

        # Global Features
        x = self.mlp2(x)  # Shape: (B, 1024, N)
        x = torch.max(x, 2)[0]  # Global max pooling, Shape: (B, 1024)

        # Classification
        x = self.fc(x)  # Shape: (B, num_classes)
        return x

if __name__ == "__main__":
    batch_size = 32
    num_points = 1024
    num_classes = 10

    model = PointNet(num_classes=num_classes)
    input_data = torch.rand(batch_size, 3, num_points)  # Random point cloud data
    output = model(input_data)

    print("Output shape:", output.shape)  # Expected: (batch_size, num_classes)
