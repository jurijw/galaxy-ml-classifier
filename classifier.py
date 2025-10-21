"""
Galaxy Zoo CNN Training Script

This script loads galaxy images from folders (elliptical and spiral),
applies preprocessing and augmentation, defines a simple CNN,
trains the model, evaluates its performance, and saves the trained weights.
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """Load galaxy image dataset and create train and validation DataLoaders.

    Args:
        data_dir: Path to the folder containing subfolders 'elliptical' and 'spiral'.
        batch_size: Batch size for the DataLoaders.
        val_split: Fraction of data to use for validation.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


class GalaxyCNN(nn.Module):
    """Simple CNN for galaxy classification into elliptical or spiral."""

    def __init__(self):
        super().__init__()
        # Here, we use 2 convolutional layers. The first has 3 inputs corresponding
        # to the 3 rgb input channels feeding into 16 3x3 kernels.
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 10
) -> None:
    """Train the CNN model.

    Args:
        model: CNN model to train.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on (CPU or GPU).
        num_epochs: Number of training epochs.
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")


def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Evaluate the CNN on the validation set.

    Args:
        model: Trained CNN model.
        val_loader: DataLoader for validation data.
        device: Device to run evaluation on.

    Returns:
        Validation accuracy in percent.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    """Main entry point: load data, train CNN, evaluate, and save model."""
    data_dir = "images"
    batch_size = 32
    num_epochs = 10
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    model = GalaxyCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_model(model, train_loader, criterion, optimizer, device, num_epochs)
    val_acc = evaluate_model(model, val_loader, device)

    print(f"Validation Accuracy: {val_acc:.2f}%")

    torch.save(model.state_dict(), "galaxy_cnn.pth")
    print("Model saved as galaxy_cnn.pth")


if __name__ == "__main__":
    main()
