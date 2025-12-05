#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST Training with Fitted Device
Same setup as 03_mnist_training.py but using fitted PiecewiseStepDevice
Generates accuracy and loss plots.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, PiecewiseStepDevice
from aihwkit.simulator.rpu_base import cuda

# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

# Path where the datasets will be stored
PATH_DATASET = os.path.join("data", "DATASET")

# Network definition (same as 03_mnist_training.py)
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training parameters
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.05

# Load fitted device config
with open('xor_device_config.json', 'r') as f:
    device_config = json.load(f)

# Create fitted device
device_params = {k: v for k, v in device_config.items() if not k.startswith('_')}
fitted_device = PiecewiseStepDevice(**device_params)
RPU_CONFIG = SingleRPUConfig(device=fitted_device)

print(f"Device: PiecewiseStepDevice (dw_min={device_config['dw_min']})")


def load_images():
    """Load MNIST images."""
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)

    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_data, validation_data


def create_analog_network():
    """Create analog MLP network with fitted device."""
    model = AnalogSequential(
        AnalogLinear(INPUT_SIZE, HIDDEN_SIZES[0], True, rpu_config=RPU_CONFIG),
        nn.Sigmoid(),
        AnalogLinear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], True, rpu_config=RPU_CONFIG),
        nn.Sigmoid(),
        AnalogLinear(HIDDEN_SIZES[1], OUTPUT_SIZE, True, rpu_config=RPU_CONFIG),
        nn.LogSoftmax(dim=1),
    )

    if USE_CUDA:
        model.cuda()

    return model


def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(train_loader), correct / total * 100


def evaluate(model, val_loader, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            images = images.view(images.shape[0], -1)

            output = model(images)
            loss = criterion(output, labels)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(val_loader), correct / total * 100


def plot_results(train_losses, train_accs, val_losses, val_accs):
    """Plot and save training results."""
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('MNIST Training Loss (Fitted Device)', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('MNIST Training Accuracy (Fitted Device)', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mnist_fitted_device_training.png', dpi=150, bbox_inches='tight')
    print("Saved: mnist_fitted_device_training.png")
    plt.close()


def main():
    print("=" * 70)
    print("MNIST Training with Fitted Device")
    print("=" * 70)
    print(f"Setup (same as 03_mnist_training.py):")
    print(f"  Input size: {INPUT_SIZE}")
    print(f"  Hidden sizes: {HIDDEN_SIZES}")
    print(f"  Output size: {OUTPUT_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Scheduler: StepLR(step_size=10, gamma=0.5)")
    print(f"  Device: Fitted PiecewiseStepDevice")
    print("=" * 70)

    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, val_loader = load_images()
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    print("\nCreating model...")
    model = create_analog_network()
    print(model)

    # Optimizer and scheduler (same as 03_mnist_training.py)
    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    # Training
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step()

    # Plot results
    print("\nGenerating plots...")
    plot_results(train_losses, train_accs, val_losses, val_accs)

    # Save results
    results = {
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'best_val_acc': max(val_accs),
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'hidden_sizes': HIDDEN_SIZES,
            'scheduler': 'StepLR(step_size=10, gamma=0.5)',
            'device': f"Fitted PiecewiseStepDevice (dw_min={device_config['dw_min']})",
        }
    }

    with open('mnist_fitted_device_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: mnist_fitted_device_results.json")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final Train Accuracy: {train_accs[-1]:.2f}%")
    print(f"Final Validation Accuracy: {val_accs[-1]:.2f}%")
    print(f"Best Validation Accuracy: {max(val_accs):.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
