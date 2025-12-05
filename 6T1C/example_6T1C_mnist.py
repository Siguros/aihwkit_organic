# -*- coding: utf-8 -*-

"""Example: MNIST Training with 6T1C Device Preset.

This example demonstrates how to train a simple neural network on MNIST
using the 6T1C device preset configuration.

The 6T1C (6 Transistors, 1 Capacitor) device is a capacitor-based synaptic
memory with the following characteristics:
- ~1000 conductance states
- Exponential retention decay (τ ≈ 775 min)
- Near-linear weight update behavior

Usage:
    python example_6T1C_mnist.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add the 6T1C directory to path for importing preset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# AIHWKit imports
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD

# Import 6T1C preset
from preset_6T1C import (
    SixT1CPreset,
    SixT1CPresetNoRetention,
    SixT1C2Preset,
    TikiTakaSixT1CPreset,
    print_device_info,
)


# =============================================================================
# Configuration
# =============================================================================

# Training parameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.01

# Network architecture
INPUT_SIZE = 784  # 28x28 MNIST images
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10

# Device selection
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Select which 6T1C preset to use
# Options: 'single', 'single_no_retention', 'dual', 'tiki_taka'
PRESET_TYPE = 'single'


# =============================================================================
# Data Loading
# =============================================================================

def load_mnist_data():
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=USE_CUDA
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=USE_CUDA
    )

    return train_loader, test_loader


# =============================================================================
# Model Definition
# =============================================================================

def create_analog_model(preset_type='single'):
    """Create analog neural network with 6T1C devices.

    Args:
        preset_type: Type of preset to use
            - 'single': Single 6T1C device with retention
            - 'single_no_retention': Single device without retention
            - 'dual': Two devices per cross-point
            - 'tiki_taka': Tiki-taka optimizer

    Returns:
        AnalogSequential model
    """
    # Select RPU configuration based on preset type
    if preset_type == 'single':
        rpu_config = SixT1CPreset()
        print("Using: SixT1CPreset (single device with retention)")
    elif preset_type == 'single_no_retention':
        rpu_config = SixT1CPresetNoRetention()
        print("Using: SixT1CPresetNoRetention (single device without retention)")
    elif preset_type == 'dual':
        rpu_config = SixT1C2Preset()
        print("Using: SixT1C2Preset (two devices per cross-point)")
    elif preset_type == 'tiki_taka':
        rpu_config = TikiTakaSixT1CPreset()
        print("Using: TikiTakaSixT1CPreset (Tiki-taka optimizer)")
    else:
        raise ValueError(f"Unknown preset type: {preset_type}")

    # Create analog model
    model = AnalogSequential(
        AnalogLinear(INPUT_SIZE, HIDDEN_SIZE, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(HIDDEN_SIZE, OUTPUT_SIZE, bias=True, rpu_config=rpu_config),
    )

    return model


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_epoch(model, train_loader, optimizer, criterion, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, INPUT_SIZE).to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, INPUT_SIZE).to(DEVICE)
            target = target.to(DEVICE)

            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return test_loss, accuracy


# =============================================================================
# Main
# =============================================================================

def main():
    """Main training loop."""
    print("=" * 70)
    print("6T1C Device MNIST Training Example")
    print("=" * 70)

    # Print device info
    print_device_info()

    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Preset type: {PRESET_TYPE}")
    print()

    # Load data
    print("Loading MNIST data...")
    train_loader, test_loader = load_mnist_data()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    print("\nCreating analog model...")
    model = create_analog_model(PRESET_TYPE)
    model = model.to(DEVICE)

    # Print model summary
    print(f"\nModel architecture:")
    print(model)

    # Setup optimizer
    optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE)
    optimizer.regroup_param_groups(model)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    best_accuracy = 0
    results = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, epoch
        )

        test_loss, test_acc = evaluate(model, test_loader, criterion)

        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            # Optionally save best model
            # torch.save(model.state_dict(), '6t1c_mnist_best.pth')

    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nBest Test Accuracy: {best_accuracy:.2f}%")

    print("\nEpoch Results:")
    print("-" * 60)
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>12} {'Test Loss':>12} {'Test Acc':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['epoch']:>6} {r['train_loss']:>12.4f} {r['train_acc']:>11.2f}% "
              f"{r['test_loss']:>12.4f} {r['test_acc']:>11.2f}%")
    print("-" * 60)

    return results


if __name__ == "__main__":
    main()
