#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet CIFAR10 Training with Best Configuration
LR=0.15, WD=0.0001, Batch Size=128, Cosine Scheduler, 100 epochs
"""

import gc
import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor, no_grad, manual_seed
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import datasets, transforms

from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.simulator.configs import (
    SingleRPUConfig,
    FloatingPointRPUConfig,
    MappingParameter,
    PiecewiseStepDevice,
)
from aihwkit.simulator.rpu_base import cuda


# ==============================================================================
# Device Setup
# ==============================================================================
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

# ==============================================================================
# Paths
# ==============================================================================
PATH_DATASET = os.path.join(os.path.dirname(__file__), "data", "DATASET")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "cifar10_best_config_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# Best Configuration from Stage 1 Screening
# ==============================================================================
BEST_CONFIG = {
    'lr': 0.15,
    'weight_decay': 0.0001,
    'batch_size': 128,
    'scheduler': 'cosine',
}

N_EPOCHS = 100
N_CLASSES = 10
SEED = 42

# ==============================================================================
# Load Fitted Device Configuration
# ==============================================================================
DEVICE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'xor_device_config.json')

with open(DEVICE_CONFIG_PATH, 'r') as f:
    device_config = json.load(f)

# Extract device parameters (exclude metadata)
device_params = {k: v for k, v in device_config.items() if not k.startswith('_')}
FITTED_DEVICE = PiecewiseStepDevice(**device_params)

print(f"Loaded fitted device config:")
print(f"  dw_min: {device_config['dw_min']}")
print(f"  write_noise_std: {device_config['write_noise_std']:.4f}")


def create_fitted_rpu_config():
    """Create RPU config with fitted PiecewiseStepDevice."""
    mapping = MappingParameter(
        weight_scaling_omega=0.6,
        max_input_size=512,
        max_output_size=512
    )
    config = SingleRPUConfig(device=FITTED_DEVICE)
    config.mapping = mapping
    return config


def create_fp_rpu_config():
    """Create FloatingPoint RPU config for input/output layers."""
    return FloatingPointRPUConfig()


# ==============================================================================
# ResNet Model Definition
# ==============================================================================
class ResidualBlockFitted(nn.Module):
    """Residual block with Fitted Device analog convolutional layers."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1):
        super().__init__()

        rpu_config = create_fitted_rpu_config()

        self.conv1 = AnalogConv2d(
            in_ch, hidden_ch,
            kernel_size=3, padding=1, stride=stride,
            bias=False,
            rpu_config=rpu_config
        )
        self.bn1 = nn.BatchNorm2d(hidden_ch)

        self.conv2 = AnalogConv2d(
            hidden_ch, hidden_ch,
            kernel_size=3, padding=1,
            bias=False,
            rpu_config=create_fitted_rpu_config()
        )
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = AnalogConv2d(
                in_ch, hidden_ch,
                kernel_size=1, stride=stride,
                bias=False,
                rpu_config=create_fitted_rpu_config()
            )
        else:
            self.convskip = None

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.convskip:
            x = self.convskip(x)
        y += x
        return F.relu(y)


def concatenate_layer_blocks(in_ch, hidden_ch, num_layer, first_layer=False):
    """Concatenate multiple residual blocks to form a layer."""
    layers = []
    for i in range(num_layer):
        if i == 0 and not first_layer:
            layers.append(ResidualBlockFitted(in_ch, hidden_ch, use_conv=True, stride=2))
        else:
            layers.append(ResidualBlockFitted(hidden_ch, hidden_ch))
    return layers


def create_model():
    """
    Create ResNet model:
    - Input Conv layer: FloatingPoint (digital)
    - Hidden Conv layers: Fitted PiecewiseStepDevice (analog)
    - Final FC layer: FloatingPoint (digital)
    """
    block_per_layers = (3, 4, 6, 3)  # ResNet-like structure
    base_channel = 16
    channel = (base_channel, 2 * base_channel, 4 * base_channel)

    # Input layer - FloatingPoint for stability
    l0 = nn.Sequential(
        AnalogConv2d(
            3, channel[0],
            kernel_size=3, stride=1, padding=1,
            bias=True,
            rpu_config=create_fp_rpu_config()
        ),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
    )

    # Hidden layers - Fitted Device
    l1 = nn.Sequential(
        *concatenate_layer_blocks(channel[0], channel[0], block_per_layers[0], first_layer=True)
    )
    l2 = nn.Sequential(*concatenate_layer_blocks(channel[0], channel[1], block_per_layers[1]))
    l3 = nn.Sequential(*concatenate_layer_blocks(channel[1], channel[2], block_per_layers[2]))

    # Final classification layer - FloatingPoint for stability
    l4 = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        AnalogLinear(
            channel[2], N_CLASSES,
            bias=True,
            rpu_config=create_fp_rpu_config()
        )
    )

    model = nn.Sequential(l0, l1, l2, l3, l4)
    return model


# ==============================================================================
# Data Loading
# ==============================================================================
def load_images(batch_size):
    """Load CIFAR10 with data augmentation."""
    mean = Tensor([0.4914, 0.4822, 0.4465])
    std = Tensor([0.2470, 0.2435, 0.2616])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=train_transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=val_transform)

    num_workers = 4 if USE_CUDA else 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=USE_CUDA,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=USE_CUDA,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader


# ==============================================================================
# Training Functions
# ==============================================================================
def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch_max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(train_loader.dataset), 100.0 * correct / total


def evaluate(model, val_loader, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            output = model(images)
            loss = criterion(output, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch_max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(val_loader.dataset), 100.0 * correct / total


# ==============================================================================
# Main Training
# ==============================================================================
def main():
    print("=" * 80)
    print("ResNet CIFAR10 Training - Best Configuration")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Learning Rate: {BEST_CONFIG['lr']}")
    print(f"  Weight Decay: {BEST_CONFIG['weight_decay']}")
    print(f"  Batch Size: {BEST_CONFIG['batch_size']}")
    print(f"  Scheduler: {BEST_CONFIG['scheduler']}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"\nDevice Configuration:")
    print(f"  Input layer: FloatingPointRPUConfig")
    print(f"  Hidden layers: PiecewiseStepDevice (fitted, dw_min={device_config['dw_min']})")
    print(f"  Final FC layer: FloatingPointRPUConfig")
    print("=" * 80)

    # Set seeds
    manual_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True

    # Create model
    print("\nCreating model...")
    model = create_model()
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Count analog layers
    analog_conv = sum(1 for m in model.modules() if isinstance(m, AnalogConv2d))
    analog_linear = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"Analog Conv2d layers: {analog_conv}")
    print(f"Analog Linear layers: {analog_linear}")

    # Load data
    print("\nLoading CIFAR10 data...")
    train_loader, val_loader = load_images(BEST_CONFIG['batch_size'])
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Optimizer
    optimizer = AnalogSGD(
        model.parameters(),
        lr=BEST_CONFIG['lr'],
        momentum=0.9,
        weight_decay=BEST_CONFIG['weight_decay']
    )
    optimizer.regroup_param_groups(model)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    best_epoch = 0

    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)

    start_time = time.time()

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pth"))
            marker = " *BEST*"
        else:
            marker = ""

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | "
              f"Train: {train_acc:5.2f}% (loss={train_loss:.4f}) | "
              f"Val: {val_acc:5.2f}% (loss={val_loss:.4f}) | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s{marker}", flush=True)

    total_time = time.time() - start_time

    # ==============================================================================
    # Save Results
    # ==============================================================================
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  Final Validation Accuracy: {val_accs[-1]:.2f}%")
    print(f"  Total Training Time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")

    # Save metrics to CSV
    epochs = list(range(1, N_EPOCHS + 1))
    df = pd.DataFrame({
        'Epoch': epochs,
        'Train_Loss': train_losses,
        'Train_Accuracy': train_accs,
        'Val_Loss': val_losses,
        'Val_Accuracy': val_accs
    })
    df.to_csv(os.path.join(RESULTS_DIR, 'training_metrics.csv'), index=False)

    # Save summary
    summary = {
        'config': BEST_CONFIG,
        'n_epochs': N_EPOCHS,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_val_acc': val_accs[-1],
        'final_train_acc': train_accs[-1],
        'total_time_minutes': total_time / 60,
        'device_config': {
            'dw_min': device_config['dw_min'],
            'write_noise_std': device_config['write_noise_std'],
        }
    }
    with open(os.path.join(RESULTS_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Training Loss\nLR={BEST_CONFIG["lr"]}, WD={BEST_CONFIG["weight_decay"]}', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].axhline(y=best_val_acc, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_val_acc:.2f}%')
    axes[1].axvline(x=best_epoch, color='g', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Training Accuracy\nBest: {best_val_acc:.2f}% @ epoch {best_epoch}', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("  - best_model.pth")
    print("  - training_metrics.csv")
    print("  - training_summary.json")
    print("  - training_curves.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
