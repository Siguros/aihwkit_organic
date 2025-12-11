#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet CIFAR10 Hyperparameter Search with Fitted PiecewiseStepDevice

- Input layer & Last FC layer: FloatingPointRPUConfig (digital precision)
- Hidden Conv layers: Fitted PiecewiseStepDevice (from organic device fitting)

Searches: Learning Rate, Weight Decay, Scheduler parameters, Batch Size
Saves optimal results with accuracy/loss curves.
"""

import gc
import json
import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor, no_grad, manual_seed
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR

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
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")
RESULTS_DIR = os.path.join(os.getcwd(), "cifar10_resnet_search_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# Fixed Training Parameters
# ==============================================================================
N_CLASSES = 10
SEED = 42

# Two-stage search parameters
STAGE1_EPOCHS = 10   # Quick screening
STAGE2_EPOCHS = 100  # Full training for top candidates
TOP_K = 10           # Number of top configs to fully train

# ==============================================================================
# Hyperparameter Search Space (Extended for 24h search)
# ==============================================================================
SEARCH_SPACE = {
    'lr': [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15],
    'weight_decay': [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
    'batch_size': [128],  # Fixed
    'scheduler': ['cosine'],  # Fixed: Cosine Annealing
}

# Total configs = 8 * 7 * 1 * 1 = 56 configurations

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


def get_scheduler(optimizer, scheduler_type, n_epochs):
    """Create learning rate scheduler."""
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    elif scheduler_type == 'step':
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'multistep':
        return MultiStepLR(optimizer, milestones=[50, 75, 90], gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


# ==============================================================================
# Single Configuration Training
# ==============================================================================
def train_single_config(config, train_loader, val_loader, n_epochs, save_results=False, config_name=None):
    """Train a single hyperparameter configuration."""

    manual_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True

    # Create model
    model = create_model()
    model.to(DEVICE)

    # Optimizer
    optimizer = AnalogSGD(
        model.parameters(),
        lr=config['lr'],
        momentum=0.9,
        weight_decay=config['weight_decay']
    )
    optimizer.regroup_param_groups(model)

    # Scheduler
    scheduler = get_scheduler(optimizer, config['scheduler'], n_epochs)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            if save_results and config_name:
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{config_name}_best.pth"))

        scheduler.step()

        # Print progress every epoch
        print(f"    Epoch {epoch+1:3d}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Best={best_val_acc:.2f}%", flush=True)

    # Save results if requested
    if save_results and config_name:
        save_training_results(config_name, config, train_losses, train_accs, val_losses, val_accs, best_val_acc, best_epoch)

    # Memory cleanup
    del model, optimizer, scheduler, criterion
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'final_val_acc': val_accs[-1],
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_val_loss': val_losses[-1],
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }


def save_training_results(config_name, config, train_losses, train_accs, val_losses, val_accs, best_val_acc, best_epoch):
    """Save training results to files."""
    config_dir = os.path.join(RESULTS_DIR, config_name)
    os.makedirs(config_dir, exist_ok=True)

    # Save metrics to Excel
    epochs = list(range(1, len(train_losses) + 1))
    df = pd.DataFrame({
        'Epoch': epochs,
        'Train_Loss': train_losses,
        'Train_Accuracy': train_accs,
        'Val_Loss': val_losses,
        'Val_Accuracy': val_accs
    })
    df.to_excel(os.path.join(config_dir, 'metrics.xlsx'), index=False)

    # Save config
    with open(os.path.join(config_dir, 'config.json'), 'w') as f:
        json.dump({**config, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch}, f, indent=2)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Training Loss\n{config_name}', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].axhline(y=best_val_acc, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_val_acc:.2f}%')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Training Accuracy\n{config_name}', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Results saved to {config_dir}")


# ==============================================================================
# Main Search Function (Two-Stage)
# ==============================================================================
def main_search():
    """Run two-stage hyperparameter search.

    Stage 1: Quick screening with STAGE1_EPOCHS epochs for all configurations
    Stage 2: Full training with STAGE2_EPOCHS epochs for top TOP_K configurations
    """
    print("=" * 80)
    print("ResNet CIFAR10 Two-Stage Hyperparameter Search with Fitted Device")
    print("=" * 80)
    print(f"\nDevice Configuration:")
    print(f"  Input layer: FloatingPointRPUConfig")
    print(f"  Hidden layers: PiecewiseStepDevice (fitted, dw_min={device_config['dw_min']})")
    print(f"  Final FC layer: FloatingPointRPUConfig")
    print(f"\nTwo-Stage Search Strategy:")
    print(f"  Stage 1: {STAGE1_EPOCHS} epochs (quick screening)")
    print(f"  Stage 2: {STAGE2_EPOCHS} epochs (top {TOP_K} configs)")
    print(f"  Seed: {SEED}")
    print(f"\nSearch Space:")
    for key, values in SEARCH_SPACE.items():
        print(f"  {key}: {values}")

    # Generate all configurations
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_configs = len(all_configs)

    print(f"\nTotal configurations: {total_configs}")
    print("=" * 80)

    # ==========================================================================
    # STAGE 1: Quick Screening
    # ==========================================================================
    print("\n" + "=" * 80)
    print(f"STAGE 1: Quick Screening ({STAGE1_EPOCHS} epochs each)")
    print("=" * 80)

    stage1_results = []
    start_time = time.time()

    for idx, config in enumerate(all_configs):
        config_name = f"LR{config['lr']}_WD{config['weight_decay']}_BS{config['batch_size']}_{config['scheduler']}"
        print(f"\n[Stage1 {idx+1}/{total_configs}] {config_name}")

        # Load data with current batch size
        train_loader, val_loader = load_images(config['batch_size'])

        config_start = time.time()
        result = train_single_config(config, train_loader, val_loader, n_epochs=STAGE1_EPOCHS, save_results=False)
        config_time = time.time() - config_start

        result['config'] = config
        result['config_name'] = config_name
        result['time_seconds'] = config_time
        stage1_results.append(result)

        print(f"  Result: Val Acc={result['best_val_acc']:.2f}% (Time: {config_time/60:.1f} min)")

        # Memory cleanup after each config
        del train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    stage1_time = (time.time() - start_time) / 60

    # Sort by best validation accuracy
    stage1_results.sort(key=lambda x: x['best_val_acc'], reverse=True)

    # Print Stage 1 results
    print("\n" + "=" * 80)
    print(f"STAGE 1 COMPLETE - Top {TOP_K} Configurations")
    print("=" * 80)
    for i, r in enumerate(stage1_results[:TOP_K]):
        print(f"{i+1}. {r['config_name']}: {r['best_val_acc']:.2f}%")

    # Save Stage 1 results
    stage1_df = pd.DataFrame([
        {
            'rank': i+1,
            'config_name': r['config_name'],
            'lr': r['config']['lr'],
            'weight_decay': r['config']['weight_decay'],
            'best_val_acc': r['best_val_acc'],
            'time_minutes': r['time_seconds'] / 60
        }
        for i, r in enumerate(stage1_results)
    ])
    stage1_df.to_excel(os.path.join(RESULTS_DIR, 'stage1_screening_results.xlsx'), index=False)

    # ==========================================================================
    # STAGE 2: Full Training for Top Configurations
    # ==========================================================================
    print("\n" + "=" * 80)
    print(f"STAGE 2: Full Training ({STAGE2_EPOCHS} epochs for top {TOP_K})")
    print("=" * 80)

    stage2_results = []
    top_configs = stage1_results[:TOP_K]
    stage2_start = time.time()

    for idx, stage1_result in enumerate(top_configs):
        config = stage1_result['config']
        config_name = stage1_result['config_name']
        print(f"\n[Stage2 {idx+1}/{TOP_K}] {config_name}")

        train_loader, val_loader = load_images(config['batch_size'])

        config_start = time.time()
        result = train_single_config(
            config, train_loader, val_loader,
            n_epochs=STAGE2_EPOCHS,
            save_results=True,
            config_name=f"TOP{idx+1}_{config_name}"
        )
        config_time = time.time() - config_start

        result['config'] = config
        result['config_name'] = config_name
        result['stage1_acc'] = stage1_result['best_val_acc']
        result['time_seconds'] = config_time
        stage2_results.append(result)

        print(f"  Final: Val Acc={result['final_val_acc']:.2f}%, Best={result['best_val_acc']:.2f}% (epoch {result['best_epoch']})")
        print(f"  Time: {config_time/60:.1f} minutes")

        # Memory cleanup after each config
        del train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    stage2_time = (time.time() - stage2_start) / 60
    total_time = (time.time() - start_time) / 60

    # Sort Stage 2 results
    stage2_results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    best_result = stage2_results[0]

    # ==========================================================================
    # Save Final Results
    # ==========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    for i, r in enumerate(stage2_results):
        print(f"{i+1}. {r['config_name']}")
        print(f"   Stage1: {r['stage1_acc']:.2f}% -> Stage2: {r['best_val_acc']:.2f}%")

    # Save summary
    summary = {
        'optimal_config': best_result['config'],
        'optimal_results': {
            'best_val_acc': best_result['best_val_acc'],
            'best_epoch': best_result['best_epoch'],
            'final_val_acc': best_result['final_val_acc'],
        },
        'search_strategy': {
            'stage1_epochs': STAGE1_EPOCHS,
            'stage2_epochs': STAGE2_EPOCHS,
            'top_k': TOP_K,
        },
        'timing': {
            'stage1_minutes': stage1_time,
            'stage2_minutes': stage2_time,
            'total_minutes': total_time,
            'total_hours': total_time / 60,
        },
        'total_configs_screened': total_configs,
        'device_config': {
            'input_layer': 'FloatingPointRPUConfig',
            'hidden_layers': 'PiecewiseStepDevice (fitted)',
            'output_layer': 'FloatingPointRPUConfig',
            'dw_min': device_config['dw_min'],
        },
        'stage2_results': [
            {
                'rank': i+1,
                'config_name': r['config_name'],
                'config': r['config'],
                'stage1_acc': r['stage1_acc'],
                'best_val_acc': r['best_val_acc'],
                'best_epoch': r['best_epoch'],
                'final_val_acc': r['final_val_acc'],
            }
            for i, r in enumerate(stage2_results)
        ]
    }

    with open(os.path.join(RESULTS_DIR, 'search_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save Stage 2 results to Excel
    stage2_df = pd.DataFrame([
        {
            'rank': i+1,
            'config_name': r['config_name'],
            'lr': r['config']['lr'],
            'weight_decay': r['config']['weight_decay'],
            'stage1_acc': r['stage1_acc'],
            'best_val_acc': r['best_val_acc'],
            'best_epoch': r['best_epoch'],
            'final_val_acc': r['final_val_acc'],
            'time_minutes': r['time_seconds'] / 60
        }
        for i, r in enumerate(stage2_results)
    ])
    stage2_df.to_excel(os.path.join(RESULTS_DIR, 'stage2_final_results.xlsx'), index=False)

    print("\n" + "=" * 80)
    print("TWO-STAGE SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nOptimal Configuration: {best_result['config_name']}")
    print(f"  LR: {best_result['config']['lr']}")
    print(f"  Weight Decay: {best_result['config']['weight_decay']}")
    print(f"  Best Accuracy: {best_result['best_val_acc']:.2f}% (epoch {best_result['best_epoch']})")
    print(f"\nTiming:")
    print(f"  Stage 1 ({total_configs} configs x {STAGE1_EPOCHS} epochs): {stage1_time:.1f} min")
    print(f"  Stage 2 ({TOP_K} configs x {STAGE2_EPOCHS} epochs): {stage2_time:.1f} min")
    print(f"  Total: {total_time:.1f} min ({total_time/60:.1f} hours)")
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("=" * 80)


def quick_test():
    """Quick test with a single configuration to verify setup."""
    print("=" * 80)
    print("Quick Test - Single Configuration")
    print("=" * 80)

    config = {
        'lr': 0.05,
        'weight_decay': 5e-4,
        'batch_size': 128,
        'scheduler': 'cosine'
    }

    print(f"\nTest config: {config}")
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

    print("\nLoading data...")
    train_loader, val_loader = load_images(config['batch_size'])

    print("\nStarting training (10 epochs for test)...")

    result = train_single_config(
        config, train_loader, val_loader,
        n_epochs=10,
        save_results=True,
        config_name="TEST_RUN"
    )

    print(f"\nTest Results:")
    print(f"  Final Val Accuracy: {result['final_val_acc']:.2f}%")
    print(f"  Best Val Accuracy: {result['best_val_acc']:.2f}% (epoch {result['best_epoch']})")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        quick_test()
    else:
        main_search()
