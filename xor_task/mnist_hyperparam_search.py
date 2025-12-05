#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST Hyperparameter Search for Fitted Device
Based on 03_mnist_training.py example

Searches for optimal:
- Learning rate
- Hidden layer sizes
- Batch size
"""

import json
import os
import time
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, PiecewiseStepDevice
from aihwkit.simulator.rpu_base import cuda

# Check device - Use GPU if available
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

# Path where the datasets will be stored
PATH_DATASET = os.path.join("data", "DATASET")

# Network definition
INPUT_SIZE = 784
OUTPUT_SIZE = 10

# Load fitted device config
with open('xor_device_config.json', 'r') as f:
    device_config = json.load(f)

# Create fitted device
device_params = {k: v for k, v in device_config.items() if not k.startswith('_')}
fitted_device = PiecewiseStepDevice(**device_params)
RPU_CONFIG = SingleRPUConfig(device=fitted_device)

# Search space - Only Learning Rate (Layer config and batch size fixed)
LR_VALUES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
HIDDEN_SIZES = [256, 128]  # Fixed: same as original MNIST example
BATCH_SIZE = 64            # Fixed

# Training parameters
NUM_EPOCHS = 15   # More epochs for better evaluation
NUM_TRIALS = 3    # Trials per config


def load_images(batch_size):
    """Load MNIST images."""
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)

    train_data = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_data, validation_data


def create_analog_network():
    """Create analog MLP network with fixed architecture."""
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


def train_single_config(lr, num_epochs, trial_seed):
    """Train a single configuration and return results."""
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)

    # Load data
    train_loader, val_loader = load_images(BATCH_SIZE)

    # Create model
    model = create_analog_network()

    # Optimizer and scheduler
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    criterion = nn.NLLLoss()

    # Training loop
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

    return {
        'final_val_acc': val_accs[-1],
        'final_val_loss': val_losses[-1],
        'best_val_acc': max(val_accs),
        'train_losses': train_losses,
        'val_accs': val_accs,
    }


def evaluate_config(lr):
    """Evaluate a configuration with multiple trials."""
    results = []

    for trial in range(NUM_TRIALS):
        result = train_single_config(lr, NUM_EPOCHS, 42 + trial)
        results.append(result)

    avg_final_acc = np.mean([r['final_val_acc'] for r in results])
    std_final_acc = np.std([r['final_val_acc'] for r in results])
    avg_best_acc = np.mean([r['best_val_acc'] for r in results])

    return {
        'avg_final_acc': avg_final_acc,
        'std_final_acc': std_final_acc,
        'avg_best_acc': avg_best_acc,
        'all_results': results,
    }


def main():
    print("=" * 70)
    print("MNIST Hyperparameter Search for Fitted Device")
    print("=" * 70)
    print(f"Device: PiecewiseStepDevice (dw_min={device_config['dw_min']})")
    print(f"Fixed config:")
    print(f"  Hidden sizes: {HIDDEN_SIZES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"Search space:")
    print(f"  LR: {LR_VALUES}")
    print(f"  Epochs per config: {NUM_EPOCHS}")
    print(f"  Trials per config: {NUM_TRIALS}")

    total_configs = len(LR_VALUES)
    print(f"  Total configs: {total_configs}")
    print("=" * 70)

    # Download dataset first
    print("\nDownloading MNIST dataset...")
    _ = load_images(BATCH_SIZE)
    print("Dataset ready.\n")

    all_results = []
    best_config = None
    best_acc = 0

    config_idx = 0
    start_time = time.time()

    for lr in LR_VALUES:
        config_idx += 1
        print(f"\n[{config_idx}/{total_configs}] LR={lr}")

        config_start = time.time()
        result = evaluate_config(lr)
        config_time = time.time() - config_start

        result['lr'] = lr
        result['time_seconds'] = config_time
        all_results.append(result)

        print(f"  Accuracy: {result['avg_final_acc']:.2f}% ± {result['std_final_acc']:.2f}%")
        print(f"  Best Acc: {result['avg_best_acc']:.2f}%")
        print(f"  Time: {config_time:.1f}s")

        if result['avg_best_acc'] > best_acc:
            best_acc = result['avg_best_acc']
            best_config = result.copy()

    total_time = (time.time() - start_time) / 60

    # Sort by best accuracy
    all_results.sort(key=lambda x: x['avg_best_acc'], reverse=True)

    # Print all configs
    print("\n" + "=" * 70)
    print("ALL CONFIGURATIONS (sorted by accuracy)")
    print("=" * 70)
    for i, r in enumerate(all_results):
        print(f"{i+1}. LR={r['lr']}")
        print(f"   Accuracy: {r['avg_final_acc']:.2f}% ± {r['std_final_acc']:.2f}%, Best: {r['avg_best_acc']:.2f}%")

    # Save results
    output = {
        'optimal_config': {
            'lr': best_config['lr'],
            'hidden_sizes': HIDDEN_SIZES,
            'batch_size': BATCH_SIZE,
            'input_size': INPUT_SIZE,
            'output_size': OUTPUT_SIZE,
            'activation': 'Sigmoid',
            'loss': 'NLLLoss',
            'scheduler': 'StepLR(step_size=5, gamma=0.5)',
        },
        'optimal_results': {
            'avg_final_acc': best_config['avg_final_acc'],
            'std_final_acc': best_config['std_final_acc'],
            'avg_best_acc': best_config['avg_best_acc'],
        },
        'device_info': f"Fitted PiecewiseStepDevice (dw_min={device_config['dw_min']})",
        'search_time_minutes': total_time,
        'search_epochs': NUM_EPOCHS,
        'search_trials': NUM_TRIALS,
        'all_configs': [
            {
                'lr': r['lr'],
                'avg_best_acc': r['avg_best_acc'],
                'avg_final_acc': r['avg_final_acc'],
                'std_final_acc': r['std_final_acc'],
            }
            for r in all_results
        ],
    }

    with open('mnist_optimal_hyperparams_fitted.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: mnist_optimal_hyperparams_fitted.json")

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"LR: {best_config['lr']}")
    print(f"Hidden sizes: {HIDDEN_SIZES} (fixed)")
    print(f"Batch size: {BATCH_SIZE} (fixed)")
    print(f"Best Accuracy: {best_config['avg_best_acc']:.2f}%")
    print(f"Total search time: {total_time:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
