#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST 24-Hour Hyperparameter Search for Fitted Device
Searches: Learning Rate, Scheduler (step_size, gamma), Weight Initialization
Saves optimal results with accuracy/loss plots and weight distribution every 10 epochs.
"""

import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import torch.nn.init as init

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

# Path settings
PATH_DATASET = os.path.join("data", "DATASET")
RESULTS_DIR = "mnist_search_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Network definition (fixed)
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training parameters (fixed)
EPOCHS = 50
BATCH_SIZE = 128

# Search space (~144 configs for 24 hours)
LR_VALUES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
STEP_SIZE_VALUES = [5, 10, 15]
GAMMA_VALUES = [0.3, 0.5, 0.7]
INIT_TYPES = ["normal", "xavier"]

# Load fitted device config
with open('xor_device_config.json', 'r') as f:
    device_config = json.load(f)

device_params = {k: v for k, v in device_config.items() if not k.startswith('_')}
fitted_device = PiecewiseStepDevice(**device_params)
RPU_CONFIG = SingleRPUConfig(device=fitted_device)


def load_images():
    """Load MNIST images."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_data, validation_data


def create_analog_network():
    """Create analog MLP network with digital bias (default in MappingParameter)."""
    model = AnalogSequential(
        AnalogLinear(INPUT_SIZE, HIDDEN_SIZES[0], bias=True, rpu_config=RPU_CONFIG),
        nn.Sigmoid(),
        AnalogLinear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], bias=True, rpu_config=RPU_CONFIG),
        nn.Sigmoid(),
        AnalogLinear(HIDDEN_SIZES[1], OUTPUT_SIZE, bias=True, rpu_config=RPU_CONFIG),
        nn.LogSoftmax(dim=1),
    )
    if USE_CUDA:
        model.cuda()
    return model


def custom_weight_init(model, init_type):
    """Apply custom weight initialization."""
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            weight, bias = module.get_weights()
            if init_type == "normal":
                init.normal_(weight, mean=0.0, std=0.05)
            elif init_type == "xavier":
                init.xavier_uniform_(weight)
            elif init_type == "kaiming":
                init.kaiming_uniform_(weight, nonlinearity='sigmoid')
            module.set_weights(weight, bias)


def get_weight_distribution(model):
    """Extract weight distribution from model."""
    all_weights = []
    layer_weights = {}

    for idx, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, AnalogLinear):
            weight, _ = module.get_weights()
            weight_np = weight.detach().cpu().numpy().flatten()
            all_weights.extend(weight_np)
            layer_weights[f"layer_{idx}"] = weight_np

    return np.array(all_weights), layer_weights


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


def save_training_results(config_name, train_losses, train_accs, val_losses, val_accs,
                          weight_distributions, save_dir):
    """Save training results to Excel and PNG."""
    # Create directory for this config
    config_dir = os.path.join(save_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)

    # 1. Save accuracy/loss to Excel
    epochs = list(range(1, len(train_losses) + 1))
    df_metrics = pd.DataFrame({
        'Epoch': epochs,
        'Train_Loss': train_losses,
        'Train_Accuracy': train_accs,
        'Val_Loss': val_losses,
        'Val_Accuracy': val_accs
    })
    df_metrics.to_excel(os.path.join(config_dir, 'metrics.xlsx'), index=False)

    # 2. Save accuracy/loss plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Training Loss\n{config_name}', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Training Accuracy\n{config_name}', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config_dir, 'metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Save weight distributions to Excel and PNG
    weight_epochs = sorted(weight_distributions.keys())

    # Excel: weight statistics per epoch
    weight_stats = []
    for epoch in weight_epochs:
        weights = weight_distributions[epoch]['all']
        weight_stats.append({
            'Epoch': epoch,
            'Mean': np.mean(weights),
            'Std': np.std(weights),
            'Min': np.min(weights),
            'Max': np.max(weights),
            'Median': np.median(weights)
        })
    df_weights = pd.DataFrame(weight_stats)
    df_weights.to_excel(os.path.join(config_dir, 'weight_stats.xlsx'), index=False)

    # PNG: Overlaid weight distribution histogram (Epoch 1 vs Epoch 50)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for different epochs
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(weight_epochs)))

    # Get first and last epoch for main comparison
    first_epoch = min(weight_epochs)
    last_epoch = max(weight_epochs)

    # Plot all epochs with varying transparency
    for idx, epoch in enumerate(weight_epochs):
        weights = weight_distributions[epoch]['all']

        if epoch == first_epoch:
            # First epoch: dashed gray line
            ax.hist(weights, bins=100, density=True, alpha=0.5,
                   color='gray', linestyle='--', histtype='step',
                   linewidth=2, label=f'Epoch {epoch} (Initial)')
        elif epoch == last_epoch:
            # Last epoch: solid colored line
            ax.hist(weights, bins=100, density=True, alpha=0.7,
                   color='blue', histtype='stepfilled',
                   linewidth=2, label=f'Epoch {epoch} (Final)')
        else:
            # Middle epochs: lighter colors
            ax.hist(weights, bins=100, density=True, alpha=0.3,
                   color=colors[idx], histtype='step',
                   linewidth=1, label=f'Epoch {epoch}')

    # Set x-axis range to show saturation walls
    ax.set_xlim(-1.2, 1.2)

    # Add vertical lines at saturation boundaries
    ax.axvline(x=-1, color='red', linestyle=':', alpha=0.7, linewidth=1.5, label='Saturation boundary')
    ax.axvline(x=1, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)

    # Use log scale for y-axis to better show distribution tails
    ax.set_yscale('log')

    ax.set_xlabel('Weight Value', fontsize=12)
    ax.set_ylabel('Density (log scale)', fontsize=12)
    ax.set_title(f'Weight Distribution Evolution\n{config_name}', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    first_weights = weight_distributions[first_epoch]['all']
    last_weights = weight_distributions[last_epoch]['all']
    stats_text = (f'Initial (Epoch {first_epoch}):\n'
                  f'  Mean: {np.mean(first_weights):.4f}, Std: {np.std(first_weights):.4f}\n'
                  f'Final (Epoch {last_epoch}):\n'
                  f'  Mean: {np.mean(last_weights):.4f}, Std: {np.std(last_weights):.4f}\n'
                  f'Saturation: weights at ±1 boundary')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(config_dir, 'weight_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save detailed weight distribution data to Excel
    # Create histogram data for each epoch
    bins = np.linspace(-1.2, 1.2, 101)  # 100 bins
    bin_centers = (bins[:-1] + bins[1:]) / 2

    hist_data = {'Bin_Center': bin_centers}
    for epoch in weight_epochs:
        weights = weight_distributions[epoch]['all']
        hist, _ = np.histogram(weights, bins=bins, density=True)
        hist_data[f'Epoch_{epoch}_Density'] = hist

    df_hist = pd.DataFrame(hist_data)
    df_hist.to_excel(os.path.join(config_dir, 'weight_distribution.xlsx'), index=False)

    print(f"  Saved results to {config_dir}")


def train_single_config(lr, step_size, gamma, init_type, train_loader, val_loader,
                        save_results=False, config_name=None, save_dir=None):
    """Train a single configuration."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = create_analog_network()
    custom_weight_init(model, init_type)

    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.NLLLoss()

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    weight_distributions = {}

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save weight distribution every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            all_weights, layer_weights = get_weight_distribution(model)
            weight_distributions[epoch + 1] = {
                'all': all_weights,
                'layers': layer_weights
            }

        scheduler.step()

    # Save results if requested
    if save_results and config_name and save_dir:
        save_training_results(config_name, train_losses, train_accs, val_losses, val_accs,
                             weight_distributions, save_dir)

    return {
        'final_val_acc': val_accs[-1],
        'best_val_acc': max(val_accs),
        'final_val_loss': val_losses[-1],
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'weight_distributions': weight_distributions
    }


def test_save_functionality():
    """Test that Excel and PNG saving works correctly."""
    print("=" * 70)
    print("Testing Save Functionality")
    print("=" * 70)

    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, val_loader = load_images()

    # Test with one configuration
    test_config = {
        'lr': 0.05,
        'step_size': 10,
        'gamma': 0.5,
        'init_type': 'normal'
    }

    config_name = f"TEST_LR{test_config['lr']}_Step{test_config['step_size']}_Gamma{test_config['gamma']}_Init{test_config['init_type']}"

    print(f"\nTraining test config: {config_name}")
    print("This will train for 50 epochs and save results...")

    start_time = time.time()
    result = train_single_config(
        test_config['lr'],
        test_config['step_size'],
        test_config['gamma'],
        test_config['init_type'],
        train_loader,
        val_loader,
        save_results=True,
        config_name=config_name,
        save_dir=RESULTS_DIR
    )
    elapsed_time = time.time() - start_time

    print(f"\nTest Results:")
    print(f"  Final Val Accuracy: {result['final_val_acc']:.2f}%")
    print(f"  Best Val Accuracy: {result['best_val_acc']:.2f}%")
    print(f"  Training Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

    # Verify files exist
    test_dir = os.path.join(RESULTS_DIR, config_name)
    files_to_check = ['metrics.xlsx', 'metrics.png', 'weight_stats.xlsx', 'weight_distribution.png']

    print(f"\nVerifying saved files in {test_dir}:")
    all_exist = True
    for f in files_to_check:
        path = os.path.join(test_dir, f)
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  {f}: {status}")
        if not exists:
            all_exist = False

    if all_exist:
        print("\nAll files saved successfully!")
    else:
        print("\nERROR: Some files are missing!")

    return elapsed_time, all_exist


def main_search():
    """Run the full hyperparameter search."""
    print("=" * 70)
    print("MNIST 24-Hour Hyperparameter Search")
    print("=" * 70)
    print(f"Device: Fitted PiecewiseStepDevice (dw_min={device_config['dw_min']})")
    print(f"\nFixed settings:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Hidden sizes: {HIDDEN_SIZES}")
    print(f"\nSearch space:")
    print(f"  LR: {LR_VALUES}")
    print(f"  Step size: {STEP_SIZE_VALUES}")
    print(f"  Gamma: {GAMMA_VALUES}")
    print(f"  Init type: {INIT_TYPES}")

    total_configs = len(LR_VALUES) * len(STEP_SIZE_VALUES) * len(GAMMA_VALUES) * len(INIT_TYPES)
    print(f"\nTotal configurations: {total_configs}")
    print("=" * 70)

    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, val_loader = load_images()

    all_results = []
    best_config = None
    best_acc = 0

    config_idx = 0
    start_time = time.time()

    for lr in LR_VALUES:
        for step_size in STEP_SIZE_VALUES:
            for gamma in GAMMA_VALUES:
                for init_type in INIT_TYPES:
                    config_idx += 1
                    config_name = f"LR{lr}_Step{step_size}_Gamma{gamma}_Init{init_type}"

                    print(f"\n[{config_idx}/{total_configs}] {config_name}")

                    config_start = time.time()
                    result = train_single_config(
                        lr, step_size, gamma, init_type,
                        train_loader, val_loader,
                        save_results=False  # Don't save during search
                    )
                    config_time = time.time() - config_start

                    result['lr'] = lr
                    result['step_size'] = step_size
                    result['gamma'] = gamma
                    result['init_type'] = init_type
                    result['config_name'] = config_name
                    result['time_seconds'] = config_time
                    all_results.append(result)

                    print(f"  Val Accuracy: {result['final_val_acc']:.2f}% (Best: {result['best_val_acc']:.2f}%)")
                    print(f"  Time: {config_time:.1f}s")

                    if result['best_val_acc'] > best_acc:
                        best_acc = result['best_val_acc']
                        best_config = result.copy()
                        print(f"  *** NEW BEST ***")

    total_time = (time.time() - start_time) / 60

    # Sort results
    all_results.sort(key=lambda x: x['best_val_acc'], reverse=True)

    # Print top 10
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 70)
    for i, r in enumerate(all_results[:10]):
        print(f"{i+1}. {r['config_name']}")
        print(f"   Best Acc: {r['best_val_acc']:.2f}%, Final Acc: {r['final_val_acc']:.2f}%")

    # Train and save best config with full results
    print("\n" + "=" * 70)
    print("Training OPTIMAL configuration with full saves...")
    print("=" * 70)

    optimal_result = train_single_config(
        best_config['lr'],
        best_config['step_size'],
        best_config['gamma'],
        best_config['init_type'],
        train_loader, val_loader,
        save_results=True,
        config_name=f"OPTIMAL_{best_config['config_name']}",
        save_dir=RESULTS_DIR
    )

    # Save search summary
    summary = {
        'optimal_config': {
            'lr': best_config['lr'],
            'step_size': best_config['step_size'],
            'gamma': best_config['gamma'],
            'init_type': best_config['init_type'],
        },
        'optimal_results': {
            'best_val_acc': optimal_result['best_val_acc'],
            'final_val_acc': optimal_result['final_val_acc'],
        },
        'search_time_minutes': total_time,
        'total_configs_tested': total_configs,
        'top_10': [
            {
                'config_name': r['config_name'],
                'best_val_acc': r['best_val_acc'],
                'final_val_acc': r['final_val_acc'],
            }
            for r in all_results[:10]
        ]
    }

    with open(os.path.join(RESULTS_DIR, 'search_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)
    print(f"Optimal Config: {best_config['config_name']}")
    print(f"Best Accuracy: {optimal_result['best_val_acc']:.2f}%")
    print(f"Total Search Time: {total_time:.1f} minutes")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode: verify saving functionality
        test_save_functionality()
    else:
        # Full search mode
        main_search()
