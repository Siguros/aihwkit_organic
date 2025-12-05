#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare XOR Task Performance:
1. IdealizedPresetDevice (ideal baseline)
2. PiecewiseStepDevice with uniform piecewise (same states as fitted)
3. Fitted PiecewiseStepDevice (realistic device model)

Based on aihwkit example 01_simple_layer.py style.
Uses ReLU activation + MSE loss for proper analog training.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch.nn import ReLU, Sequential
from torch.nn.functional import mse_loss

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice, PiecewiseStepDevice
from aihwkit.simulator.presets.devices import IdealizedPresetDevice

# XOR Dataset (3rd input is bias=1.0)
x_train = Tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
y_train = Tensor([[0.0], [1.0], [1.0], [0.0]])


def train_xor(rpu_config, num_epochs=2000, lr=0.1, num_trials=10):
    """Train XOR task and return metrics."""
    all_loss_histories = []
    all_acc_histories = []
    convergence_epochs = []

    for trial in range(num_trials):
        torch.manual_seed(42 + trial)
        np.random.seed(42 + trial)

        # Define model: 3x3x1 structure, bias=False (3rd input acts as bias)
        model = Sequential(
            AnalogLinear(3, 3, bias=False, rpu_config=rpu_config),
            ReLU(),
            AnalogLinear(3, 1, bias=False, rpu_config=rpu_config),
        )

        # Define analog-aware optimizer
        opt = AnalogSGD(model.parameters(), lr=lr)
        opt.regroup_param_groups(model)

        loss_history = []
        acc_history = []
        converged = None

        for epoch in range(num_epochs):
            opt.zero_grad()
            pred = model(x_train)
            loss = mse_loss(pred, y_train)
            loss.backward()
            opt.step()

            loss_history.append(loss.item())
            preds = (pred > 0.5).float()
            acc = (preds == y_train).float().mean().item() * 100
            acc_history.append(acc)

            if acc == 100 and converged is None:
                converged = epoch + 1

        all_loss_histories.append(loss_history)
        all_acc_histories.append(acc_history)
        convergence_epochs.append(converged if converged else num_epochs)

    return {
        'loss_histories': all_loss_histories,
        'acc_histories': all_acc_histories,
        'convergence_epochs': convergence_epochs,
    }


# =============================================================================
# Setup Devices
# =============================================================================
print("=" * 70)
print("XOR Task: Device Comparison")
print("=" * 70)

# Load fitted config
with open('optimized_device_config.json', 'r') as f:
    config = json.load(f)

# Use the fitted parameters directly from config
FITTED_DW_MIN = config['dw_min']
FITTED_NOISE_STD = 0.0  # No noise to isolate piecewise parameter effect

# 1. IdealizedPresetDevice (ideal baseline)
ideal_device = IdealizedPresetDevice()
ideal_config = SingleRPUConfig(device=ideal_device)
print(f"\n1. IdealizedPresetDevice (dw_min={ideal_device.dw_min})")

# 2. PiecewiseStepDevice with UNIFORM piecewise (same dw_min as fitted)
uniform_piecewise = [1.0] * 10
uniform_device = PiecewiseStepDevice(
    w_min=-1, w_max=1,
    w_min_dtod=0.0, w_max_dtod=0.0,
    dw_min=FITTED_DW_MIN,
    dw_min_std=0.0, dw_min_dtod=0.0,
    up_down=0.0,
    up_down_dtod=0.0,
    write_noise_std=FITTED_NOISE_STD,
    piecewise_up=uniform_piecewise,
    piecewise_down=uniform_piecewise,
)
uniform_config = SingleRPUConfig(device=uniform_device)
print(f"2. Uniform PiecewiseStepDevice (dw_min={FITTED_DW_MIN})")

# 3. Fitted PiecewiseStepDevice (realistic device)
fitted_device = PiecewiseStepDevice(
    w_min=config['w_min'],
    w_max=config['w_max'],
    w_min_dtod=0.0, w_max_dtod=0.0,
    dw_min=FITTED_DW_MIN,
    dw_min_std=0.0, dw_min_dtod=0.0,
    up_down=config['up_down'],
    up_down_dtod=0.0,
    write_noise_std=FITTED_NOISE_STD,
    piecewise_up=config['piecewise_up'],
    piecewise_down=config['piecewise_down'],
)
fitted_config = SingleRPUConfig(device=fitted_device)
print(f"3. Fitted PiecewiseStepDevice (dw_min={FITTED_DW_MIN}, realistic)")

# =============================================================================
# Train All Models
# =============================================================================
NUM_TRIALS = 30
NUM_EPOCHS = 2000
LR = 0.1  # Same as 01_simple_layer.py

devices = [
    ('Ideal Device', ideal_config, 'tab:blue'),
    ('Linear Device', uniform_config, 'tab:green'),
    ('Fitted Device', fitted_config, 'tab:red'),
]

print(f"\nTraining: {NUM_TRIALS} trials, {NUM_EPOCHS} epochs, lr={LR}")
print("-" * 70)

all_results = []
for name, rpu_config, color in devices:
    print(f"Training {name}...")
    results = train_xor(rpu_config, NUM_EPOCHS, LR, NUM_TRIALS)
    results['name'] = name
    results['color'] = color
    results['lr'] = LR
    all_results.append(results)

# =============================================================================
# Analyze Results
# =============================================================================
print("\n" + "=" * 70)
print("Results Summary")
print("=" * 70)

for results in all_results:
    conv_epochs = results['convergence_epochs']
    successful = [e for e in conv_epochs if e < NUM_EPOCHS]
    success_rate = len(successful) / NUM_TRIALS * 100
    results['success_rate'] = success_rate
    results['successful_epochs'] = successful

    name = results['name']
    print(f"\n{name}:")
    print(f"  Success Rate: {success_rate:.0f}% ({len(successful)}/{NUM_TRIALS})")
    if successful:
        print(f"  Avg convergence: {np.mean(successful):.1f} +/- {np.std(successful):.1f} epochs")

# =============================================================================
# Create Comparison Plot
# =============================================================================
fig = plt.figure(figsize=(16, 12))

# Layout: 2 rows x 3 cols for curves + 1 row for bar charts
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.3)

# Row 1: Accuracy curves for each device (Mean only)
for idx, results in enumerate(all_results):
    ax = fig.add_subplot(gs[0, idx])
    color = results['color']

    # Calculate mean
    acc_array = np.array(results['acc_histories'])
    mean_acc = np.mean(acc_array, axis=0)
    epochs = np.arange(len(mean_acc))

    # Plot mean only
    ax.plot(epochs, mean_acc, color=color, linewidth=2)
    ax.axhline(y=100, color='black', linestyle='--', alpha=0.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(results['name'], fontsize=11, fontweight='bold')
    ax.set_ylim([20, 105])
    ax.grid(True, alpha=0.3)

# Row 2: Loss curves (Mean only)
for idx, results in enumerate(all_results):
    ax = fig.add_subplot(gs[1, idx])
    color = results['color']

    # Calculate mean
    loss_array = np.array(results['loss_histories'])
    mean_loss = np.mean(loss_array, axis=0)
    epochs = np.arange(len(mean_loss))

    # Plot mean only
    ax.plot(epochs, mean_loss, color=color, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training Loss', fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

# Row 3 Left: Success Rate Bar Chart
ax1 = fig.add_subplot(gs[2, 0])
names = [r['name'] for r in all_results]
success_rates = [r['success_rate'] for r in all_results]
colors = [r['color'] for r in all_results]

bars = ax1.bar(range(len(names)), success_rates, color=colors, edgecolor='black', alpha=0.8)
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, fontsize=9)
ax1.set_ylabel('Success Rate (%)')
ax1.set_title('XOR Task Success Rate\n(Achieving 100% Accuracy)', fontweight='bold')
ax1.set_ylim([0, 110])
ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5)
for bar, rate in zip(bars, success_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{rate:.0f}%', ha='center', fontweight='bold', fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# Row 3 Middle: Convergence Box Plot
ax2 = fig.add_subplot(gs[2, 1])
conv_data = [r['convergence_epochs'] for r in all_results]
bp = ax2.boxplot(conv_data, tick_labels=names, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax2.set_ylabel('Epochs to 100%')
ax2.set_title('Convergence Speed\n(Lower is Better)', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Row 3 Right: Piecewise Parameters Comparison
ax3 = fig.add_subplot(gs[2, 2])
x_seg = np.arange(10)
width = 0.2

# Plot piecewise parameters
ax3.bar(x_seg - width, uniform_piecewise, width, label='Uniform UP/DOWN', color='tab:green', alpha=0.7)
ax3.bar(x_seg, config['piecewise_up'], width, label='Fitted UP', color='tab:red', alpha=0.7)
ax3.bar(x_seg + width, config['piecewise_down'], width, label='Fitted DOWN', color='tab:orange', alpha=0.7)

ax3.set_xlabel('Segment Index')
ax3.set_ylabel('Piecewise Value')
ax3.set_title('Piecewise Parameters\n(Step Size per Segment)', fontweight='bold')
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=1.0, color='black', linestyle=':', alpha=0.5)
ax3.set_xticks(x_seg)

# Main title
fig.suptitle('XOR Task Performance Comparison (ReLU + MSE Loss)',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('device_comparison_xor.png', dpi=150, bbox_inches='tight')
print("\nSaved: device_comparison_xor.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"""
Device Performance Summary:
---------------------------
1. Ideal Device:   {all_results[0]['success_rate']:.0f}% success rate
   - IdealizedPresetDevice (dw_min={ideal_device.dw_min})

2. Linear Device:  {all_results[1]['success_rate']:.0f}% success rate
   - Uniform piecewise (dw_min={FITTED_DW_MIN})

3. Fitted Device:  {all_results[2]['success_rate']:.0f}% success rate
   - Realistic device model based on actual measurement data
   - Non-uniform piecewise parameters

Training Configuration:
- Network: 3x3x1 (input with bias -> hidden -> output)
- Learning Rate: {LR}
- Loss Function: MSE
- Activation: ReLU (hidden), None (output)
- Bias: Input layer (3rd input = 1.0)
""")
