#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate XOR Training Plots for Multiple Devices:
1. Ideal Device (IdealizedPresetDevice)
2. Linear Device (Uniform PiecewiseStepDevice)
3. Fitted Device (Non-uniform PiecewiseStepDevice)

Each device generates:
- xor_{device}_acc_loss.png
- xor_{device}_output_distribution.png
- xor_{device}_decision_boundary.png
- Corresponding xlsx files

Uses optimal hyperparameters from xor_optimal_hyperparams.json
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch.nn import ReLU, Sequential
from torch.nn.functional import mse_loss
import torch.nn.init as init

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, PiecewiseStepDevice
from aihwkit.simulator.configs.devices import IdealDevice

# Global seed for reproducibility
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# XOR Dataset (3rd input is bias=1.0)
x_train = Tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
y_train = Tensor([[0.0], [1.0], [1.0], [0.0]])

# Load configs
with open('xor_device_config.json', 'r') as f:
    device_config = json.load(f)

with open('xor_optimal_hyperparams.json', 'r') as f:
    hyperparams = json.load(f)

optimal = hyperparams['optimal_config']

# Training parameters from optimal config
LR = optimal['lr']
INIT_TYPE = optimal['init_type']
INIT_STD = optimal['init_std']
NUM_TRIALS = 100
MAX_EPOCHS = optimal['max_epochs']
EARLY_STOP_PATIENCE = 50
LOSS_THRESHOLD = 0.02


def custom_weight_init(model, init_type, init_std):
    """Apply custom weight initialization."""
    for module in model.modules():
        if isinstance(module, AnalogLinear):
            weight, bias = module.get_weights()
            if init_type == "normal":
                init.normal_(weight, mean=0.0, std=init_std)
            elif init_type == "uniform":
                init.uniform_(weight, -init_std, init_std)
            module.set_weights(weight, bias)


def train_xor_with_early_stop(rpu_config, lr, init_type, init_std, max_epochs, early_stop_patience, loss_threshold, num_trials):
    """Train XOR and return results."""
    all_trial_info = []

    for trial in range(num_trials):
        torch.manual_seed(42 + trial)
        np.random.seed(42 + trial)

        model = Sequential(
            AnalogLinear(3, 3, bias=False, rpu_config=rpu_config),
            ReLU(),
            AnalogLinear(3, 1, bias=False, rpu_config=rpu_config),
        )
        custom_weight_init(model, init_type, init_std)

        initial_weights = []
        for module in model.modules():
            if isinstance(module, AnalogLinear):
                w, b = module.get_weights()
                initial_weights.append((w.clone(), b.clone() if b is not None else None))

        opt = AnalogSGD(model.parameters(), lr=lr)
        opt.regroup_param_groups(model)

        loss_history = []
        acc_history = []
        consecutive_good = 0
        converged_epoch = None
        final_weights = None

        for epoch in range(max_epochs):
            opt.zero_grad()
            pred = model(x_train)
            loss = mse_loss(pred, y_train)
            loss.backward()
            opt.step()

            current_loss = loss.item()
            loss_history.append(current_loss)
            preds = (pred > 0.5).float()
            acc = (preds == y_train).float().mean().item() * 100
            acc_history.append(acc)

            loss_ok = (loss_threshold is None) or (current_loss < loss_threshold)
            if acc == 100 and loss_ok:
                consecutive_good += 1
                if consecutive_good >= early_stop_patience:
                    converged_epoch = epoch + 1 - early_stop_patience + 1
                    final_weights = []
                    for module in model.modules():
                        if isinstance(module, AnalogLinear):
                            w, b = module.get_weights()
                            final_weights.append((w.clone(), b.clone() if b is not None else None))
                    with torch.no_grad():
                        final_pred = model(x_train).numpy()
                    break
            else:
                consecutive_good = 0

        if converged_epoch is not None:
            trial_info = {
                'trial': trial,
                'converged_epoch': converged_epoch,
                'final_loss': loss_history[-1],
                'loss_history': loss_history,
                'acc_history': acc_history,
                'final_predictions': final_pred.flatten().tolist(),
                'initial_weights': initial_weights,
                'final_weights': final_weights,
            }
            all_trial_info.append(trial_info)

        if (trial + 1) % 10 == 0:
            successful = len(all_trial_info)
            print(f"    Trial {trial + 1}/{num_trials} - Converged: {successful}")

    return all_trial_info


def generate_plots_for_device(device_name, rpu_config, output_prefix, device_info_str):
    """Generate all plots and xlsx files for a given device."""

    print(f"\n{'='*70}")
    print(f"Processing: {device_name}")
    print(f"{'='*70}")
    print(f"Device info: {device_info_str}")

    # Train trials
    print(f"\nTraining {NUM_TRIALS} trials...")
    stable_trials = train_xor_with_early_stop(
        rpu_config,
        lr=LR,
        init_type=INIT_TYPE,
        init_std=INIT_STD,
        max_epochs=MAX_EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        loss_threshold=LOSS_THRESHOLD,
        num_trials=NUM_TRIALS,
    )

    print(f"\nConverged trials: {len(stable_trials)}/{NUM_TRIALS}")

    if len(stable_trials) == 0:
        print(f"WARNING: No stable trials found for {device_name}!")
        return None

    # ============================================================
    # 1) Accuracy/Loss Plot
    # ============================================================
    print(f"\nCreating {output_prefix}_acc_loss.png...")

    max_iterations = 2000
    acc_matrix = np.full((len(stable_trials), max_iterations), np.nan)
    loss_matrix = np.full((len(stable_trials), max_iterations), np.nan)

    for i, t in enumerate(stable_trials):
        acc_hist = np.array(t['acc_history'])
        loss_hist = np.array(t['loss_history'])
        hist_len = min(len(acc_hist), max_iterations)

        acc_matrix[i, :hist_len] = acc_hist[:hist_len]
        loss_matrix[i, :hist_len] = loss_hist[:hist_len]

        # LVCF padding
        if hist_len < max_iterations:
            acc_matrix[i, hist_len:] = acc_hist[-1]
            loss_matrix[i, hist_len:] = loss_hist[-1]

    mean_acc = np.nanmean(acc_matrix, axis=0)
    std_acc = np.nanstd(acc_matrix, axis=0)
    mean_loss = np.nanmean(loss_matrix, axis=0)
    std_loss = np.nanstd(loss_matrix, axis=0)

    iterations = np.arange(max_iterations)
    n_trials = len(stable_trials)
    sem_acc = std_acc / np.sqrt(n_trials)
    sem_loss = std_loss / np.sqrt(n_trials)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(iterations, mean_acc, color='tab:blue', linewidth=2.5, label=f'Mean (n={n_trials})')
    ax1.fill_between(iterations,
                     np.maximum(mean_acc - sem_acc, 0),
                     np.minimum(mean_acc + sem_acc, 105),
                     alpha=0.3, color='tab:blue', label='±1 SEM')
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.7, linewidth=2, label='100% Target')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Training Accuracy (Stable 100% Trials)', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.set_xlim([0, max_iterations])
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    avg_conv = np.mean([t['converged_epoch'] for t in stable_trials])
    std_conv = np.std([t['converged_epoch'] for t in stable_trials])
    textstr = f'Convergence: {avg_conv:.1f} ± {std_conv:.1f} iterations'
    ax1.text(0.95, 0.05, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2 = axes[1]
    ax2.plot(iterations, mean_loss, color='tab:red', linewidth=2.5, label=f'Mean (n={n_trials})')
    ax2.fill_between(iterations,
                     np.maximum(mean_loss - sem_loss, 1e-6),
                     mean_loss + sem_loss,
                     alpha=0.3, color='tab:red', label='±1 SEM')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_title('Training Loss (Stable 100% Trials)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_xlim([0, max_iterations])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    final_loss_mean = np.mean([t['final_loss'] for t in stable_trials])
    final_loss_std = np.std([t['final_loss'] for t in stable_trials])
    textstr = f'Final Loss: {final_loss_mean:.4f} ± {final_loss_std:.4f}'
    ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle(f'XOR Training with {device_name}\n(LR={LR}, Init={INIT_TYPE}, std={INIT_STD})',
                 fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_acc_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_prefix}_acc_loss.png")

    # ============================================================
    # 2) Output Distribution Plot
    # ============================================================
    print(f"Creating {output_prefix}_output_distribution.png...")

    all_predictions = np.array([t['final_predictions'] for t in stable_trials])
    mean_predictions = np.mean(all_predictions, axis=0)
    std_predictions = np.std(all_predictions, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    input_labels = ['(0, 0)', '(0, 1)', '(1, 0)', '(1, 1)']
    x_positions = np.array([0, 1, 2, 3])
    targets = y_train.numpy().flatten()

    np.random.seed(42)
    for preds in all_predictions:
        jitter = np.random.uniform(-0.1, 0.1, len(preds))
        ax.scatter(x_positions + jitter, preds, c='lightblue', alpha=0.3, s=30)

    ax.errorbar(x_positions, mean_predictions, yerr=std_predictions,
                fmt='o', markersize=12, color='blue', capsize=5, capthick=2,
                ecolor='blue', elinewidth=2, label='Predicted (mean ± std)')

    ax.scatter(x_positions, targets, c='red', s=300, marker='*',
               zorder=10, label='Target', edgecolors='darkred', linewidths=1)

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Threshold (0.5)')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(input_labels, fontsize=12)
    ax.set_xlabel('Input (X1, X2)', fontsize=12)
    ax.set_ylabel('Output Value', fontsize=12)
    ax.set_title(f'XOR Output Distribution - {device_name}\n({len(stable_trials)} stable trials)', fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for i in range(4):
        y_offset = 0.15 if targets[i] == 0 else -0.2
        ax.annotate(f'μ={mean_predictions[i]:.3f}\nσ={std_predictions[i]:.3f}',
                    xy=(x_positions[i], mean_predictions[i]),
                    xytext=(x_positions[i] + 0.25, mean_predictions[i] + y_offset),
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_output_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_prefix}_output_distribution.png")

    # ============================================================
    # 3) Decision Boundary Plot
    # ============================================================
    print(f"Creating {output_prefix}_decision_boundary.png...")

    best_trial = min(stable_trials, key=lambda t: t['final_loss'])
    print(f"  Best trial: {best_trial['trial']} (loss={best_trial['final_loss']:.6f})")

    best_model = Sequential(
        AnalogLinear(3, 3, bias=False, rpu_config=rpu_config),
        ReLU(),
        AnalogLinear(3, 1, bias=False, rpu_config=rpu_config),
    )

    saved_weights = best_trial['final_weights']
    weight_idx = 0
    for module in best_model.modules():
        if isinstance(module, AnalogLinear):
            w, b = saved_weights[weight_idx]
            module.set_weights(w, b)
            weight_idx += 1

    best_model.eval()

    NUM_RUNS = 50
    with torch.no_grad():
        preds_sum = np.zeros(4)
        for _ in range(NUM_RUNS):
            preds_sum += best_model(x_train).numpy().flatten()
        final_predictions = preds_sum / NUM_RUNS

    fig, ax = plt.subplots(figsize=(8, 8))

    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape)]
    grid_tensor = Tensor(grid_points)

    with torch.no_grad():
        Z_sum = np.zeros(xx.shape)
        for _ in range(NUM_RUNS):
            Z_sum += best_model(grid_tensor).numpy().reshape(xx.shape)
        Z = Z_sum / NUM_RUNS

    contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21), cmap='RdYlBu_r', alpha=0.8)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Network Output', fontsize=12)

    x_data = x_train[:, :2].numpy()
    y_data = y_train.numpy().flatten()

    class_0 = x_data[y_data == 0]
    class_1 = x_data[y_data == 1]

    ax.scatter(class_0[:, 0], class_0[:, 1], c='blue', s=200, marker='o',
               edgecolors='white', linewidths=2, label='Target: 0', zorder=5)
    ax.scatter(class_1[:, 0], class_1[:, 1], c='red', s=200, marker='o',
               edgecolors='white', linewidths=2, label='Target: 1', zorder=5)

    for i, (x, y) in enumerate(x_data):
        pred_val = final_predictions[i]
        ax.annotate(f'Pred: {pred_val:.3f}',
                    xy=(x, y), xytext=(x + 0.15, y + 0.1),
                    fontsize=10, color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel('Input X1', fontsize=12)
    ax.set_ylabel('Input X2', fontsize=12)
    ax.set_title(f'XOR Decision Boundary - {device_name}\n'
                 f'Best Trial {best_trial["trial"]}, Loss={best_trial["final_loss"]:.6f}',
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_decision_boundary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_prefix}_decision_boundary.png")

    # ============================================================
    # Save xlsx files
    # ============================================================
    print(f"\nSaving xlsx files...")

    # 1) Training Accuracy/Loss
    acc_loss_data = {
        'Iteration': iterations,
        'Accuracy_Mean': mean_acc,
        'Accuracy_Std': np.clip(std_acc, 0, 100),
        'Accuracy_SEM': np.clip(sem_acc, 0, 100),
        'Loss_Mean': mean_loss,
        'Loss_Std': np.clip(std_loss, 0, 100),
        'Loss_SEM': np.clip(sem_loss, 0, 100),
    }
    df_acc_loss = pd.DataFrame(acc_loss_data)
    df_acc_loss.to_excel(f'{output_prefix}_training_acc_loss.xlsx', index=False)
    print(f"  Saved: {output_prefix}_training_acc_loss.xlsx")

    # 2) Output Distribution
    output_dist_data = {
        'Input': ['(0,0)', '(0,1)', '(1,0)', '(1,1)'],
        'Target': [0, 1, 1, 0],
        'Prediction_Mean': mean_predictions,
        'Prediction_Std': std_predictions,
    }
    df_output = pd.DataFrame(output_dist_data)
    df_output.to_excel(f'{output_prefix}_output_distribution.xlsx', index=False)
    print(f"  Saved: {output_prefix}_output_distribution.xlsx")

    # 2-1) Output Distribution Points
    np.random.seed(42)
    output_dist_points_list = []
    for trial_idx, preds in enumerate(all_predictions):
        jitter = np.random.uniform(-0.1, 0.1, len(preds))
        for input_idx, (pred, jit) in enumerate(zip(preds, jitter)):
            output_dist_points_list.append({
                'Trial': trial_idx,
                'Input_Index': input_idx,
                'Input': input_labels[input_idx],
                'X_Position': x_positions[input_idx],
                'X_Position_Jittered': x_positions[input_idx] + jit,
                'Prediction': pred,
                'Target': targets[input_idx],
            })
    df_output_points = pd.DataFrame(output_dist_points_list)
    df_output_points.to_excel(f'{output_prefix}_output_distribution_points.xlsx', index=False)
    print(f"  Saved: {output_prefix}_output_distribution_points.xlsx")

    # 3) Decision Boundary Grid
    decision_boundary_grid = {
        'X1': xx.ravel(),
        'X2': yy.ravel(),
        'Prediction': Z.ravel(),
    }
    df_decision_grid = pd.DataFrame(decision_boundary_grid)
    df_decision_grid.to_excel(f'{output_prefix}_decision_boundary_grid.xlsx', index=False)
    print(f"  Saved: {output_prefix}_decision_boundary_grid.xlsx")

    # 4) Decision Boundary Points
    xor_points_data = {
        'X1': x_data[:, 0],
        'X2': x_data[:, 1],
        'Target': y_data,
        'Prediction': final_predictions,
    }
    df_xor_points = pd.DataFrame(xor_points_data)
    df_xor_points.to_excel(f'{output_prefix}_decision_boundary_points.xlsx', index=False)
    print(f"  Saved: {output_prefix}_decision_boundary_points.xlsx")

    # 5) Configuration
    config_data = {
        'Parameter': [
            'Device Type',
            'Device Info',
            'Network Architecture',
            'Activation Function',
            'Learning Rate',
            'Weight Init Type',
            'Weight Init Std',
            'Loss Function',
            'Max Epochs',
            'Early Stop Patience',
            'Loss Threshold',
            'Num Trials',
            'Success Rate',
            'Avg Convergence Epoch',
            'Std Convergence Epoch',
            'Best Trial',
            'Best Trial Loss',
        ],
        'Value': [
            device_name,
            device_info_str,
            optimal['network'],
            optimal['activation'],
            LR,
            INIT_TYPE,
            INIT_STD,
            optimal['loss'],
            MAX_EPOCHS,
            EARLY_STOP_PATIENCE,
            LOSS_THRESHOLD,
            NUM_TRIALS,
            f"{len(stable_trials)}/{NUM_TRIALS} ({len(stable_trials)/NUM_TRIALS*100:.1f}%)",
            f"{avg_conv:.1f}",
            f"{std_conv:.1f}",
            best_trial['trial'],
            f"{best_trial['final_loss']:.6f}",
        ]
    }
    df_config = pd.DataFrame(config_data)
    df_config.to_excel(f'{output_prefix}_configuration.xlsx', index=False)
    print(f"  Saved: {output_prefix}_configuration.xlsx")

    # 6) Best Trial Weights
    with pd.ExcelWriter(f'{output_prefix}_best_trial_weights.xlsx', engine='openpyxl') as writer:
        for layer_idx, (w, b) in enumerate(best_trial['initial_weights']):
            w_np = w.numpy()
            df_w = pd.DataFrame(w_np,
                               columns=[f'Input_{j}' for j in range(w_np.shape[1])],
                               index=[f'Neuron_{i}' for i in range(w_np.shape[0])])
            df_w.to_excel(writer, sheet_name=f'Layer{layer_idx+1}_Initial')

        for layer_idx, (w, b) in enumerate(best_trial['final_weights']):
            w_np = w.numpy()
            df_w = pd.DataFrame(w_np,
                               columns=[f'Input_{j}' for j in range(w_np.shape[1])],
                               index=[f'Neuron_{i}' for i in range(w_np.shape[0])])
            df_w.to_excel(writer, sheet_name=f'Layer{layer_idx+1}_Final')

        summary_data = {
            'Info': ['Device', 'Best Trial', 'Converged Epoch', 'Final Loss',
                     'Layer 1 Shape', 'Layer 2 Shape', 'Init Type', 'Init Std'],
            'Value': [device_name, best_trial['trial'], best_trial['converged_epoch'],
                      f"{best_trial['final_loss']:.6f}",
                      f"{best_trial['initial_weights'][0][0].shape}",
                      f"{best_trial['initial_weights'][1][0].shape}",
                      INIT_TYPE, INIT_STD]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f"  Saved: {output_prefix}_best_trial_weights.xlsx")

    return {
        'device_name': device_name,
        'success_rate': len(stable_trials) / NUM_TRIALS * 100,
        'avg_convergence': avg_conv,
        'std_convergence': std_conv,
        'best_trial_loss': best_trial['final_loss'],
    }


# =============================================================================
# Main: Setup and Run for All Devices
# =============================================================================
print("=" * 70)
print("XOR Plot Generation for Multiple Devices")
print("=" * 70)
print(f"Network: {optimal['network']} with {optimal['activation']}")
print(f"LR: {LR}, Init: {INIT_TYPE}, Std: {INIT_STD}")
print(f"Trials: {NUM_TRIALS}, Max Epochs: {MAX_EPOCHS}")
print(f"Early stop: {EARLY_STOP_PATIENCE} consecutive at 100% AND loss < {LOSS_THRESHOLD}")

# Define devices
devices = []

# 1. Ideal Device
ideal_device = IdealDevice()
ideal_config = SingleRPUConfig(device=ideal_device)
devices.append({
    'name': 'Ideal Device',
    'config': ideal_config,
    'prefix': 'xor_Ideal',
    'info': 'IdealDevice (perfect floating-point, no noise)',
})

# 2. Linear Device (Uniform PiecewiseStepDevice)
uniform_piecewise = [1.0] * 10
linear_device = PiecewiseStepDevice(
    w_min=-1, w_max=1,
    w_min_dtod=0.0, w_max_dtod=0.0,
    dw_min=device_config['dw_min'],
    dw_min_std=0.0, dw_min_dtod=0.0,
    up_down=0.0,
    up_down_dtod=0.0,
    write_noise_std=0.0,  # No noise for linear
    piecewise_up=uniform_piecewise,
    piecewise_down=uniform_piecewise,
)
linear_config = SingleRPUConfig(device=linear_device)
devices.append({
    'name': 'Linear Device',
    'config': linear_config,
    'prefix': 'xor_Linear',
    'info': f'Uniform PiecewiseStepDevice (dw_min={device_config["dw_min"]}, piecewise=[1.0]*10, noise=0)',
})

# 3. Fitted Device
device_params = {k: v for k, v in device_config.items() if not k.startswith('_')}
fitted_device = PiecewiseStepDevice(**device_params)
fitted_config = SingleRPUConfig(device=fitted_device)
devices.append({
    'name': 'Fitted Device',
    'config': fitted_config,
    'prefix': 'xor_Fitted',
    'info': f'Fitted PiecewiseStepDevice (dw_min={device_config["dw_min"]}, noise={device_config["write_noise_std"]:.6f})',
})

# Run for all devices
all_results = []
for device in devices:
    result = generate_plots_for_device(
        device['name'],
        device['config'],
        device['prefix'],
        device['info'],
    )
    if result:
        all_results.append(result)

# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

for result in all_results:
    print(f"\n{result['device_name']}:")
    print(f"  Success Rate: {result['success_rate']:.1f}%")
    print(f"  Avg Convergence: {result['avg_convergence']:.1f} ± {result['std_convergence']:.1f} epochs")
    print(f"  Best Trial Loss: {result['best_trial_loss']:.6f}")

print("\n" + "=" * 70)
print("All devices processed successfully!")
print("=" * 70)
