#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate XOR Training Plots:
1. xor_successful_acc_loss.png - Accuracy and Loss curves (stable 100% trials only, with early stop)
2. xor_output_distribution.png - Output distribution with targets as stars
3. xor_decision_boundary.png - Decision boundary visualization

Uses optimal hyperparameters from xor_optimal_hyperparams.json
Also saves data to xlsx files.
"""

import json
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
EARLY_STOP_PATIENCE = 50  # 50 consecutive iterations at 100% acc AND loss < threshold
LOSS_THRESHOLD = 0.02  # Target loss threshold for good output quality

print("=" * 70)
print("XOR Plot Generation (Using xor_optimal_hyperparams.json)")
print("=" * 70)
print(f"Device config: xor_device_config.json")
print(f"Network: {optimal['network']} with {optimal['activation']}")
print(f"LR: {LR}, Init: {INIT_TYPE}, Std: {INIT_STD}")
print(f"write_noise_std: {device_config['write_noise_std']:.6f}")
if LOSS_THRESHOLD is not None:
    print(f"Early stop: {EARLY_STOP_PATIENCE} consecutive iterations at 100% AND loss < {LOSS_THRESHOLD}")
else:
    print(f"Early stop: {EARLY_STOP_PATIENCE} consecutive iterations at 100% accuracy")


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


def train_xor_with_early_stop(rpu_config, lr, init_type, init_std, max_epochs, early_stop_patience, loss_threshold):
    """
    Train XOR and return results.
    Early stop when:
    1. 100% accuracy is achieved AND
    2. Loss stays below threshold for early_stop_patience epochs
    Returns model weights for later use.
    """
    all_trial_info = []

    for trial in range(NUM_TRIALS):
        torch.manual_seed(42 + trial)
        np.random.seed(42 + trial)

        model = Sequential(
            AnalogLinear(3, 3, bias=False, rpu_config=rpu_config),
            ReLU(),
            AnalogLinear(3, 1, bias=False, rpu_config=rpu_config),
        )
        custom_weight_init(model, init_type, init_std)

        # Save initial weights after initialization
        initial_weights = []
        for module in model.modules():
            if isinstance(module, AnalogLinear):
                w, b = module.get_weights()
                initial_weights.append((w.clone(), b.clone() if b is not None else None))

        opt = AnalogSGD(model.parameters(), lr=lr)
        opt.regroup_param_groups(model)

        loss_history = []
        acc_history = []
        consecutive_good = 0  # Consecutive epochs with 100% acc AND loss < threshold
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

            # Check accuracy (and optionally loss threshold)
            loss_ok = (loss_threshold is None) or (current_loss < loss_threshold)
            if acc == 100 and loss_ok:
                consecutive_good += 1
                if consecutive_good >= early_stop_patience:
                    converged_epoch = epoch + 1 - early_stop_patience + 1
                    # Save weights at convergence
                    final_weights = []
                    for module in model.modules():
                        if isinstance(module, AnalogLinear):
                            w, b = module.get_weights()
                            final_weights.append((w.clone(), b.clone() if b is not None else None))
                    # Get final predictions
                    with torch.no_grad():
                        final_pred = model(x_train).numpy()
                    break
            else:
                consecutive_good = 0

        # Only save if converged (stable 100% with low loss)
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
            print(f"  Trial {trial + 1}/{NUM_TRIALS} - Converged: {successful}")

    return all_trial_info


# Create fitted device config
device_params = {k: v for k, v in device_config.items() if not k.startswith('_')}
device = PiecewiseStepDevice(**device_params)
rpu_config = SingleRPUConfig(device=device)

# Train trials
if LOSS_THRESHOLD is not None:
    print(f"\nTraining {NUM_TRIALS} trials (early stop: 100% acc AND loss < {LOSS_THRESHOLD})...")
else:
    print(f"\nTraining {NUM_TRIALS} trials (early stop: 100% acc for {EARLY_STOP_PATIENCE} iterations)...")
stable_trials = train_xor_with_early_stop(
    rpu_config,
    lr=LR,
    init_type=INIT_TYPE,
    init_std=INIT_STD,
    max_epochs=MAX_EPOCHS,
    early_stop_patience=EARLY_STOP_PATIENCE,
    loss_threshold=LOSS_THRESHOLD,
)

print(f"\nConverged trials (100% acc + low loss): {len(stable_trials)}/{NUM_TRIALS}")

if len(stable_trials) == 0:
    print("ERROR: No stable trials found!")
    exit(1)

# ============================================================
# 1) xor_successful_acc_loss.png - Mean/Std with early stop
# ============================================================
print("\n" + "-" * 70)
print("Creating xor_successful_acc_loss.png...")
print("-" * 70)

# Standard iteration-based plotting (0 to max_iterations)
# Pad with final values after convergence
max_iterations = 2000
convergence_epochs_list = [t['converged_epoch'] for t in stable_trials]

acc_matrix = np.full((len(stable_trials), max_iterations), np.nan)
loss_matrix = np.full((len(stable_trials), max_iterations), np.nan)

for i, t in enumerate(stable_trials):
    acc_hist = np.array(t['acc_history'])
    loss_hist = np.array(t['loss_history'])
    hist_len = min(len(acc_hist), max_iterations)

    # Fill actual training history
    acc_matrix[i, :hist_len] = acc_hist[:hist_len]
    loss_matrix[i, :hist_len] = loss_hist[:hist_len]

    # Last Value Carry-Forward (LVCF) padding
    # Pad with last recorded values (not fixed 100%)
    if hist_len < max_iterations:
        acc_matrix[i, hist_len:] = acc_hist[-1]
        loss_matrix[i, hist_len:] = loss_hist[-1]

# Calculate mean and std
mean_acc = np.nanmean(acc_matrix, axis=0)
std_acc = np.nanstd(acc_matrix, axis=0)
mean_loss = np.nanmean(loss_matrix, axis=0)
std_loss = np.nanstd(loss_matrix, axis=0)

iterations = np.arange(max_iterations)

# Calculate SEM (Standard Error of Mean) instead of Std
# SEM = Std / sqrt(N), shows confidence in the mean estimate
n_trials = len(stable_trials)
sem_acc = std_acc / np.sqrt(n_trials)
sem_loss = std_loss / np.sqrt(n_trials)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Accuracy curves with SEM band
ax1 = axes[0]

# Plot mean with SEM band
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

# Add text box with statistics
avg_conv = np.mean([t['converged_epoch'] for t in stable_trials])
std_conv = np.std([t['converged_epoch'] for t in stable_trials])
textstr = f'Convergence: {avg_conv:.1f} ± {std_conv:.1f} iterations'
ax1.text(0.95, 0.05, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Right: Loss curves with SEM band
ax2 = axes[1]

# Plot mean with SEM band
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

# Add text box with final loss
final_loss_mean = np.mean([t['final_loss'] for t in stable_trials])
final_loss_std = np.std([t['final_loss'] for t in stable_trials])
textstr = f'Final Loss: {final_loss_mean:.4f} ± {final_loss_std:.4f}'
ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.suptitle(f'XOR Training with Fitted Device\n'
             f'(LR={LR}, Init={INIT_TYPE}, std={INIT_STD}, noise={device_config["write_noise_std"]:.4f})',
             fontsize=12, y=1.02)

plt.tight_layout()
plt.savefig('xor_successful_acc_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xor_successful_acc_loss.png")

# ============================================================
# 2) xor_output_distribution.png - Using converged model predictions
# ============================================================
print("\n" + "-" * 70)
print("Creating xor_output_distribution.png...")
print("-" * 70)

# Collect final predictions from all stable trials
all_predictions = np.array([t['final_predictions'] for t in stable_trials])

mean_predictions = np.mean(all_predictions, axis=0)
std_predictions = np.std(all_predictions, axis=0)

fig, ax = plt.subplots(figsize=(10, 6))

input_labels = ['(0, 0)', '(0, 1)', '(1, 0)', '(1, 1)']
x_positions = np.array([0, 1, 2, 3])
targets = y_train.numpy().flatten()

# Plot individual predictions as scatter points
np.random.seed(42)
for preds in all_predictions:
    jitter = np.random.uniform(-0.1, 0.1, len(preds))
    ax.scatter(x_positions + jitter, preds, c='lightblue', alpha=0.3, s=30)

# Plot mean predictions with error bars
ax.errorbar(x_positions, mean_predictions, yerr=std_predictions,
            fmt='o', markersize=12, color='blue', capsize=5, capthick=2,
            ecolor='blue', elinewidth=2, label='Predicted (mean ± std)')

# Plot targets as stars
ax.scatter(x_positions, targets, c='red', s=300, marker='*',
           zorder=10, label='Target', edgecolors='darkred', linewidths=1)

# Decision threshold line
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Threshold (0.5)')

# Styling
ax.set_xticks(x_positions)
ax.set_xticklabels(input_labels, fontsize=12)
ax.set_xlabel('Input (X1, X2)', fontsize=12)
ax.set_ylabel('Output Value', fontsize=12)
ax.set_title(f'XOR Output Distribution\n({len(stable_trials)} stable trials, early stopped)', fontsize=14)
ax.set_ylim(-0.1, 1.1)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add annotation for each input
for i in range(4):
    y_offset = 0.15 if targets[i] == 0 else -0.2
    ax.annotate(f'μ={mean_predictions[i]:.3f}\nσ={std_predictions[i]:.3f}',
                xy=(x_positions[i], mean_predictions[i]),
                xytext=(x_positions[i] + 0.25, mean_predictions[i] + y_offset),
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('xor_output_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xor_output_distribution.png")

# ============================================================
# 3) xor_decision_boundary.png - Using best trial's weights
# ============================================================
print("\n" + "-" * 70)
print("Creating xor_decision_boundary.png...")
print("-" * 70)

# Find best trial (lowest final loss)
best_trial = min(stable_trials, key=lambda t: t['final_loss'])
print(f"Best trial: {best_trial['trial']} (loss={best_trial['final_loss']:.6f}, converged at epoch {best_trial['converged_epoch']})")

# Use the same aihwkit rpu_config for inference
# Create model with same config and load saved weights
best_model = Sequential(
    AnalogLinear(3, 3, bias=False, rpu_config=rpu_config),
    ReLU(),
    AnalogLinear(3, 1, bias=False, rpu_config=rpu_config),
)

# Load weights from best trial
saved_weights = best_trial['final_weights']
weight_idx = 0
for module in best_model.modules():
    if isinstance(module, AnalogLinear):
        w, b = saved_weights[weight_idx]
        module.set_weights(w, b)
        weight_idx += 1

best_model.eval()

# Get predictions using the saved weights (with aihwkit - may have noise)
# Average multiple runs to get stable prediction
NUM_RUNS = 50
with torch.no_grad():
    preds_sum = np.zeros(4)
    for _ in range(NUM_RUNS):
        preds_sum += best_model(x_train).numpy().flatten()
    final_predictions = preds_sum / NUM_RUNS

print(f"Predictions (aihwkit, avg of {NUM_RUNS} runs): {final_predictions}")
print(f"Targets: {y_train.numpy().flatten()}")

# Check if predictions are correct
preds_binary = (final_predictions > 0.5).astype(float)
correct = np.all(preds_binary == targets)
print(f"All correct: {correct}")

fig, ax = plt.subplots(figsize=(8, 8))

# Create meshgrid for decision boundary
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid_points = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape)]
grid_tensor = Tensor(grid_points)

# Compute decision boundary (average multiple runs for stability)
print(f"Computing decision boundary (averaging {NUM_RUNS} runs with aihwkit)...")
with torch.no_grad():
    Z_sum = np.zeros(xx.shape)
    for _ in range(NUM_RUNS):
        Z_sum += best_model(grid_tensor).numpy().reshape(xx.shape)
    Z = Z_sum / NUM_RUNS

# Plot decision boundary
contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21), cmap='RdYlBu_r', alpha=0.8)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

# Add colorbar
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Network Output', fontsize=12)

# Plot XOR data points
x_data = x_train[:, :2].numpy()
y_data = y_train.numpy().flatten()

class_0 = x_data[y_data == 0]
class_1 = x_data[y_data == 1]

ax.scatter(class_0[:, 0], class_0[:, 1], c='blue', s=200, marker='o',
           edgecolors='white', linewidths=2, label='Target: 0', zorder=5)
ax.scatter(class_1[:, 0], class_1[:, 1], c='red', s=200, marker='o',
           edgecolors='white', linewidths=2, label='Target: 1', zorder=5)

# Add predicted values as annotations
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
ax.set_title(f'XOR Decision Boundary (Best Trial {best_trial["trial"]})\n'
             f'Converged at epoch {best_trial["converged_epoch"]}, Loss={best_trial["final_loss"]:.6f}',
             fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xor_decision_boundary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: xor_decision_boundary.png")

# Save decision boundary grid data for xlsx
decision_boundary_grid = {
    'X1': xx.ravel(),
    'X2': yy.ravel(),
    'Prediction': Z.ravel(),
}
df_decision_grid = pd.DataFrame(decision_boundary_grid)

# Also save XOR data points with predictions
xor_points_data = {
    'X1': x_data[:, 0],
    'X2': x_data[:, 1],
    'Target': y_data,
    'Prediction': final_predictions,
}

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Generated Files:
  1. xor_successful_acc_loss.png - Mean/Std curves (stable 100% trials, early stopped)
  2. xor_output_distribution.png - Output distribution from converged models
  3. xor_decision_boundary.png - Decision boundary from best trial's weights

Results:
  - Stable 100% trials: {len(stable_trials)}/{NUM_TRIALS}
  - Average convergence: {avg_conv:.1f} ± {std_conv:.1f} epochs
  - Best trial: {best_trial['trial']} (loss={best_trial['final_loss']:.6f})

Training Configuration:
  - Network: {optimal['network']} with {optimal['activation']}
  - LR: {LR}, Init: {INIT_TYPE}, std={INIT_STD}
  - write_noise_std: {device_config['write_noise_std']:.6f}
  - Early stop: {EARLY_STOP_PATIENCE} consecutive epochs at 100%
""")

# ============================================================
# Save results to xlsx files
# ============================================================
print("\n" + "-" * 70)
print("Saving results to xlsx files...")
print("-" * 70)

# 1) Training Accuracy/Loss data (with clipping for std/sem)
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
df_acc_loss.to_excel('xor_training_acc_loss.xlsx', index=False)
print("Saved: xor_training_acc_loss.xlsx")

# 2) Output Distribution data
output_dist_data = {
    'Input': ['(0,0)', '(0,1)', '(1,0)', '(1,1)'],
    'Target': [0, 1, 1, 0],
    'Prediction_Mean': mean_predictions,
    'Prediction_Std': std_predictions,
}
df_output = pd.DataFrame(output_dist_data)
df_output.to_excel('xor_output_distribution.xlsx', index=False)

# 2-1) Output Distribution data points (all individual predictions)
# Save all_predictions with jitter for reproducibility
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
df_output_points.to_excel('xor_output_distribution_points.xlsx', index=False)
print("Saved: xor_output_distribution_points.xlsx")
print("Saved: xor_output_distribution.xlsx")

# 3) Decision Boundary data
# Save grid data (200x200 = 40000 points)
df_decision_grid.to_excel('xor_decision_boundary_grid.xlsx', index=False)
print("Saved: xor_decision_boundary_grid.xlsx")

# Save XOR points with predictions
df_xor_points = pd.DataFrame(xor_points_data)
df_xor_points.to_excel('xor_decision_boundary_points.xlsx', index=False)
print("Saved: xor_decision_boundary_points.xlsx")

# 4) Configuration summary
config_data = {
    'Parameter': [
        'Network Architecture',
        'Activation Function',
        'Learning Rate',
        'Weight Init Type',
        'Weight Init Std',
        'Loss Function',
        'Max Epochs',
        'Early Stop Patience',
        'Loss Threshold',
        'Device Write Noise Std',
        'Num Trials',
        'Success Rate',
        'Avg Convergence Epoch',
        'Std Convergence Epoch',
        'Best Trial',
        'Best Trial Loss',
    ],
    'Value': [
        optimal['network'],
        optimal['activation'],
        LR,
        INIT_TYPE,
        INIT_STD,
        optimal['loss'],
        MAX_EPOCHS,
        EARLY_STOP_PATIENCE,
        LOSS_THRESHOLD,
        device_config['write_noise_std'],
        NUM_TRIALS,
        f"{len(stable_trials)}/{NUM_TRIALS} ({len(stable_trials)/NUM_TRIALS*100:.1f}%)",
        f"{avg_conv:.1f}",
        f"{std_conv:.1f}",
        best_trial['trial'],
        f"{best_trial['final_loss']:.6f}",
    ]
}
df_config = pd.DataFrame(config_data)
df_config.to_excel('xor_configuration.xlsx', index=False)
print("Saved: xor_configuration.xlsx")

# 5) Best trial weights (initial and final) for each layer
# Save to Excel with multiple sheets
with pd.ExcelWriter('xor_best_trial_weights.xlsx', engine='openpyxl') as writer:
    # Initial weights
    for layer_idx, (w, b) in enumerate(best_trial['initial_weights']):
        w_np = w.numpy()
        df_w = pd.DataFrame(w_np,
                           columns=[f'Input_{j}' for j in range(w_np.shape[1])],
                           index=[f'Neuron_{i}' for i in range(w_np.shape[0])])
        df_w.to_excel(writer, sheet_name=f'Layer{layer_idx+1}_Initial')

    # Final weights (after training)
    for layer_idx, (w, b) in enumerate(best_trial['final_weights']):
        w_np = w.numpy()
        df_w = pd.DataFrame(w_np,
                           columns=[f'Input_{j}' for j in range(w_np.shape[1])],
                           index=[f'Neuron_{i}' for i in range(w_np.shape[0])])
        df_w.to_excel(writer, sheet_name=f'Layer{layer_idx+1}_Final')

    # Summary sheet
    summary_data = {
        'Info': ['Best Trial', 'Converged Epoch', 'Final Loss',
                 'Layer 1 Shape', 'Layer 2 Shape',
                 'Init Type', 'Init Std'],
        'Value': [best_trial['trial'], best_trial['converged_epoch'],
                  f"{best_trial['final_loss']:.6f}",
                  f"{best_trial['initial_weights'][0][0].shape}",
                  f"{best_trial['initial_weights'][1][0].shape}",
                  INIT_TYPE, INIT_STD]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_excel(writer, sheet_name='Summary', index=False)

print("Saved: xor_best_trial_weights.xlsx")

print("\n" + "=" * 70)
print("All xlsx files saved successfully!")
print("=" * 70)
