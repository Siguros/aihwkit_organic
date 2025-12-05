#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended XOR Task Analysis:
- Fig-A: Learning curves with mean ± std (publication style)
- Fig-B: Output distribution for 4 input combinations
"""

import json
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch.nn import Sigmoid, MSELoss

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, PiecewiseStepDevice
from aihwkit.simulator.configs.devices import IdealDevice

# XOR Dataset
x_train = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = Tensor([[0.0], [1.0], [1.0], [0.0]])
INPUT_LABELS = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
TARGET_LABELS = ['0', '1', '1', '0']


def train_xor_extended(rpu_config, num_epochs=3000, lr=0.05, num_trials=30,
                       early_stop=False, patience=50):
    """
    Train XOR task and return:
    - loss/accuracy histories
    - model_snapshots (deepcopy of model objects at FPE or final epoch)
    """
    all_loss_histories = []
    all_acc_histories = []
    convergence_epochs = []
    model_snapshots = []  # Store deepcopy of model objects (NOT state_dict)

    for trial in range(num_trials):
        torch.manual_seed(42 + trial)
        np.random.seed(42 + trial)

        model = AnalogSequential(
            AnalogLinear(2, 8, bias=True, rpu_config=rpu_config),
            Sigmoid(),
            AnalogLinear(8, 1, bias=True, rpu_config=rpu_config),
            Sigmoid()
        )

        optimizer = AnalogSGD(model.parameters(), lr=lr)
        optimizer.regroup_param_groups(model)
        criterion = MSELoss()

        loss_hist = []
        acc_hist = []
        converged = None
        fpe_model = None  # First Perfect Epoch model (deepcopy)

        best_acc = -1.0
        wait = 0

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Re-evaluate AFTER optimizer.step() to get actual model state
            with torch.no_grad():
                outputs_after_step = model(x_train)
            preds = (outputs_after_step > 0.5).float()
            acc = (preds == y_train).float().mean().item() * 100

            loss_hist.append(loss.item())
            acc_hist.append(acc)

            # FPE (First Perfect Epoch) - record epoch but DON'T stop
            # Continue training until loss is low enough for good margin
            if acc == 100 and converged is None:
                converged = epoch + 1

            # Save model when BOTH accuracy is 100% AND loss is sufficiently low
            # This ensures good margin (outputs close to 0 and 1, not just > 0.5)
            if acc == 100 and loss.item() < 0.01 and fpe_model is None:
                fpe_model = copy.deepcopy(model)

            # Early stopping (optional)
            if early_stop:
                if acc > best_acc + 1e-9:
                    best_acc, wait = acc, 0
                else:
                    wait += 1
                if acc == 100 and wait >= patience:
                    # Pad remaining epochs
                    loss_hist.extend([loss_hist[-1]] * (num_epochs - epoch - 1))
                    acc_hist.extend([acc_hist[-1]] * (num_epochs - epoch - 1))
                    break

        # Save model snapshot: FPE model if exists, otherwise final model
        if fpe_model is not None:
            model_snapshots.append(fpe_model)
        else:
            model_snapshots.append(copy.deepcopy(model))

        all_loss_histories.append(loss_hist)
        all_acc_histories.append(acc_hist)
        convergence_epochs.append(converged if converged else num_epochs)

    return {
        'loss_histories': all_loss_histories,
        'acc_histories': all_acc_histories,
        'convergence_epochs': convergence_epochs,
        'model_snapshots': model_snapshots,  # deepcopy of model objects
        'rpu_config': rpu_config,
        'lr': lr,
    }


def pick_representative_trial(results, num_epochs):
    """
    Select representative trial:
    - Priority: FPE median among successful trials
    - Fallback: trial with minimum final loss
    """
    conv = results['convergence_epochs']
    succ = [(i, e) for i, e in enumerate(conv) if e is not None and e <= num_epochs]
    if succ:
        # FPE median closest trial
        epochs = np.array([e for _, e in succ])
        median = np.median(epochs)
        idx = np.argmin(np.abs(epochs - median))
        return succ[idx][0], succ[idx][1]  # (trial_index, fpe)
    # All failed: minimum final loss trial
    finals = [lh[-1] for lh in results['loss_histories']]
    t_idx = int(np.argmin(finals))
    return t_idx, None


def evaluate_output_distribution(model, x_eval, num_evals=200,
                                  eval_read_noise_std=0.0, use_logits=False):
    """
    Evaluate the passed model object directly for output distribution.
    No state_dict loading - uses the model object as-is.

    Args:
        model: Trained model object (deepcopy recommended)
        x_eval: Input tensor for evaluation
        num_evals: Number of forward passes for distribution
        eval_read_noise_std: Additional read noise (0.0 for none)
        use_logits: Whether outputs are logits (threshold=0) or sigmoid (threshold=0.5)
    """
    model.eval()
    thresh = 0.0 if use_logits else 0.5

    outs = [[] for _ in range(4)]
    with torch.inference_mode():
        for _ in range(num_evals):
            o = model(x_eval)
            y = o
            if eval_read_noise_std > 0:
                y = y + eval_read_noise_std * torch.randn_like(y)
            for i in range(4):
                outs[i].append(float(y[i].item()))
    return outs, thresh


def plot_paper_learning_curves(all_results, out_path='fig_a_learning_curves.png',
                               crop_by_quantile=True, q=0.95, margin=1.2, floor=200):
    """
    Publication-style learning curves with mean ± std shading.
    Optional x-axis cropping based on convergence quantile.
    """
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))

    def pick_xmax(conv_epochs, default_cap):
        succ = [e for e in conv_epochs if e is not None and e < default_cap]
        if not succ or not crop_by_quantile:
            return default_cap
        return int(max(floor, min(default_cap, np.ceil(np.quantile(succ, q) * margin))))

    xmax = None
    for r in all_results:
        # Pad histories to same length
        max_len = max(len(h) for h in r['acc_histories'])
        padded_acc = []
        padded_loss = []
        for h_acc, h_loss in zip(r['acc_histories'], r['loss_histories']):
            if len(h_acc) < max_len:
                padded_acc.append(h_acc + [h_acc[-1]] * (max_len - len(h_acc)))
                padded_loss.append(h_loss + [h_loss[-1]] * (max_len - len(h_loss)))
            else:
                padded_acc.append(h_acc)
                padded_loss.append(h_loss)

        acc = np.array(padded_acc)
        loss = np.array(padded_loss)
        acc_m, acc_s = acc.mean(0), acc.std(0)
        loss_m, loss_s = loss.mean(0), loss.std(0)
        epochs = np.arange(len(acc_m))

        ax_acc.plot(epochs, acc_m, color=r['color'], linewidth=2, label=r['name'])
        ax_acc.fill_between(epochs, acc_m - acc_s, acc_m + acc_s, color=r['color'], alpha=0.18)

        ax_loss.plot(epochs, loss_m, color=r['color'], linewidth=2, label=r['name'])
        ax_loss.fill_between(epochs, loss_m - loss_s, loss_m + loss_s, color=r['color'], alpha=0.18)

        # Compute xmax candidate for this device
        xmax_d = pick_xmax(r['convergence_epochs'], default_cap=len(acc_m))
        xmax = xmax_d if xmax is None else max(xmax, xmax_d)

    ax_acc.axhline(100, linestyle='--', color='k', alpha=0.5)
    ax_acc.set_xlabel('Epoch', fontsize=12)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=12)
    ax_acc.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax_acc.set_ylim(0, 105)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(loc='lower right', fontsize=10)

    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Loss (MSE)', fontsize=12)
    ax_loss.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc='upper right', fontsize=10)

    if xmax is not None:
        ax_acc.set_xlim(0, xmax)
        ax_loss.set_xlim(0, xmax)

    fig.suptitle('XOR Task Learning Curves (Mean ± Std)', y=1.02, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")


def plot_output_distribution_paper(all_results, devices, x_eval,
                                   out_path='fig_b_output_distribution.png',
                                   num_evals=200, eval_read_noise_std=0.0, use_logits=False):
    """
    Publication-style output distribution plot with violin/box + threshold line.
    Uses representative trial selection (FPE median or min final loss).
    Uses model objects directly (no state_dict loading).
    """
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for idx, (res, dev) in enumerate(zip(all_results, devices)):
        _, _, color, _ = dev  # Only use color from devices
        name = res['name']
        num_epochs = max(len(h) for h in res['acc_histories'])
        t_idx, fpe = pick_representative_trial(res, num_epochs=num_epochs)

        # Get model object directly (no state_dict loading)
        model = res['model_snapshots'][t_idx]

        data, thresh = evaluate_output_distribution(
            model, x_eval, num_evals=num_evals,
            eval_read_noise_std=eval_read_noise_std, use_logits=use_logits
        )

        ax = axes[idx]
        positions = [1, 2, 3, 4]

        # Violin plot
        parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.55)
            pc.set_facecolor(color)

        # Box plot overlay
        bp = ax.boxplot(data, positions=positions, widths=0.2, patch_artist=True, showfliers=True)
        for b in bp['boxes']:
            b.set_facecolor('white')
            b.set_alpha(0.9)

        # Threshold line
        ax.axhline(thresh, color='r', linestyle='--', linewidth=2, label='Threshold')

        # Target markers
        targets = [0, 1, 1, 0]
        for i, t in enumerate(targets):
            ax.scatter(i + 1, t, s=140, marker='*', color='gold', edgecolor='k',
                       zorder=5, label='Target' if i == 0 else '')

        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['(0,0)→0', '(0,1)→1', '(1,0)→1', '(1,1)→0'])
        ax.set_ylabel('Output (Sigmoid)' if not use_logits else 'Output (Logit)')
        fpe_str = fpe if fpe else 'N/A'
        ax.set_title(f"{name}\n(rep trial #{t_idx + 1}, FPE={fpe_str})", color=color, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        if idx == 0:
            ax.legend(loc='lower right', fontsize=9)

    fig.suptitle(f"XOR Output Distributions (K={num_evals} evals per input, noise={eval_read_noise_std})",
                 y=0.98, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")


# =============================================================================
# Setup Devices
# =============================================================================
print("=" * 70)
print("Extended XOR Task Analysis")
print("=" * 70)

# Load fitted config
with open('optimized_device_config.json', 'r') as f:
    config = json.load(f)

FITTED_DW_MIN = config['dw_min']

# 1. Ideal Device (truly ideal with no noise)
ideal_device = IdealDevice()
ideal_config = SingleRPUConfig(device=ideal_device)
# Disable noise for deterministic training and evaluation
ideal_config.forward.out_noise = 0.0
ideal_config.forward.inp_noise = 0.0
ideal_config.backward.out_noise = 0.0
ideal_config.backward.inp_noise = 0.0

# 2. Linear Device (uniform piecewise)
uniform_piecewise = [1.0] * 10
linear_device = PiecewiseStepDevice(
    w_min=-1, w_max=1,
    w_min_dtod=0.0, w_max_dtod=0.0,
    dw_min=FITTED_DW_MIN,
    dw_min_std=0.0, dw_min_dtod=0.0,
    up_down=0.0, up_down_dtod=0.0,
    write_noise_std=0.0,
    piecewise_up=uniform_piecewise,
    piecewise_down=uniform_piecewise,
)
linear_config = SingleRPUConfig(device=linear_device)
# Disable noise for deterministic training and evaluation
linear_config.forward.out_noise = 0.0
linear_config.forward.inp_noise = 0.0
linear_config.backward.out_noise = 0.0
linear_config.backward.inp_noise = 0.0

# 3. Fitted Device
fitted_device = PiecewiseStepDevice(
    w_min=config['w_min'], w_max=config['w_max'],
    w_min_dtod=0.0, w_max_dtod=0.0,
    dw_min=FITTED_DW_MIN,
    dw_min_std=0.0, dw_min_dtod=0.0,
    up_down=config['up_down'], up_down_dtod=0.0,
    write_noise_std=0.0,
    piecewise_up=config['piecewise_up'],
    piecewise_down=config['piecewise_down'],
)
fitted_config = SingleRPUConfig(device=fitted_device)
# Disable noise for deterministic training and evaluation
fitted_config.forward.out_noise = 0.0
fitted_config.forward.inp_noise = 0.0
fitted_config.backward.out_noise = 0.0
fitted_config.backward.inp_noise = 0.0

# =============================================================================
# Train All Models
# =============================================================================
NUM_TRIALS = 30
NUM_EPOCHS = 3000

devices = [
    ('Ideal Device', ideal_config, 'tab:blue', 5.0),
    ('Linear Device', linear_config, 'tab:green', 5.0),
    ('Fitted Device', fitted_config, 'tab:red', 0.05),
]

print(f"\nTraining: {NUM_TRIALS} trials, {NUM_EPOCHS} epochs")
print("-" * 70)

all_results = []
for name, rpu_config, color, lr in devices:
    print(f"Training {name} (lr={lr})...")
    results = train_xor_extended(rpu_config, NUM_EPOCHS, lr, NUM_TRIALS)
    results['name'] = name
    results['color'] = color
    all_results.append(results)

# =============================================================================
# Fig-A: Publication-style Learning Curves (Mean ± Std)
# =============================================================================
print("\nGenerating Fig-A: Learning Curves...")
plot_paper_learning_curves(all_results, out_path='fig_a_learning_curves.png',
                           crop_by_quantile=True, q=0.95, margin=1.2, floor=200)

# =============================================================================
# Fig-B: Output Distribution (Violin/Box plots)
# =============================================================================
print("\nGenerating Fig-B: Output Distribution...")
# forward_noise=0.0 (default) for deterministic evaluation
# eval_read_noise_std=0.0 to disable additional read noise
plot_output_distribution_paper(all_results, devices, x_train,
                               out_path='fig_b_output_distribution.png',
                               num_evals=200, eval_read_noise_std=0.0, use_logits=False)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Results Summary")
print("=" * 70)

for results in all_results:
    conv_epochs = results['convergence_epochs']
    # Fixed: use <= instead of < for boundary condition
    successful = [e for e in conv_epochs if e is not None and e <= NUM_EPOCHS]
    success_rate = len(successful) / NUM_TRIALS * 100

    print(f"\n{results['name']} (lr={results['lr']}):")
    print(f"  Success Rate: {success_rate:.0f}% ({len(successful)}/{NUM_TRIALS})")
    if successful:
        print(f"  Avg convergence: {np.mean(successful):.1f} ± {np.std(successful):.1f} epochs")

print("\n" + "=" * 70)
print("Generated Figures:")
print("  - fig_a_learning_curves.png")
print("  - fig_b_output_distribution.png")
print("=" * 70)
