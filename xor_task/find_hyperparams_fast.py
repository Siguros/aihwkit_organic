#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Hyperparameter Search for Idealized and Linear Devices on XOR Task.
Minimal search space for quick results.
"""

import json
import numpy as np
import torch
from torch import Tensor
from torch.nn import ReLU, Sequential
from torch.nn.functional import mse_loss
import torch.nn.init as init

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, PiecewiseStepDevice
from aihwkit.simulator.presets.devices import IdealizedPresetDevice

# Global seed
GLOBAL_SEED = 42

# XOR Dataset (3rd input is bias=1.0)
x_train = Tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
y_train = Tensor([[0.0], [1.0], [1.0], [0.0]])

# Search parameters - minimal for speed
NUM_TRIALS = 10
MAX_EPOCHS = 3000
EARLY_STOP_PATIENCE = 30
LOSS_THRESHOLD = 0.02

# Focused search space based on typical good values
LR_VALUES = [0.1, 0.2, 0.3, 0.4]
INIT_TYPES = ['normal', 'uniform']
INIT_STD_VALUES = [0.05, 0.1, 0.15]


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


def train_xor_trial(rpu_config, lr, init_type, init_std, trial_seed):
    """Train single XOR trial and return convergence epoch (or None if not converged)."""
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)

    model = Sequential(
        AnalogLinear(3, 3, bias=False, rpu_config=rpu_config),
        ReLU(),
        AnalogLinear(3, 1, bias=False, rpu_config=rpu_config),
    )
    custom_weight_init(model, init_type, init_std)

    opt = AnalogSGD(model.parameters(), lr=lr)
    opt.regroup_param_groups(model)

    consecutive_good = 0

    for epoch in range(MAX_EPOCHS):
        opt.zero_grad()
        pred = model(x_train)
        loss = mse_loss(pred, y_train)
        loss.backward()
        opt.step()

        current_loss = loss.item()
        preds = (pred > 0.5).float()
        acc = (preds == y_train).float().mean().item() * 100

        if acc == 100 and current_loss < LOSS_THRESHOLD:
            consecutive_good += 1
            if consecutive_good >= EARLY_STOP_PATIENCE:
                return epoch + 1 - EARLY_STOP_PATIENCE + 1, current_loss
        else:
            consecutive_good = 0

    return None, None


def evaluate_config(rpu_config, lr, init_type, init_std):
    """Evaluate a hyperparameter configuration."""
    convergence_epochs = []
    final_losses = []

    for trial in range(NUM_TRIALS):
        conv_epoch, final_loss = train_xor_trial(
            rpu_config, lr, init_type, init_std, GLOBAL_SEED + trial
        )
        if conv_epoch is not None:
            convergence_epochs.append(conv_epoch)
            final_losses.append(final_loss)

    success_rate = len(convergence_epochs) / NUM_TRIALS * 100
    avg_conv = np.mean(convergence_epochs) if convergence_epochs else MAX_EPOCHS
    std_conv = np.std(convergence_epochs) if convergence_epochs else 0
    avg_loss = np.mean(final_losses) if final_losses else 1.0

    return {
        'lr': lr,
        'init_type': init_type,
        'init_std': init_std,
        'success_rate': success_rate,
        'avg_convergence': avg_conv,
        'std_convergence': std_conv,
        'avg_final_loss': avg_loss,
    }


def search_device(device_name, rpu_config):
    """Search for optimal hyperparameters for a device."""
    print(f"\n{'='*60}")
    print(f"Searching: {device_name}")
    print(f"{'='*60}")

    all_results = []
    total = len(LR_VALUES) * len(INIT_TYPES) * len(INIT_STD_VALUES)
    count = 0

    for lr in LR_VALUES:
        for init_type in INIT_TYPES:
            for init_std in INIT_STD_VALUES:
                count += 1
                result = evaluate_config(rpu_config, lr, init_type, init_std)
                all_results.append(result)
                print(f"  [{count}/{total}] LR={lr}, {init_type}, std={init_std} -> "
                      f"{result['success_rate']:.0f}% ({result['avg_convergence']:.0f} epochs)")

    # Find best config
    perfect = [r for r in all_results if r['success_rate'] == 100]
    if perfect:
        perfect.sort(key=lambda x: x['avg_convergence'])
        best = perfect[0]
    else:
        all_results.sort(key=lambda x: (-x['success_rate'], x['avg_convergence']))
        best = all_results[0]

    print(f"\nBest: LR={best['lr']}, {best['init_type']}, std={best['init_std']}")
    print(f"  Success: {best['success_rate']:.0f}%, Avg conv: {best['avg_convergence']:.0f}")

    return best, all_results


def main():
    print("=" * 60)
    print("Fast XOR Hyperparameter Search")
    print("=" * 60)

    # Load device config
    with open('xor_device_config.json', 'r') as f:
        device_config = json.load(f)

    results = {}

    # 1. Idealized Device
    ideal_device = IdealizedPresetDevice()
    ideal_rpu = SingleRPUConfig(device=ideal_device)
    best_ideal, all_ideal = search_device('Idealized Device', ideal_rpu)
    results['idealized'] = {
        'best': best_ideal,
        'device_info': {'type': 'IdealizedPresetDevice', 'dw_min': ideal_device.dw_min}
    }

    # 2. Linear Device
    uniform_piecewise = [1.0] * 10
    linear_device = PiecewiseStepDevice(
        w_min=-1, w_max=1,
        w_min_dtod=0.0, w_max_dtod=0.0,
        dw_min=device_config['dw_min'],
        dw_min_std=0.0, dw_min_dtod=0.0,
        up_down=0.0, up_down_dtod=0.0,
        write_noise_std=0.0,
        piecewise_up=uniform_piecewise,
        piecewise_down=uniform_piecewise,
    )
    linear_rpu = SingleRPUConfig(device=linear_device)
    best_linear, all_linear = search_device('Linear Device', linear_rpu)
    results['linear'] = {
        'best': best_linear,
        'device_info': {'type': 'PiecewiseStepDevice (Linear)', 'dw_min': device_config['dw_min']}
    }

    # Save results
    for device_type in ['idealized', 'linear']:
        best = results[device_type]['best']
        output = {
            'optimal_config': {
                'network': '3x3x1',
                'activation': 'ReLU',
                'lr': best['lr'],
                'init_type': best['init_type'],
                'init_std': best['init_std'],
                'loss': 'MSE',
                'max_epochs': 5000,
            },
            'final_results': {
                'success_rate': best['success_rate'],
                'avg_convergence': best['avg_convergence'],
                'std_convergence': best['std_convergence'],
                'avg_final_loss': best['avg_final_loss'],
            },
            'device_info': results[device_type]['device_info'],
        }
        filename = f'xor_optimal_hyperparams_{device_type}.json'
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {filename}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for device_type in ['idealized', 'linear']:
        best = results[device_type]['best']
        print(f"\n{device_type.capitalize()}:")
        print(f"  LR={best['lr']}, init={best['init_type']}, std={best['init_std']}")
        print(f"  Success: {best['success_rate']:.0f}%, Avg epochs: {best['avg_convergence']:.0f}")


if __name__ == '__main__':
    main()
