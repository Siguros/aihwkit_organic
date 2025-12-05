#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Optimization for Ideal and Linear Devices
Finds optimal learning rate and weight initialization for XOR task.
"""

import json
import time
import numpy as np
import torch
from torch import Tensor
from torch.nn import ReLU, Sequential
from torch.nn.functional import mse_loss
import torch.nn.init as init

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, PiecewiseStepDevice
from aihwkit.simulator.configs.devices import IdealDevice

# Global seed
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# XOR Dataset
x_train = Tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
y_train = Tensor([[0.0], [1.0], [1.0], [0.0]])

# Search space
LR_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]
INIT_TYPES = ["normal", "uniform"]
INIT_STD_VALUES = [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]

# Training parameters
NUM_TRIALS = 30  # trials per config
MAX_EPOCHS = 3000
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


def train_single_trial(rpu_config, lr, init_type, init_std, trial_seed):
    """Train a single trial and return convergence info."""
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
                return {
                    'converged': True,
                    'epoch': epoch + 1 - EARLY_STOP_PATIENCE + 1,
                    'final_loss': current_loss,
                }
        else:
            consecutive_good = 0

    return {'converged': False, 'epoch': MAX_EPOCHS, 'final_loss': current_loss}


def evaluate_config(rpu_config, lr, init_type, init_std):
    """Evaluate a configuration with multiple trials."""
    results = []
    for trial in range(NUM_TRIALS):
        result = train_single_trial(rpu_config, lr, init_type, init_std, 42 + trial)
        results.append(result)

    converged_trials = [r for r in results if r['converged']]
    success_rate = len(converged_trials) / NUM_TRIALS * 100

    if converged_trials:
        avg_conv = np.mean([r['epoch'] for r in converged_trials])
        std_conv = np.std([r['epoch'] for r in converged_trials])
        avg_loss = np.mean([r['final_loss'] for r in converged_trials])
    else:
        avg_conv = MAX_EPOCHS
        std_conv = 0
        avg_loss = 1.0

    return {
        'success_rate': success_rate,
        'avg_convergence': avg_conv,
        'std_convergence': std_conv,
        'avg_loss': avg_loss,
    }


def optimize_device(device_name, rpu_config):
    """Run hyperparameter optimization for a device."""
    print(f"\n{'='*70}")
    print(f"Optimizing: {device_name}")
    print(f"{'='*70}")

    start_time = time.time()
    all_results = []
    best_config = None
    best_score = -1  # success_rate first, then -avg_convergence

    total_configs = len(LR_VALUES) * len(INIT_TYPES) * len(INIT_STD_VALUES)
    config_idx = 0

    for lr in LR_VALUES:
        for init_type in INIT_TYPES:
            for init_std in INIT_STD_VALUES:
                config_idx += 1
                print(f"\n[{config_idx}/{total_configs}] LR={lr}, Init={init_type}, Std={init_std}")

                result = evaluate_config(rpu_config, lr, init_type, init_std)
                result['lr'] = lr
                result['init_type'] = init_type
                result['init_std'] = init_std
                all_results.append(result)

                print(f"  Success: {result['success_rate']:.1f}%, "
                      f"Avg Conv: {result['avg_convergence']:.1f} ± {result['std_convergence']:.1f}")

                # Score: prioritize 100% success, then fastest convergence
                score = result['success_rate'] * 10000 - result['avg_convergence']
                if score > best_score:
                    best_score = score
                    best_config = result.copy()

    elapsed_time = (time.time() - start_time) / 60

    # Get all 100% success configs
    perfect_configs = [r for r in all_results if r['success_rate'] == 100]
    perfect_configs.sort(key=lambda x: x['avg_convergence'])

    return {
        'device_name': device_name,
        'best_config': best_config,
        'all_100_percent_configs': perfect_configs[:20],  # Top 20
        'search_time_minutes': elapsed_time,
        'all_results': all_results,
    }


def main():
    # Load device config for dw_min
    with open('xor_device_config.json', 'r') as f:
        device_config = json.load(f)

    print("="*70)
    print("XOR Hyperparameter Optimization for Ideal and Linear Devices")
    print("="*70)
    print(f"Search space:")
    print(f"  LR: {LR_VALUES}")
    print(f"  Init types: {INIT_TYPES}")
    print(f"  Init std: {INIT_STD_VALUES}")
    print(f"  Trials per config: {NUM_TRIALS}")
    print(f"  Total configs: {len(LR_VALUES) * len(INIT_TYPES) * len(INIT_STD_VALUES)}")

    # 1. Ideal Device
    ideal_device = IdealDevice()
    ideal_config = SingleRPUConfig(device=ideal_device)
    ideal_results = optimize_device("Ideal Device", ideal_config)

    # Save Ideal results
    ideal_output = {
        "optimal_config": {
            "network": "3x3x1",
            "activation": "ReLU",
            "lr": ideal_results['best_config']['lr'],
            "init_type": ideal_results['best_config']['init_type'],
            "init_std": ideal_results['best_config']['init_std'],
            "loss": "MSE",
            "max_epochs": MAX_EPOCHS,
        },
        "final_results": {
            "success_rate": ideal_results['best_config']['success_rate'],
            "avg_convergence": ideal_results['best_config']['avg_convergence'],
            "std_convergence": ideal_results['best_config']['std_convergence'],
            "avg_loss": ideal_results['best_config']['avg_loss'],
        },
        "device_info": "IdealDevice (perfect floating-point, no noise)",
        "search_time_minutes": ideal_results['search_time_minutes'],
        "all_100_percent_configs": [
            {
                "lr": c['lr'],
                "init_type": c['init_type'],
                "init_std": c['init_std'],
                "avg_conv": c['avg_convergence'],
                "std_conv": c['std_convergence'],
            }
            for c in ideal_results['all_100_percent_configs']
        ],
    }

    with open('xor_optimal_hyperparams_ideal.json', 'w') as f:
        json.dump(ideal_output, f, indent=2)
    print(f"\nSaved: xor_optimal_hyperparams_ideal.json")

    # 2. Linear Device
    linear_device = PiecewiseStepDevice(
        w_min=-1, w_max=1,
        w_min_dtod=0.0, w_max_dtod=0.0,
        dw_min=device_config['dw_min'],
        dw_min_std=0.0, dw_min_dtod=0.0,
        up_down=0.0, up_down_dtod=0.0,
        write_noise_std=0.0,
        piecewise_up=[1.0]*10,
        piecewise_down=[1.0]*10,
    )
    linear_config = SingleRPUConfig(device=linear_device)
    linear_results = optimize_device("Linear Device", linear_config)

    # Save Linear results
    linear_output = {
        "optimal_config": {
            "network": "3x3x1",
            "activation": "ReLU",
            "lr": linear_results['best_config']['lr'],
            "init_type": linear_results['best_config']['init_type'],
            "init_std": linear_results['best_config']['init_std'],
            "loss": "MSE",
            "max_epochs": MAX_EPOCHS,
        },
        "final_results": {
            "success_rate": linear_results['best_config']['success_rate'],
            "avg_convergence": linear_results['best_config']['avg_convergence'],
            "std_convergence": linear_results['best_config']['std_convergence'],
            "avg_loss": linear_results['best_config']['avg_loss'],
        },
        "device_info": f"Linear PiecewiseStepDevice (dw_min={device_config['dw_min']}, uniform piecewise, no noise)",
        "search_time_minutes": linear_results['search_time_minutes'],
        "all_100_percent_configs": [
            {
                "lr": c['lr'],
                "init_type": c['init_type'],
                "init_std": c['init_std'],
                "avg_conv": c['avg_convergence'],
                "std_conv": c['std_convergence'],
            }
            for c in linear_results['all_100_percent_configs']
        ],
    }

    with open('xor_optimal_hyperparams_linear.json', 'w') as f:
        json.dump(linear_output, f, indent=2)
    print(f"Saved: xor_optimal_hyperparams_linear.json")

    # Print summary
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)

    print(f"\nIdeal Device:")
    print(f"  Best: LR={ideal_results['best_config']['lr']}, "
          f"Init={ideal_results['best_config']['init_type']}, "
          f"Std={ideal_results['best_config']['init_std']}")
    print(f"  Success: {ideal_results['best_config']['success_rate']:.1f}%, "
          f"Avg Conv: {ideal_results['best_config']['avg_convergence']:.1f}")

    print(f"\nLinear Device:")
    print(f"  Best: LR={linear_results['best_config']['lr']}, "
          f"Init={linear_results['best_config']['init_type']}, "
          f"Std={linear_results['best_config']['init_std']}")
    print(f"  Success: {linear_results['best_config']['success_rate']:.1f}%, "
          f"Avg Conv: {linear_results['best_config']['avg_convergence']:.1f}")


if __name__ == "__main__":
    main()
