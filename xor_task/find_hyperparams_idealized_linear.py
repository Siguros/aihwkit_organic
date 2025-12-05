#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Search for Idealized and Linear Devices on XOR Task.

Searches for optimal:
- Learning Rate (LR)
- Weight initialization type (normal/uniform)
- Weight initialization std

Saves results to:
- xor_optimal_hyperparams_idealized.json
- xor_optimal_hyperparams_linear.json
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
from aihwkit.simulator.presets.devices import IdealizedPresetDevice

# Global seed
GLOBAL_SEED = 42

# XOR Dataset (3rd input is bias=1.0)
x_train = Tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
y_train = Tensor([[0.0], [1.0], [1.0], [0.0]])

# Search parameters
NUM_TRIALS_SEARCH = 30  # Per config during search
NUM_TRIALS_FINAL = 100  # For final validation
MAX_EPOCHS = 5000
EARLY_STOP_PATIENCE = 50
LOSS_THRESHOLD = 0.02

# Search space
LR_RANGE = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 0.35, 0.4, 0.5]
INIT_TYPES = ['normal', 'uniform']
INIT_STD_RANGE = [0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5]


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


def train_xor_trial(rpu_config, lr, init_type, init_std, max_epochs, trial_seed):
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

    for epoch in range(max_epochs):
        opt.zero_grad()
        pred = model(x_train)
        loss = mse_loss(pred, y_train)
        loss.backward()
        opt.step()

        current_loss = loss.item()
        preds = (pred > 0.5).float()
        acc = (preds == y_train).float().mean().item() * 100

        loss_ok = (LOSS_THRESHOLD is None) or (current_loss < LOSS_THRESHOLD)
        if acc == 100 and loss_ok:
            consecutive_good += 1
            if consecutive_good >= EARLY_STOP_PATIENCE:
                return epoch + 1 - EARLY_STOP_PATIENCE + 1  # Convergence epoch
        else:
            consecutive_good = 0

    return None  # Did not converge


def evaluate_config(rpu_config, lr, init_type, init_std, num_trials):
    """Evaluate a hyperparameter configuration."""
    convergence_epochs = []

    for trial in range(num_trials):
        conv_epoch = train_xor_trial(
            rpu_config, lr, init_type, init_std, MAX_EPOCHS, GLOBAL_SEED + trial
        )
        if conv_epoch is not None:
            convergence_epochs.append(conv_epoch)

    success_rate = len(convergence_epochs) / num_trials * 100

    if convergence_epochs:
        avg_conv = np.mean(convergence_epochs)
        std_conv = np.std(convergence_epochs)
    else:
        avg_conv = MAX_EPOCHS
        std_conv = 0

    return {
        'lr': lr,
        'init_type': init_type,
        'init_std': init_std,
        'success_rate': success_rate,
        'avg_convergence': avg_conv,
        'std_convergence': std_conv,
        'num_successful': len(convergence_epochs),
    }


def search_hyperparams(device_name, rpu_config):
    """Search for optimal hyperparameters."""
    print(f"\n{'='*70}")
    print(f"Hyperparameter Search for {device_name}")
    print(f"{'='*70}")
    print(f"Search space: {len(LR_RANGE)} LRs x {len(INIT_TYPES)} init_types x {len(INIT_STD_RANGE)} init_stds")
    print(f"Total configs: {len(LR_RANGE) * len(INIT_TYPES) * len(INIT_STD_RANGE)}")
    print(f"Trials per config (search): {NUM_TRIALS_SEARCH}")

    start_time = time.time()
    all_results = []
    total_configs = len(LR_RANGE) * len(INIT_TYPES) * len(INIT_STD_RANGE)
    config_count = 0

    for lr in LR_RANGE:
        for init_type in INIT_TYPES:
            for init_std in INIT_STD_RANGE:
                config_count += 1
                result = evaluate_config(rpu_config, lr, init_type, init_std, NUM_TRIALS_SEARCH)
                all_results.append(result)

                if config_count % 20 == 0:
                    elapsed = time.time() - start_time
                    remaining = elapsed / config_count * (total_configs - config_count)
                    print(f"  Progress: {config_count}/{total_configs} "
                          f"({config_count/total_configs*100:.1f}%) "
                          f"- ETA: {remaining/60:.1f} min")

    elapsed_time = time.time() - start_time
    print(f"\nSearch completed in {elapsed_time/60:.1f} minutes")

    # Find 100% success configs
    perfect_configs = [r for r in all_results if r['success_rate'] == 100]
    print(f"\nConfigs with 100% success rate: {len(perfect_configs)}")

    if perfect_configs:
        # Sort by avg convergence
        perfect_configs.sort(key=lambda x: x['avg_convergence'])
        best_config = perfect_configs[0]
        print(f"Best config (fastest convergence): LR={best_config['lr']}, "
              f"init={best_config['init_type']}, std={best_config['init_std']}")
        print(f"  Avg convergence: {best_config['avg_convergence']:.1f} ± {best_config['std_convergence']:.1f}")
    else:
        # Find highest success rate
        all_results.sort(key=lambda x: (-x['success_rate'], x['avg_convergence']))
        best_config = all_results[0]
        print(f"Best config (highest success rate): LR={best_config['lr']}, "
              f"init={best_config['init_type']}, std={best_config['init_std']}")
        print(f"  Success rate: {best_config['success_rate']:.1f}%")

    return all_results, perfect_configs, best_config


def validate_best_config(device_name, rpu_config, best_config):
    """Validate best config with more trials."""
    print(f"\n{'='*70}")
    print(f"Validating Best Config for {device_name}")
    print(f"{'='*70}")
    print(f"Config: LR={best_config['lr']}, init={best_config['init_type']}, std={best_config['init_std']}")
    print(f"Running {NUM_TRIALS_FINAL} trials...")

    result = evaluate_config(
        rpu_config, best_config['lr'], best_config['init_type'], best_config['init_std'], NUM_TRIALS_FINAL
    )

    print(f"\nValidation Results:")
    print(f"  Success Rate: {result['success_rate']:.1f}% ({result['num_successful']}/{NUM_TRIALS_FINAL})")
    print(f"  Avg Convergence: {result['avg_convergence']:.1f} ± {result['std_convergence']:.1f}")

    return result


def save_results(device_name, output_file, all_results, perfect_configs, best_config, validation_result, device_info):
    """Save search results to JSON."""
    # Get top 20 configs by success rate then convergence
    all_results_sorted = sorted(all_results, key=lambda x: (-x['success_rate'], x['avg_convergence']))
    top_configs = all_results_sorted[:20]

    data = {
        'optimal_config': {
            'network': '3x3x1',
            'activation': 'ReLU',
            'lr': best_config['lr'],
            'init_type': best_config['init_type'],
            'init_std': best_config['init_std'],
            'loss': 'MSE',
            'max_epochs': MAX_EPOCHS,
        },
        'final_results': {
            'success_rate': validation_result['success_rate'],
            'avg_convergence': validation_result['avg_convergence'],
            'std_convergence': validation_result['std_convergence'],
            'num_successful': validation_result['num_successful'],
            'total_trials': NUM_TRIALS_FINAL,
        },
        'device_info': device_info,
        'search_params': {
            'lr_range': LR_RANGE,
            'init_types': INIT_TYPES,
            'init_std_range': INIT_STD_RANGE,
            'trials_per_config': NUM_TRIALS_SEARCH,
            'early_stop_patience': EARLY_STOP_PATIENCE,
            'loss_threshold': LOSS_THRESHOLD,
        },
        'all_100_percent_configs': [
            {
                'lr': c['lr'],
                'init_type': c['init_type'],
                'init_std': c['init_std'],
                'avg_conv': c['avg_convergence'],
                'std_conv': c['std_convergence'],
            }
            for c in sorted(perfect_configs, key=lambda x: x['avg_convergence'])
        ],
        'top_20_configs': [
            {
                'lr': c['lr'],
                'init_type': c['init_type'],
                'init_std': c['init_std'],
                'success_rate': c['success_rate'],
                'avg_conv': c['avg_convergence'],
                'std_conv': c['std_convergence'],
            }
            for c in top_configs
        ],
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved: {output_file}")


def main():
    print("=" * 70)
    print("XOR Hyperparameter Search for Idealized and Linear Devices")
    print("=" * 70)

    # Load device config for Linear device dw_min
    with open('xor_device_config.json', 'r') as f:
        device_config = json.load(f)

    # ===========================================================================
    # 1. Idealized Device
    # ===========================================================================
    ideal_device = IdealizedPresetDevice()
    ideal_config = SingleRPUConfig(device=ideal_device)
    ideal_info = {
        'type': 'IdealizedPresetDevice',
        'dw_min': ideal_device.dw_min,
    }

    all_results_ideal, perfect_ideal, best_ideal = search_hyperparams(
        'Idealized Device', ideal_config
    )
    validation_ideal = validate_best_config('Idealized Device', ideal_config, best_ideal)
    save_results(
        'Idealized Device',
        'xor_optimal_hyperparams_idealized.json',
        all_results_ideal, perfect_ideal, best_ideal, validation_ideal, ideal_info
    )

    # ===========================================================================
    # 2. Linear Device (Uniform PiecewiseStepDevice)
    # ===========================================================================
    uniform_piecewise = [1.0] * 10
    linear_device = PiecewiseStepDevice(
        w_min=-1, w_max=1,
        w_min_dtod=0.0, w_max_dtod=0.0,
        dw_min=device_config['dw_min'],
        dw_min_std=0.0, dw_min_dtod=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        write_noise_std=0.0,  # No noise
        piecewise_up=uniform_piecewise,
        piecewise_down=uniform_piecewise,
    )
    linear_config = SingleRPUConfig(device=linear_device)
    linear_info = {
        'type': 'PiecewiseStepDevice (Uniform/Linear)',
        'dw_min': device_config['dw_min'],
        'piecewise': 'uniform [1.0]*10',
        'write_noise_std': 0.0,
    }

    all_results_linear, perfect_linear, best_linear = search_hyperparams(
        'Linear Device', linear_config
    )
    validation_linear = validate_best_config('Linear Device', linear_config, best_linear)
    save_results(
        'Linear Device',
        'xor_optimal_hyperparams_linear.json',
        all_results_linear, perfect_linear, best_linear, validation_linear, linear_info
    )

    # ===========================================================================
    # Summary
    # ===========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nIdealized Device:")
    print(f"  Optimal: LR={best_ideal['lr']}, init={best_ideal['init_type']}, std={best_ideal['init_std']}")
    print(f"  Validation: {validation_ideal['success_rate']:.1f}% success, "
          f"{validation_ideal['avg_convergence']:.1f}±{validation_ideal['std_convergence']:.1f} epochs")
    print(f"  Configs with 100% success: {len(perfect_ideal)}")

    print(f"\nLinear Device:")
    print(f"  Optimal: LR={best_linear['lr']}, init={best_linear['init_type']}, std={best_linear['init_std']}")
    print(f"  Validation: {validation_linear['success_rate']:.1f}% success, "
          f"{validation_linear['avg_convergence']:.1f}±{validation_linear['std_convergence']:.1f} epochs")
    print(f"  Configs with 100% success: {len(perfect_linear)}")

    print("\n" + "=" * 70)
    print("Saved files:")
    print("  - xor_optimal_hyperparams_idealized.json")
    print("  - xor_optimal_hyperparams_linear.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
