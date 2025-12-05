#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Hyperparameter Search for Idealized Device only.
Wider search range to find better convergence.
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
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.presets.devices import IdealizedPresetDevice

GLOBAL_SEED = 42
x_train = Tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
y_train = Tensor([[0.0], [1.0], [1.0], [0.0]])

NUM_TRIALS = 10
MAX_EPOCHS = 5000
EARLY_STOP_PATIENCE = 30
LOSS_THRESHOLD = 0.02

# Extended search space for Idealized - IdealizedPresetDevice has very small dw_min (0.0002)
# Need smaller LR and larger init_std to compensate
LR_VALUES = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
INIT_TYPES = ['normal', 'uniform']
INIT_STD_VALUES = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]


def custom_weight_init(model, init_type, init_std):
    for module in model.modules():
        if isinstance(module, AnalogLinear):
            weight, bias = module.get_weights()
            if init_type == "normal":
                init.normal_(weight, mean=0.0, std=init_std)
            elif init_type == "uniform":
                init.uniform_(weight, -init_std, init_std)
            module.set_weights(weight, bias)


def train_xor_trial(rpu_config, lr, init_type, init_std, trial_seed):
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


def main():
    print("=" * 60)
    print("Extended Idealized Device Hyperparameter Search")
    print("=" * 60)

    ideal_device = IdealizedPresetDevice()
    print(f"IdealizedPresetDevice dw_min: {ideal_device.dw_min}")

    rpu_config = SingleRPUConfig(device=ideal_device)

    all_results = []
    total = len(LR_VALUES) * len(INIT_TYPES) * len(INIT_STD_VALUES)
    count = 0

    for lr in LR_VALUES:
        for init_type in INIT_TYPES:
            for init_std in INIT_STD_VALUES:
                count += 1
                result = evaluate_config(rpu_config, lr, init_type, init_std)
                all_results.append(result)

                status = f"{result['success_rate']:.0f}%"
                if result['success_rate'] > 0:
                    status += f" ({result['avg_convergence']:.0f} epochs)"
                print(f"  [{count}/{total}] LR={lr}, {init_type}, std={init_std} -> {status}")

    # Find best
    perfect = [r for r in all_results if r['success_rate'] == 100]
    high_success = [r for r in all_results if r['success_rate'] >= 80]

    if perfect:
        perfect.sort(key=lambda x: x['avg_convergence'])
        best = perfect[0]
    elif high_success:
        high_success.sort(key=lambda x: (-x['success_rate'], x['avg_convergence']))
        best = high_success[0]
    else:
        all_results.sort(key=lambda x: (-x['success_rate'], x['avg_convergence']))
        best = all_results[0]

    print(f"\n{'='*60}")
    print(f"Best config: LR={best['lr']}, {best['init_type']}, std={best['init_std']}")
    print(f"  Success: {best['success_rate']:.0f}%, Avg conv: {best['avg_convergence']:.0f}")

    # Show top 5
    all_results.sort(key=lambda x: (-x['success_rate'], x['avg_convergence']))
    print(f"\nTop 5 configs:")
    for i, r in enumerate(all_results[:5]):
        print(f"  {i+1}. LR={r['lr']}, {r['init_type']}, std={r['init_std']} -> "
              f"{r['success_rate']:.0f}% ({r['avg_convergence']:.0f} epochs)")

    # Save
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
        'device_info': {
            'type': 'IdealizedPresetDevice',
            'dw_min': ideal_device.dw_min
        },
    }
    with open('xor_optimal_hyperparams_idealized.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: xor_optimal_hyperparams_idealized.json")


if __name__ == '__main__':
    main()
