# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""24-hour Hyperparameter Search for Fitted PiecewiseStepDevice ResNet18 CIFAR10.

3-Phase Search Strategy:
1. Phase 1: 50 random combinations x 50 epochs (screening)
2. Phase 2: Top 5 configurations x 100 epochs (validation)
3. Phase 3: Best configuration x 300 epochs (final training)

Fixed parameters:
- Architecture: ResNet18
- Device model: PiecewiseStepDevice from xor_device_config.json
- Batch size: 128

Search parameters:
- Learning rate: [0.01, 0.05, 0.1, 0.2]
- Momentum: [0.9, 0.95]
- Weight decay: [1e-4, 5e-4, 1e-3]
- Warmup ratio: [0.0, 0.04, 0.1]
- LR schedule: ['cosine', 'step']
- weight_scaling_omega: [0.4, 0.6, 0.8]
- Nesterov: [True, False]
"""
# pylint: disable=invalid-name

import os
import sys
import json
import time
import itertools
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Imports from PyTorch
import torch
from torch import nn, device, no_grad, manual_seed, save
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from torchvision import datasets, transforms

# Progress bar
from tqdm import tqdm

# Imports from aihwkit
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig, FloatingPointRPUConfig
from aihwkit.simulator.configs import MappingParameter, PiecewiseStepDevice


# ============================================================================
# Configuration
# ============================================================================
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

RESULTS = os.path.join(os.getcwd(), "results", "HYPERPARAM_SEARCH_24H")
os.makedirs(RESULTS, exist_ok=True)

# Fixed parameters
SEED = 1
BATCH_SIZE = 128
N_CLASSES = 10
NUM_WORKERS = 4

# Phase configuration
PHASE1_NUM_TRIALS = 50
PHASE1_EPOCHS = 50
PHASE2_TOP_K = 5
PHASE2_EPOCHS = 100
PHASE3_EPOCHS = 300

# Load Fitted Device Configuration
DEVICE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'xor_device_config.json')
with open(DEVICE_CONFIG_PATH, 'r') as f:
    device_config = json.load(f)
device_params = {k: v for k, v in device_config.items() if not k.startswith('_')}


# ============================================================================
# Search Space
# ============================================================================
SEARCH_SPACE = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'momentum': [0.9, 0.95],
    'weight_decay': [1e-4, 5e-4, 1e-3],
    'warmup_ratio': [0.0, 0.04, 0.1],
    'lr_schedule': ['cosine', 'step'],
    'weight_scaling_omega': [0.4, 0.6, 0.8],
    'nesterov': [True, False],
}

# Layer configuration (fixed)
LAYER_CONFIG = {
    'conv1': 'digital',
    'layer1_block0': {'conv1': 'analog', 'conv2': 'analog'},
    'layer1_block1': {'conv1': 'analog', 'conv2': 'analog'},
    'layer2_block0': {'conv1': 'analog', 'conv2': 'analog', 'downsample': 'analog'},
    'layer2_block1': {'conv1': 'analog', 'conv2': 'analog'},
    'layer3_block0': {'conv1': 'analog', 'conv2': 'analog', 'downsample': 'analog'},
    'layer3_block1': {'conv1': 'analog', 'conv2': 'analog'},
    'layer4_block0': {'conv1': 'analog', 'conv2': 'analog', 'downsample': 'analog'},
    'layer4_block1': {'conv1': 'analog', 'conv2': 'analog'},
    'fc': 'digital',
}


def create_analog_config(weight_scaling_omega=0.6):
    """Create analog configuration with specified omega."""
    fitted_device = PiecewiseStepDevice(**device_params)
    config = SingleRPUConfig(device=fitted_device)
    config.mapping = MappingParameter(
        weight_scaling_omega=weight_scaling_omega,
        learn_out_scaling=False,
        weight_scaling_lr_compensation=False,
        digital_bias=True,
        weight_scaling_columnwise=False,
        out_scaling_columnwise=False,
        max_input_size=512,
        max_output_size=512
    )
    return config


class ResidualBlockBaseline(nn.Module):
    """Residual block with configurable digital/analog layers."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1,
                 use_analog_conv1=False, use_analog_conv2=False, use_analog_convskip=False,
                 weight_scaling_omega=0.6):
        super().__init__()

        if use_analog_conv1:
            rpu_config_conv1 = create_analog_config(weight_scaling_omega)
        else:
            rpu_config_conv1 = FloatingPointRPUConfig()

        if use_analog_conv2:
            rpu_config_conv2 = create_analog_config(weight_scaling_omega)
        else:
            rpu_config_conv2 = FloatingPointRPUConfig()

        if use_analog_convskip:
            rpu_config_convskip = create_analog_config(weight_scaling_omega)
        else:
            rpu_config_convskip = FloatingPointRPUConfig()

        self.conv1 = AnalogConv2d(
            in_ch, hidden_ch, kernel_size=3, padding=1, stride=stride,
            bias=False, rpu_config=rpu_config_conv1
        )
        self.bn1 = nn.BatchNorm2d(hidden_ch)

        self.conv2 = AnalogConv2d(
            hidden_ch, hidden_ch, kernel_size=3, padding=1,
            bias=False, rpu_config=rpu_config_conv2
        )
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = AnalogConv2d(
                in_ch, hidden_ch, kernel_size=1, stride=stride,
                bias=False, rpu_config=rpu_config_convskip
            )
        else:
            self.convskip = None

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.convskip:
            x = self.convskip(x)
        y += x
        return F.relu(y)


def concatenate_layer_blocks(in_ch, hidden_ch, num_layer, first_layer=False,
                             block_configs=None, weight_scaling_omega=0.6):
    """Concatenate residual blocks."""
    if block_configs is None:
        block_configs = [{'conv1': 'digital', 'conv2': 'digital', 'downsample': 'digital'}] * num_layer

    layers = []
    for i in range(num_layer):
        config = block_configs[i]
        use_analog_conv1 = (config['conv1'] == 'analog')
        use_analog_conv2 = (config['conv2'] == 'analog')
        use_analog_downsample = (config.get('downsample', 'digital') == 'analog')

        if i == 0 and not first_layer:
            layers.append(ResidualBlockBaseline(
                in_ch, hidden_ch, use_conv=True, stride=2,
                use_analog_conv1=use_analog_conv1,
                use_analog_conv2=use_analog_conv2,
                use_analog_convskip=use_analog_downsample,
                weight_scaling_omega=weight_scaling_omega
            ))
        else:
            layers.append(ResidualBlockBaseline(
                hidden_ch, hidden_ch,
                use_analog_conv1=use_analog_conv1,
                use_analog_conv2=use_analog_conv2,
                use_analog_convskip=use_analog_conv1,
                weight_scaling_omega=weight_scaling_omega
            ))
    return layers


def create_model(weight_scaling_omega=0.6):
    """Create ResNet18 model with specified omega."""
    block_per_layers = (2, 2, 2, 2)
    base_channel = 64
    channel = (base_channel, 2 * base_channel, 4 * base_channel, 8 * base_channel)

    input_use_analog = (LAYER_CONFIG['conv1'] == 'analog')
    if input_use_analog:
        input_rpu_config = create_analog_config(weight_scaling_omega)
    else:
        input_rpu_config = FloatingPointRPUConfig()

    l0 = nn.Sequential(
        AnalogConv2d(3, channel[0], kernel_size=3, stride=1, padding=1,
                     bias=False, rpu_config=input_rpu_config),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
    )

    l1 = nn.Sequential(*concatenate_layer_blocks(
        channel[0], channel[0], block_per_layers[0], first_layer=True,
        block_configs=[LAYER_CONFIG['layer1_block0'], LAYER_CONFIG['layer1_block1']],
        weight_scaling_omega=weight_scaling_omega
    ))

    l2 = nn.Sequential(*concatenate_layer_blocks(
        channel[0], channel[1], block_per_layers[1],
        block_configs=[LAYER_CONFIG['layer2_block0'], LAYER_CONFIG['layer2_block1']],
        weight_scaling_omega=weight_scaling_omega
    ))

    l3 = nn.Sequential(*concatenate_layer_blocks(
        channel[1], channel[2], block_per_layers[2],
        block_configs=[LAYER_CONFIG['layer3_block0'], LAYER_CONFIG['layer3_block1']],
        weight_scaling_omega=weight_scaling_omega
    ))

    l4_conv = nn.Sequential(*concatenate_layer_blocks(
        channel[2], channel[3], block_per_layers[3],
        block_configs=[LAYER_CONFIG['layer4_block0'], LAYER_CONFIG['layer4_block1']],
        weight_scaling_omega=weight_scaling_omega
    ))

    fc_use_analog = (LAYER_CONFIG['fc'] == 'analog')
    if fc_use_analog:
        fc_rpu_config = create_analog_config(weight_scaling_omega)
    else:
        fc_rpu_config = FloatingPointRPUConfig()

    l5_fc = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        AnalogLinear(channel[3], N_CLASSES, bias=True, rpu_config=fc_rpu_config)
    )

    model = nn.Sequential(l0, l1, l2, l3, l4_conv, l5_fc)
    return model


def load_images():
    """Load CIFAR10 with augmentation."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=train_transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=val_transform)

    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=USE_CUDA)
    val_data = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=USE_CUDA)

    return train_data, val_data


def apply_warmup_cosine_lr(optimizer, epoch, total_epochs, base_lr, warmup_ratio=0.0, min_lr=1e-5):
    """Apply warmup + cosine annealing."""
    import math
    warmup_epochs = int(total_epochs * warmup_ratio)

    if epoch <= warmup_epochs and warmup_epochs > 0:
        current_lr = base_lr * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        current_lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr


def train_one_trial(params, train_data, val_data, num_epochs, trial_name="Trial"):
    """Train one trial with given hyperparameters."""
    manual_seed(SEED)

    # Create model
    model = create_model(weight_scaling_omega=params['weight_scaling_omega'])
    if USE_CUDA:
        model = model.to(DEVICE)

    # Create optimizer
    optimizer = AnalogSGD(
        model.parameters(),
        lr=params['learning_rate'],
        momentum=params['momentum'],
        weight_decay=params['weight_decay'],
        nesterov=params['nesterov']
    )
    optimizer.regroup_param_groups(model)

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    history = []

    # Training loop
    pbar = tqdm(range(num_epochs), desc=trial_name, leave=False)

    for epoch in pbar:
        # Adjust learning rate
        if params['lr_schedule'] == 'cosine':
            current_lr = apply_warmup_cosine_lr(
                optimizer, epoch + 1, num_epochs,
                params['learning_rate'], params['warmup_ratio']
            )
        else:  # step
            if epoch == 0:
                scheduler = StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)
            if epoch > 0:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_data:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(output.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with no_grad():
            for images, labels in val_data:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)

        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'lr': current_lr
        })

        pbar.set_postfix({
            'Train': f'{train_acc:.1f}%',
            'Val': f'{val_acc:.1f}%',
            'Best': f'{best_val_acc:.1f}%'
        })

    final_val_acc = val_acc

    # Clean up
    del model
    torch.cuda.empty_cache()

    return best_val_acc, final_val_acc, history


def generate_random_params(num_samples):
    """Generate random parameter combinations."""
    param_list = []
    random.seed(SEED)

    for _ in range(num_samples):
        params = {}
        for key, values in SEARCH_SPACE.items():
            params[key] = random.choice(values)
        param_list.append(params)

    return param_list


def params_to_str(params):
    """Convert params to readable string."""
    return (f"lr={params['learning_rate']}, mom={params['momentum']}, "
            f"wd={params['weight_decay']}, warmup={params['warmup_ratio']}, "
            f"sched={params['lr_schedule']}, omega={params['weight_scaling_omega']}, "
            f"nesterov={params['nesterov']}")


def main():
    """Run 3-phase hyperparameter search."""
    start_time = time.time()

    print("=" * 70)
    print("3-Phase Hyperparameter Search for Fitted Device ResNet18 CIFAR10")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE} (fixed)")
    print(f"\nPhase 1: {PHASE1_NUM_TRIALS} trials x {PHASE1_EPOCHS} epochs (screening)")
    print(f"Phase 2: Top {PHASE2_TOP_K} configs x {PHASE2_EPOCHS} epochs (validation)")
    print(f"Phase 3: Best config x {PHASE3_EPOCHS} epochs (final)")
    print(f"\nSearch space:")
    for key, values in SEARCH_SPACE.items():
        print(f"  {key}: {values}")

    # Load data once
    print("\nLoading CIFAR10 data...")
    train_data, val_data = load_images()

    # ========================================================================
    # PHASE 1: Screening (50 trials x 50 epochs)
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 1: Screening ({PHASE1_NUM_TRIALS} trials x {PHASE1_EPOCHS} epochs)")
    print("=" * 70)

    phase1_params = generate_random_params(PHASE1_NUM_TRIALS)
    phase1_results = []

    for trial_num, params in enumerate(phase1_params, 1):
        print(f"\n[Phase 1 - Trial {trial_num}/{PHASE1_NUM_TRIALS}]")
        print(f"  {params_to_str(params)}")

        trial_start = time.time()

        try:
            best_acc, final_acc, history = train_one_trial(
                params, train_data, val_data, PHASE1_EPOCHS,
                trial_name=f"P1-{trial_num}/{PHASE1_NUM_TRIALS}"
            )
            trial_time = time.time() - trial_start

            result = {
                'trial': trial_num,
                'phase': 1,
                'epochs': PHASE1_EPOCHS,
                'best_val_acc': best_acc,
                'final_val_acc': final_acc,
                'trial_time_sec': trial_time,
                **params
            }
            phase1_results.append(result)

            print(f"  Result: Best={best_acc:.2f}%, Final={final_acc:.2f}% ({trial_time/60:.1f} min)")

            # Save intermediate results
            df = pd.DataFrame(phase1_results)
            df = df.sort_values('best_val_acc', ascending=False)
            df.to_csv(os.path.join(RESULTS, 'phase1_results.csv'), index=False)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Get top K from Phase 1
    phase1_df = pd.DataFrame(phase1_results)
    phase1_df = phase1_df.sort_values('best_val_acc', ascending=False)
    top_k_configs = phase1_df.head(PHASE2_TOP_K).to_dict('records')

    print("\n" + "-" * 70)
    print(f"Phase 1 Complete! Top {PHASE2_TOP_K} configurations:")
    for i, cfg in enumerate(top_k_configs, 1):
        print(f"  {i}. Best={cfg['best_val_acc']:.2f}% - {params_to_str(cfg)}")

    # ========================================================================
    # PHASE 2: Validation (Top 5 x 100 epochs)
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 2: Validation (Top {PHASE2_TOP_K} x {PHASE2_EPOCHS} epochs)")
    print("=" * 70)

    phase2_results = []

    for trial_num, cfg in enumerate(top_k_configs, 1):
        params = {k: cfg[k] for k in SEARCH_SPACE.keys()}
        print(f"\n[Phase 2 - Config {trial_num}/{PHASE2_TOP_K}]")
        print(f"  {params_to_str(params)}")
        print(f"  (Phase 1 best: {cfg['best_val_acc']:.2f}%)")

        trial_start = time.time()

        try:
            best_acc, final_acc, history = train_one_trial(
                params, train_data, val_data, PHASE2_EPOCHS,
                trial_name=f"P2-{trial_num}/{PHASE2_TOP_K}"
            )
            trial_time = time.time() - trial_start

            result = {
                'config_rank': trial_num,
                'phase': 2,
                'epochs': PHASE2_EPOCHS,
                'best_val_acc': best_acc,
                'final_val_acc': final_acc,
                'phase1_best_acc': cfg['best_val_acc'],
                'trial_time_sec': trial_time,
                **params
            }
            phase2_results.append(result)

            print(f"  Result: Best={best_acc:.2f}%, Final={final_acc:.2f}% ({trial_time/60:.1f} min)")

            # Save history
            hist_df = pd.DataFrame(history)
            hist_df.to_csv(os.path.join(RESULTS, f'phase2_config{trial_num}_history.csv'), index=False)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save Phase 2 results
    phase2_df = pd.DataFrame(phase2_results)
    phase2_df = phase2_df.sort_values('best_val_acc', ascending=False)
    phase2_df.to_csv(os.path.join(RESULTS, 'phase2_results.csv'), index=False)

    # Get best config from Phase 2
    best_config = phase2_df.iloc[0].to_dict()
    best_params = {k: best_config[k] for k in SEARCH_SPACE.keys()}

    print("\n" + "-" * 70)
    print(f"Phase 2 Complete! Best configuration:")
    print(f"  Best={best_config['best_val_acc']:.2f}%")
    print(f"  {params_to_str(best_params)}")

    # ========================================================================
    # PHASE 3: Final Training (Best config x 300 epochs)
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 3: Final Training (Best config x {PHASE3_EPOCHS} epochs)")
    print("=" * 70)
    print(f"  {params_to_str(best_params)}")

    trial_start = time.time()

    best_acc, final_acc, history = train_one_trial(
        best_params, train_data, val_data, PHASE3_EPOCHS,
        trial_name=f"P3-Final"
    )
    trial_time = time.time() - trial_start

    phase3_result = {
        'phase': 3,
        'epochs': PHASE3_EPOCHS,
        'best_val_acc': best_acc,
        'final_val_acc': final_acc,
        'phase2_best_acc': best_config['best_val_acc'],
        'trial_time_sec': trial_time,
        **best_params
    }

    print(f"\n  Final Result: Best={best_acc:.2f}%, Final={final_acc:.2f}% ({trial_time/60:.1f} min)")

    # Save Phase 3 results
    phase3_df = pd.DataFrame([phase3_result])
    phase3_df.to_csv(os.path.join(RESULTS, 'phase3_final_result.csv'), index=False)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(RESULTS, 'phase3_training_history.csv'), index=False)

    # Save best config
    best_config_final = {
        'phase3_best_val_acc': best_acc,
        'phase3_final_val_acc': final_acc,
        'phase2_best_val_acc': best_config['best_val_acc'],
        'hyperparameters': best_params,
        'training_epochs': PHASE3_EPOCHS,
    }
    with open(os.path.join(RESULTS, 'best_config_final.json'), 'w') as f:
        json.dump(best_config_final, f, indent=2)

    # ========================================================================
    # Summary
    # ========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("SEARCH COMPLETED")
    print("=" * 70)
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"\nBest Configuration:")
    print(f"  {params_to_str(best_params)}")
    print(f"\nResults:")
    print(f"  Phase 1 ({PHASE1_EPOCHS} epochs): Best from {PHASE1_NUM_TRIALS} trials")
    print(f"  Phase 2 ({PHASE2_EPOCHS} epochs): {best_config['best_val_acc']:.2f}%")
    print(f"  Phase 3 ({PHASE3_EPOCHS} epochs): {best_acc:.2f}%")
    print(f"\nAll results saved to: {RESULTS}")


if __name__ == "__main__":
    main()
