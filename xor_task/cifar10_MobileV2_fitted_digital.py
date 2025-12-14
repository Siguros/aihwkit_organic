# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit baseline: MobileNetV2 CNN with CIFAR10 using Fitted LinearStepDevice.

CIFAR10 dataset on a MobileNetV2 network with configurable digital (FloatingPoint)
and analog (LinearStepDevice) layers. Uses fitted device from Current_LinearStepDevice_with_variation_config.json.
"""
# pylint: disable=invalid-name

# Imports
import os
import json
import gc
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent display issues
import matplotlib.pyplot as plt

# Imports from PyTorch.
import torch
from torch import nn, Tensor, device, no_grad, manual_seed, save
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import datasets, transforms

# Progress bar
from tqdm import tqdm

# For t-SNE analysis
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Imports from aihwkit.
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig, FloatingPointRPUConfig
from aihwkit.simulator.configs import MappingParameter
from aihwkit.simulator.configs.devices import LinearStepDevice


# Device to use
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Experiment configuration
CONFIG_NAME = "MobileNetV2_AllAnalog_except_IO"  # Configuration identifier for this experiment
EXPERIMENT_NAME = "with_noise"  # Options: "with_noise", "no_write_noise"

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", f"MOBILENETV2_{CONFIG_NAME}_300EPOCH_{EXPERIMENT_NAME}")
os.makedirs(RESULTS, exist_ok=True)
# Note: WEIGHT_PATH will be set in main() based on N_EPOCHS
WEIGHT_PATH = None  # Will be set dynamically

# Training parameters
SEED = 1
N_EPOCHS = 300
BATCH_SIZE = 128
LEARNING_RATE = 0.05
MOMENTUM = 0.9  # SGD momentum
WEIGHT_DECAY = 0.0005  # L2 regularization
NESTEROV = True  # Nesterov momentum
WARMUP_RATIO = 0.04  # Warmup ratio
N_CLASSES = 10
NUM_WORKERS = 4  # For faster data loading

# Load Fitted Device Configuration from JSON based on experiment name
if EXPERIMENT_NAME == "no_write_noise":
    CONFIG_FILENAME = 'Current_LinearStepDevice_no_write_noise_config.json'
else:
    CONFIG_FILENAME = 'Current_LinearStepDevice_with_variation_config.json'

DEVICE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)

with open(DEVICE_CONFIG_PATH, 'r') as f:
    device_config = json.load(f)

# Extract device parameters (exclude metadata)
device_params = {k: v for k, v in device_config.items() if not k.startswith('_')}
FITTED_DEVICE = LinearStepDevice(**device_params)

print(f"\n{'='*60}")
print(f"EXPERIMENT: {EXPERIMENT_NAME}")
print(f"{'='*60}")
print(f"Loaded fitted LinearStepDevice config from {CONFIG_FILENAME}:")
print(f"  dw_min: {device_config.get('dw_min', 'N/A'):.6f}")
print(f"  gamma_up: {device_config.get('gamma_up', 'N/A'):.6f}")
print(f"  gamma_down: {device_config.get('gamma_down', 'N/A'):.6f}")
print(f"\n  Write Noise Parameters:")
print(f"    write_noise_std: {device_config.get('write_noise_std', 'N/A'):.6f}")
print(f"    dw_min_std: {device_config.get('dw_min_std', 'N/A'):.6f}")
print(f"    dw_min_dtod: {device_config.get('dw_min_dtod', 'N/A'):.6f}")
print(f"    gamma_up_dtod: {device_config.get('gamma_up_dtod', 'N/A'):.6f}")
print(f"    gamma_down_dtod: {device_config.get('gamma_down_dtod', 'N/A'):.6f}")
print(f"{'='*60}\n")


def create_analog_config():
    """Create analog configuration using fitted LinearStepDevice.

    Returns:
        SingleRPUConfig: Configuration with fitted LinearStepDevice from Current_LinearStepDevice_with_variation_config.json
    """
    # Use SingleRPUConfig with the fitted device
    config = SingleRPUConfig(device=FITTED_DEVICE)

    # Add mapping for larger layers
    config.mapping = MappingParameter(
        weight_scaling_omega=0.6,
        learn_out_scaling=False,
        weight_scaling_lr_compensation=False,
        digital_bias=True,
        weight_scaling_columnwise=False,
        out_scaling_columnwise=False,
        max_input_size=512,
        max_output_size=512
    )

    return config


class ConvBNActivationAnalog(nn.Module):
    """AnalogConv2d(+groups) + BN + optional Activation.
       If use_analog=False -> FloatingPointRPUConfig for digital operation."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1,
                 activation=True, use_analog=True):
        super().__init__()
        padding = kernel_size // 2
        rpu_cfg = create_analog_config() if use_analog else FloatingPointRPUConfig()
        self.conv = AnalogConv2d(in_ch, out_ch,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=padding, groups=groups,
                                 bias=False, rpu_config=rpu_cfg)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidualAnalog(nn.Module):
    """MobileNetV2 inverted residual with linear bottleneck."""

    def __init__(self, in_ch, out_ch, stride, expand_ratio,
                 all_analog=True):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        # All internal conv layers are analog (requirement), only stem/classifier are digital
        use_analog = all_analog

        layers = []
        # 1) Expansion 1x1 (only if expand_ratio != 1)
        if expand_ratio != 1:
            layers.append(ConvBNActivationAnalog(in_ch, hidden_dim, 1,
                                                 activation=True,
                                                 use_analog=use_analog))
        else:
            hidden_dim = in_ch

        # 2) Depthwise 3x3
        layers.append(ConvBNActivationAnalog(hidden_dim, hidden_dim, 3, stride=stride,
                                             groups=hidden_dim, activation=True,
                                             use_analog=use_analog))  # depthwise

        # 3) Projection 1x1 (linear bottleneck: activation=False)
        layers.append(ConvBNActivationAnalog(hidden_dim, out_ch, 1,
                                             activation=False,
                                             use_analog=use_analog))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            out = out + x
        return out


def _make_divisible(v, divisor=8, min_value=None):
    """Make channels divisible by divisor for efficient hardware."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def create_model(width_mult: float = 1.0):
    """
    MobileNetV2 for CIFAR-10 with:
      - stem (input) digital
      - all bottleneck blocks analog (expand/dw/proj)
      - final 1x1 conv analog
      - classifier (output) digital
    Returns:
        nn.Module
    """

    # ---- Layer policy: input/output digital, others analog ----
    STEM_IS_ANALOG = False
    BLOCKS_ARE_ANALOG = True
    LAST_CONV_IS_ANALOG = True
    CLASSIFIER_IS_ANALOG = False  # output layer is digital

    # ---- CIFAR-10 standard configuration (stride adjusted) ----
    # (t, c, n, s)
    inverted_residual_setting = [
        (1,  16, 1, 1),
        (6,  24, 2, 1),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    input_channel = _make_divisible(32 * width_mult, 8)   # stem out
    last_channel  = _make_divisible(1280 * max(1.0, width_mult), 8)

    # ---------- l0: stem (digital) ----------
    l0_stem = nn.Sequential(
        ConvBNActivationAnalog(3, input_channel, kernel_size=3, stride=1,
                               activation=True, use_analog=STEM_IS_ANALOG)  # stride=1 for CIFAR-10
    )

    # Helper to build stages
    def make_stage(cfg_list, in_ch):
        layers = []
        cur_in = in_ch
        for (t, c, n, s) in cfg_list:
            out_ch = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidualAnalog(
                    cur_in, out_ch, stride=stride, expand_ratio=t,
                    all_analog=BLOCKS_ARE_ANALOG
                ))
                cur_in = out_ch
        return nn.Sequential(*layers), cur_in

    # ---------- l1, l2, l3: bottleneck stages (all analog) ----------
    # Grouped into 3 parts for hook compatibility
    l1_cfg = inverted_residual_setting[0:2]   # up to c=24
    l2_cfg = inverted_residual_setting[2:4]   # c=32..64
    l3_cfg = inverted_residual_setting[4:7]   # c=96..320

    l1_stage, ch_after_l1 = make_stage(l1_cfg, input_channel)
    l2_stage, ch_after_l2 = make_stage(l2_cfg, ch_after_l1)
    l3_stage, ch_after_l3 = make_stage(l3_cfg, ch_after_l2)

    # ---------- l4: last 1x1 conv to 1280 (analog) ----------
    l4_lastconv = nn.Sequential(
        ConvBNActivationAnalog(ch_after_l3, last_channel, kernel_size=1,
                               activation=True, use_analog=LAST_CONV_IS_ANALOG)
    )

    # ---------- l5: classifier (digital) ----------
    if CLASSIFIER_IS_ANALOG:
        fc_rpu = create_analog_config()
    else:
        fc_rpu = FloatingPointRPUConfig()

    l5_fc = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        AnalogLinear(last_channel, N_CLASSES, bias=True, rpu_config=fc_rpu)
    )

    model = nn.Sequential(l0_stem, l1_stage, l2_stage, l3_stage, l4_lastconv, l5_fc)

    # Summary output
    print("\nCreated MobileNetV2 (CIFAR-10) with mixed digital/analog:")
    print(f"  Stem: {'Analog' if STEM_IS_ANALOG else 'Digital'}")
    print("  Bottlenecks: Analog (expand/depthwise/projection)")
    print(f"  Last 1x1 conv: {'Analog' if LAST_CONV_IS_ANALOG else 'Digital'}")
    print(f"  Classifier (FC): {'Analog' if CLASSIFIER_IS_ANALOG else 'Digital'}")
    print("  Inverted residual setting (t,c,n,s):")
    print("   " + " → ".join([f"({t},{c},{n},{s})" for (t,c,n,s) in inverted_residual_setting]) + "\n")

    # Weight initialization (analog tile direct injection)
    initialize_mobilenetv2_weights(model)
    return model


def initialize_mobilenetv2_weights(model):
    """Apply Kaiming initialization to AnalogConv2d/AnalogLinear and BN.
    Works generically for MobileNetV2 with depthwise conv (groups support)."""
    import math
    print("\nApplying Kaiming initialization (generic)...")
    for name, module in model.named_modules():
        if isinstance(module, AnalogConv2d):
            if hasattr(module, 'analog_module'):
                # derive expected conv weight shape and init
                out_c = getattr(module, 'out_channels', None)
                in_c  = getattr(module, 'in_channels', None)
                k = module.kernel_size
                if isinstance(k, tuple):
                    kh, kw = k
                else:
                    kh = kw = k
                tmp_w = torch.empty(out_c, in_c // module.groups, kh, kw)
                nn.init.kaiming_normal_(tmp_w, mode='fan_out', nonlinearity='relu')
                try:
                    w, b = module.analog_module.get_weights()
                    if w.ndim == 2:
                        module.analog_module.set_weights(
                            tmp_w.view(out_c, (in_c // module.groups) * kh * kw), b
                        )
                    else:
                        module.analog_module.set_weights(tmp_w, b)
                    print(f"  Initialized {name}: Conv2d(groups={module.groups})")
                except Exception as e:
                    print(f"  Warning: init fail {name}: {e}")

        elif isinstance(module, nn.BatchNorm2d):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, AnalogLinear):
            if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'get_weights'):
                try:
                    w, b = module.analog_module.get_weights()
                    out_f, in_f = w.shape
                    tmp_w = torch.empty(out_f, in_f)
                    nn.init.kaiming_uniform_(tmp_w, a=math.sqrt(5))
                    module.analog_module.set_weights(tmp_w, b)
                    print(f"  Initialized {name}: Linear({in_f}, {out_f})")
                except Exception as e:
                    print(f"  Warning: init fail {name}: {e}")
    print("Weight initialization completed\n")


def load_images():
    """Load images for train from torchvision datasets with data augmentation.

    Returns:
        Dataset, Dataset: train data and validation data"""
    # Training transforms with basic augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=train_transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=val_transform)
    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False)
    validation_data = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True if USE_CUDA else False)

    return train_data, validation_data


def create_sgd_optimizer(model, learning_rate, momentum=0.9, weight_decay=5e-4):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate
        momentum (float): momentum factor for SGD
        weight_decay (float): weight decay factor

    Returns:
        Optimizer: created analog optimizer
    """
    optimizer = AnalogSGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=NESTEROV
    )
    optimizer.regroup_param_groups(model)

    return optimizer


def train_step(train_data, model, criterion, optimizer, epoch_num):
    """Train network for one epoch.

    Args:
        train_data (DataLoader): Training data loader
        model (nn.Module): Model to be trained
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer
        epoch_num (int): Current epoch number

    Returns:
        nn.Module, Optimizer, float, float: model, optimizer, epoch loss, epoch accuracy
    """
    total_loss = 0
    correct = 0
    total = 0

    model.train()

    # Create progress bar
    desc = f"Epoch {epoch_num}"
    pbar = tqdm(train_data, desc=desc, leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # Add training Tensor to the model (input).
        output = model(images)
        loss = criterion(output, labels)

        # Run training (backward propagation).
        loss.backward()

        # Optimize weights.
        optimizer.step()

        # Statistics
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        current_acc = 100 * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%'
        })

    epoch_loss = total_loss / len(train_data.dataset)
    epoch_acc = 100 * correct / total

    return model, optimizer, epoch_loss, epoch_acc



def test_evaluation(validation_data, model, criterion):
    """Test trained network

    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns:
        nn.Module, float, float, float: model, test epoch loss, test error, and test accuracy
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    # Create progress bar for validation
    pbar = tqdm(validation_data, desc="Validating", leave=False)

    with no_grad():
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            pred = model(images)
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch_max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()

            # Update progress bar
            current_acc = 100 * predicted_ok / total_images
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        epoch_loss = total_loss / len(validation_data.dataset)
        accuracy = predicted_ok / total_images * 100
        error = (1 - predicted_ok / total_images) * 100

    return model, epoch_loss, error, accuracy


def apply_warmup_cosine_lr(optimizer, epoch, total_epochs, base_lr, warmup_ratio=0.0, min_lr=1e-5):
    """Apply learning rate warmup + cosine annealing.

    Args:
        optimizer: SGD optimizer
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        base_lr: Base learning rate
        warmup_ratio: Fraction of epochs for warmup (0.0 = no warmup)
        min_lr: Minimum learning rate
    """
    import math

    warmup_epochs = int(total_epochs * warmup_ratio)

    if epoch <= warmup_epochs:
        # Linear warmup: lr = base_lr * (epoch / warmup_epochs)
        current_lr = base_lr * (epoch / warmup_epochs)
    else:
        # Cosine annealing after warmup
        # Progress through the cosine schedule (0 to π)
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        current_lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr


def main():
    """Train a PyTorch MobileNetV2 model with mixed analog/digital to classify CIFAR10."""
    # Seed
    manual_seed(SEED)

    # Load the images.
    train_data, validation_data = load_images()

    # Make the model
    model = create_model()

    # Initialize weights with Kaiming initialization
    initialize_mobilenetv2_weights(model)

    if USE_CUDA:
        model = model.to(DEVICE)

    print(f"Model moved to {DEVICE}")

    # Count parameters - analog tiles don't register weights as PyTorch parameters
    pytorch_params = sum(p.numel() for p in model.parameters())
    print(f"PyTorch registered parameters: {pytorch_params:,}")
    print("Note: Analog tile weights are stored internally in C++ and not counted here")

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = create_sgd_optimizer(model, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY)

    print(f"\nUsing fitted LinearStepDevice for analog layers (dw_min={device_config.get('dw_min', 'N/A')})")

    best_accuracy = 0
    best_epoch = 0

    # Set weight path based on N_EPOCHS
    global WEIGHT_PATH
    WEIGHT_PATH = os.path.join(RESULTS, f"{CONFIG_NAME}_model_weight_lr{LEARNING_RATE}_{N_EPOCHS}epoch_{EXPERIMENT_NAME}.pth")

    print("\nStarting Fitted LinearStepDevice training on CIFAR10...")
    print("=" * 60)

    # Special case: Save initial model when N_EPOCHS = 0
    if N_EPOCHS == 0:
        save(model.state_dict(), WEIGHT_PATH)
        print(f"\nN_EPOCHS = 0: Initial model saved to {WEIGHT_PATH}")
        print("No training performed.")
        return

    # Training history for saving
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    # Create overall progress bar for epochs
    epoch_pbar = tqdm(range(N_EPOCHS), desc="Overall Progress", position=0)

    for epoch in epoch_pbar:
        # Apply warmup + cosine annealing learning rate schedule
        apply_warmup_cosine_lr(optimizer, epoch + 1, N_EPOCHS, LEARNING_RATE, WARMUP_RATIO)
        current_lr = optimizer.param_groups[0]['lr']

        # Train one epoch
        model, optimizer, train_loss, train_acc = train_step(
            train_data, model, criterion, optimizer, epoch + 1
        )

        # Run validation after each epoch
        model.eval()
        _, val_loss, val_error, val_accuracy = test_evaluation(validation_data, model, criterion)
        model.train()

        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        history['learning_rate'].append(current_lr)

        # Track best accuracy (for logging only, not for saving)
        latest_val_acc = val_accuracy
        if latest_val_acc > best_accuracy:
            best_accuracy = latest_val_acc
            best_epoch = epoch
            # Save best model
            best_model_path = os.path.join(RESULTS, f"{CONFIG_NAME}_best_model_lr{LEARNING_RATE}_{N_EPOCHS}epoch_{EXPERIMENT_NAME}.pth")
            save(model.state_dict(), best_model_path)

        epoch_pbar.set_postfix({
            'Train_Acc': f'{train_acc:.2f}%',
            'Val_Acc': f'{latest_val_acc:.2f}%',
            'Best': f'{best_accuracy:.2f}%'
        })

        # Print detailed progress
        if (epoch + 1) % 1 == 0:
            val_info = f", Val Acc {latest_val_acc:.2f}%"
            tqdm.write(f"Epoch {epoch + 1:3d}: "
                      f"Train Loss {train_loss:.4f} (Acc {train_acc:.2f}%)"
                      f"{val_info}")

    # Save the final epoch model (not best model)
    print("=" * 60)
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}% at epoch {best_epoch + 1}")
    print(f"Final epoch validation accuracy: {latest_val_acc:.2f}%")
    print(f"Saving final epoch model to: {WEIGHT_PATH}")
    save(model.state_dict(), WEIGHT_PATH)
    print(f"✓ Model weights saved (final epoch)")

    # =========================================================================
    # Save Results to Excel and Plot
    # =========================================================================
    print("\n" + "=" * 60)
    print("Saving results...")

    # Save training history to Excel and CSV
    try:
        df = pd.DataFrame(history)
        excel_path = os.path.join(RESULTS, f"{CONFIG_NAME}_training_history_lr{LEARNING_RATE}_{N_EPOCHS}epoch_{EXPERIMENT_NAME}.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"✓ Training history saved to: {excel_path}")
    except Exception as e:
        print(f"✗ Failed to save Excel file: {e}")

    try:
        csv_path = os.path.join(RESULTS, f"{CONFIG_NAME}_training_history_lr{LEARNING_RATE}_{N_EPOCHS}epoch_{EXPERIMENT_NAME}.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Training history saved to: {csv_path}")
    except Exception as e:
        print(f"✗ Failed to save CSV file: {e}")

    # Create training curves plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Loss curves
        axes[0, 0].plot(history['epoch'], history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['epoch'], history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Accuracy curves
        axes[0, 1].plot(history['epoch'], history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(history['epoch'], history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].axhline(y=best_accuracy, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_accuracy:.2f}%')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Learning rate schedule
        axes[1, 0].plot(history['epoch'], history['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule (Warmup + Cosine)')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Train-Val gap (generalization)
        train_val_gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
        axes[1, 1].plot(history['epoch'], train_val_gap, 'm-', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train - Val Accuracy (%)')
        axes[1, 1].set_title('Generalization Gap')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Fitted LinearStepDevice MobileNetV2 CIFAR10 ({N_EPOCHS} epochs) - {EXPERIMENT_NAME}\n'
                     f'Best Val Acc: {best_accuracy:.2f}% @ Epoch {best_epoch + 1}', fontsize=14)
        plt.tight_layout()

        plot_path = os.path.join(RESULTS, f"{CONFIG_NAME}_training_curves_lr{LEARNING_RATE}_{N_EPOCHS}epoch_{EXPERIMENT_NAME}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Training curves saved to: {plot_path}")
    except Exception as e:
        print(f"✗ Failed to save training curves: {e}")
        plt.close('all')

    # =========================================================================
    # Detailed Analysis (t-SNE, Confusion Matrix)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Running detailed analysis (t-SNE, Confusion Matrix)...")

    # Get feature embeddings and predictions
    try:
        model.eval()
        all_features = []
        all_labels = []
        all_preds = []

        # Extract features from the layer before FC (after AdaptiveAvgPool)
        # We'll use a hook to get intermediate features
        features_hook = []

        def hook_fn(module, input, output):
            features_hook.append(output.detach().cpu())

        # Register hook on the flatten layer (before FC)
        # model structure: [l0_stem, l1_stage, l2_stage, l3_stage, l4_lastconv, l5_fc]
        # l5_fc = [AdaptiveAvgPool, Flatten, AnalogLinear]
        hook = model[5][1].register_forward_hook(hook_fn)  # Flatten layer

        with no_grad():
            for images, labels in tqdm(validation_data, desc="Extracting features"):
                images = images.to(DEVICE)
                outputs = model(images)
                _, preds = torch_max(outputs, 1)

                all_labels.extend(labels.numpy())
                all_preds.extend(preds.cpu().numpy())

        hook.remove()

        # Concatenate all features
        all_features = torch.cat(features_hook, dim=0).numpy()
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        print(f"Feature shape: {all_features.shape}")

        # Clear GPU memory
        if USE_CUDA:
            torch.cuda.empty_cache()
        gc.collect()

        # CIFAR10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

        # t-SNE visualization
        print("Running t-SNE (this may take a few minutes)...")
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
        features_2d = tsne.fit_transform(all_features)

        # Plot t-SNE
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # t-SNE colored by true labels
        scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1],
                                    c=all_labels, cmap='tab10', alpha=0.6, s=10)
        axes[0].set_title('t-SNE (True Labels)')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        cbar1 = plt.colorbar(scatter1, ax=axes[0], ticks=range(10))
        cbar1.ax.set_yticklabels(class_names)

        # t-SNE colored by predictions
        scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1],
                                    c=all_preds, cmap='tab10', alpha=0.6, s=10)
        axes[1].set_title('t-SNE (Predictions)')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        cbar2 = plt.colorbar(scatter2, ax=axes[1], ticks=range(10))
        cbar2.ax.set_yticklabels(class_names)

        plt.suptitle(f'Feature Embeddings - Fitted Device MobileNetV2 - {EXPERIMENT_NAME}\nVal Acc: {best_accuracy:.2f}%', fontsize=14)
        plt.tight_layout()

        tsne_path = os.path.join(RESULTS, f"{CONFIG_NAME}_tsne_analysis_lr{LEARNING_RATE}_{N_EPOCHS}epoch_{EXPERIMENT_NAME}.png")
        plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ t-SNE visualization saved to: {tsne_path}")

        # Confusion Matrix
        print("Creating confusion matrix...")
        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=class_names, yticklabels=class_names)
        axes[0].set_title('Confusion Matrix (Counts)')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')

        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                    xticklabels=class_names, yticklabels=class_names)
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')

        plt.suptitle(f'Confusion Matrix - Fitted Device MobileNetV2 - {EXPERIMENT_NAME}\nVal Acc: {best_accuracy:.2f}%', fontsize=14)
        plt.tight_layout()

        cm_path = os.path.join(RESULTS, f"{CONFIG_NAME}_confusion_matrix_lr{LEARNING_RATE}_{N_EPOCHS}epoch_{EXPERIMENT_NAME}.png")
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved to: {cm_path}")

        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
        per_class_df = pd.DataFrame({
            'Class': class_names,
            'Accuracy (%)': per_class_acc,
            'Correct': cm.diagonal(),
            'Total': cm.sum(axis=1)
        })
        per_class_path = os.path.join(RESULTS, f"{CONFIG_NAME}_per_class_accuracy_lr{LEARNING_RATE}_{N_EPOCHS}epoch_{EXPERIMENT_NAME}.xlsx")
        per_class_df.to_excel(per_class_path, index=False)
        print(f"✓ Per-class accuracy saved to: {per_class_path}")

        # Save t-SNE coordinates for further analysis
        tsne_df = pd.DataFrame({
            'tsne_1': features_2d[:, 0],
            'tsne_2': features_2d[:, 1],
            'true_label': all_labels,
            'predicted_label': all_preds,
            'correct': all_labels == all_preds
        })
        tsne_coords_path = os.path.join(RESULTS, f"{CONFIG_NAME}_tsne_coordinates_lr{LEARNING_RATE}_{N_EPOCHS}epoch_{EXPERIMENT_NAME}.xlsx")
        tsne_df.to_excel(tsne_coords_path, index=False)
        print(f"✓ t-SNE coordinates saved to: {tsne_coords_path}")

        # Summary report
        summary = {
            'experiment_name': EXPERIMENT_NAME,
            'experiment': f'Fitted LinearStepDevice MobileNetV2 CIFAR10 - {EXPERIMENT_NAME}',
            'architecture': {
                'model': 'MobileNetV2',
                'initial_channels': 32,
                'final_features': 1280,
                'inverted_residual_blocks': 17,
                'expansion_ratios': [1, 6, 6, 6, 6, 6, 6],
            },
            'epochs': N_EPOCHS,
            'best_val_accuracy': best_accuracy,
            'best_epoch': best_epoch + 1,
            'final_val_accuracy': latest_val_acc,
            'final_train_accuracy': history['train_acc'][-1],
            'device_config': {
                'config_file': CONFIG_FILENAME,
                'dw_min': device_config.get('dw_min'),
                'write_noise_std': device_config.get('write_noise_std'),
                'dw_min_std': device_config.get('dw_min_std'),
                'dw_min_dtod': device_config.get('dw_min_dtod'),
                'gamma_up_dtod': device_config.get('gamma_up_dtod'),
                'gamma_down_dtod': device_config.get('gamma_down_dtod'),
            },
            'hyperparameters': {
                'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE,
                'momentum': MOMENTUM,
                'weight_decay': WEIGHT_DECAY,
                'warmup_ratio': WARMUP_RATIO,
            },
            'per_class_accuracy': {name: float(acc) for name, acc in zip(class_names, per_class_acc)}
        }

        summary_path = os.path.join(RESULTS, f"mobilenetv2_experiment_summary_{N_EPOCHS}epoch_{EXPERIMENT_NAME}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Experiment summary saved to: {summary_path}")

    except Exception as e:
        print(f"✗ Failed during detailed analysis: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')

    print("\n" + "=" * 60)
    print("All results saved to:", RESULTS)
    print("=" * 60)


if __name__ == "__main__":
    main()
